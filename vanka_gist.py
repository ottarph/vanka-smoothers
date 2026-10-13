# Copyright (C) 2026 Jørgen S. Dokken
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Colored macro-patch Vanka smoother for a Poisson problem.

The mesh is ghosted by vertex (``ghost_by_vertex``), so every owned
vertex has its full star locally. Owned vertices are grouped into
islands by partitioning the per-rank dual graph; an island covers the
union of the stars of its vertices. Islands are colored so that islands
of one color share no cell. Each color is one submesh whose patch
problem (zero Dirichlet data on the island boundary) is solved at once
with MUMPS. Colors are swept multiplicatively and symmetrically, which
gives an SPD preconditioner.
"""

import argparse

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import networkx as nx
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc


# From https://gist.github.com/jorgensd/1e3d89b8c46a05fce36eb2fe88cfe9a6
def ghost_by_vertex(msh):
    """Rebuild ``msh`` so a cell is ghosted wherever it shares a vertex."""
    tdim = msh.topology.dim
    num_owned = msh.topology.index_map(tdim).size_local
    cell_to_vertex = msh.topology.connectivity(tdim, 0)
    dest, offset = msh.topology.index_map(0).index_to_dest_ranks()

    rank = msh.comm.rank
    destinations, counts = [], []
    for c in range(num_owned):
        extra = set()
        for v in cell_to_vertex.links(c):
            extra.update(dest[offset[v] : offset[v + 1]])
        extra.discard(rank)
        destinations.append([rank, *sorted(extra)])
        counts.append(1 + len(extra))
    offsets = np.zeros(len(counts) + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    layout = dolfinx.graph.adjacencylist(
        np.fromiter((r for row in destinations for r in row), dtype=np.int32), offsets
    )

    # The cells are handed back in the order they are passed below, so
    # the layout can be returned as it stands
    def keep_partition(comm, nparts, dual_graph, node_weights, edge_weights, ghosting):
        return layout

    # create_mesh numbers the points it is given by position, so supply
    # the owned nodes and refer to them through the numbering of their
    # index map, whose local ranges are that same running count
    geometry = msh.geometry
    node_map = geometry.index_map()
    nodes = node_map.local_to_global(
        geometry.dofmaps[0][:num_owned].reshape(-1).astype(np.int32)
    )
    domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            msh.topology.cell_name(),
            geometry.cmaps[0].degree,
            basix.LagrangeVariant(geometry.cmaps[0].variant),
            shape=(geometry.dim,),
            dtype=geometry.x.dtype,
        )
    )
    return dolfinx.mesh.create_mesh(
        msh.comm,
        nodes.reshape(num_owned, -1),
        domain,
        geometry.x[: node_map.size_local],
        partitioner=keep_partition,
    )


def build_islands(msh, cells_per_island):
    """Group owned vertices into islands and find the cells of each island.

    Returns:
        Island of each local vertex (-1 if not owned), and the island
        ids of each local cell, shape ``(num_cells, num_vertices_per_cell)``
        (-1 where the vertex is not owned).
    """
    tdim = msh.topology.dim
    num_owned_cells = msh.topology.index_map(tdim).size_local
    vmap = msh.topology.index_map(0)
    num_vertices = vmap.size_local + vmap.num_ghosts
    c_to_v = msh.topology.connectivity(tdim, 0)
    nv = c_to_v.links(0).size
    cv = c_to_v.array.reshape(-1, nv)

    # Partition the dual graph of the owned cells
    nparts = max(1, num_owned_cells // cells_per_island)
    if nparts == 1:
        parts = np.zeros(num_owned_cells, dtype=np.int32)
    else:
        gcells = vmap.local_to_global(cv[:num_owned_cells].reshape(-1)).astype(np.int64)
        offsets = np.arange(num_owned_cells + 1, dtype=np.int32) * nv
        dual = dolfinx.mesh.build_dual_graph(
            MPI.COMM_SELF,
            msh.topology.cell_type,
            dolfinx.graph.adjacencylist(gcells, offsets),
        )
        parts = dolfinx.graph.partitioner()(MPI.COMM_SELF, nparts, dual).array

    # Each owned vertex joins the lowest part among its owned cells
    vpart = np.full(num_vertices, np.iinfo(np.int32).max, dtype=np.int64)
    np.minimum.at(vpart, cv[:num_owned_cells].reshape(-1), np.repeat(parts, nv))
    vpart[vmap.size_local :] = -1
    orphans = np.flatnonzero(vpart == np.iinfo(np.int32).max)
    vpart[orphans] = nparts + np.arange(orphans.size)
    owned = vpart >= 0
    vertex_island = np.full(num_vertices, -1, dtype=np.int32)
    vertex_island[owned] = np.unique(vpart[owned], return_inverse=True)[1]
    return vertex_island, vertex_island[cv]


def color_islands(msh, cell_islands, strategy):
    """Color islands so that islands of one color share no cell, globally."""
    comm = msh.comm
    tdim = msh.topology.dim
    cmap = msh.topology.index_map(tdim)
    num_islands = int(cell_islands.max()) + 1 if cell_islands.size > 0 else 0

    # Islands sharing a cell are adjacent
    G = nx.Graph()
    G.add_nodes_from(range(num_islands))
    nv = cell_islands.shape[1]
    for i in range(nv):
        for j in range(i + 1, nv):
            a, b = cell_islands[:, i], cell_islands[:, j]
            keep = (a >= 0) & (b >= 0) & (a != b)
            G.add_edges_from(zip(a[keep].tolist(), b[keep].tolist(), strict=True))

    def greedy(graph):
        return nx.coloring.greedy_color(graph, strategy="largest_first")

    if strategy == "local":
        coloring = greedy(G)
        colors = np.array([coloring[k] for k in range(num_islands)], dtype=np.int32)
        num_colors = comm.allreduce(
            int(colors.max()) + 1 if num_islands > 0 else 0, op=MPI.MAX
        )
        return colors, num_colors

    # Cells that another rank also holds
    dest, offsets = cmap.index_to_dest_ranks()
    shared_cell = np.diff(offsets) > 0
    interface = np.zeros(num_islands, dtype=bool)
    for i in range(nv):
        isl = cell_islands[:, i]
        interface[isl[(isl >= 0) & shared_cell]] = True

    # Ranks holding a common cell must get different rank colors
    neighbours = comm.allgather(np.unique(dest).tolist())
    R = nx.Graph()
    R.add_nodes_from(range(comm.size))
    for r, nbrs in enumerate(neighbours):
        R.add_edges_from((r, s) for s in nbrs if s != r)
    rank_coloring = greedy(R)
    q = rank_coloring[comm.rank]
    num_rank_colors = max(rank_coloring.values()) + 1

    interior_coloring = greedy(G.subgraph(np.flatnonzero(~interface).tolist()))
    interface_coloring = greedy(G.subgraph(np.flatnonzero(interface).tolist()))
    Ki = comm.allreduce(max(interior_coloring.values(), default=-1) + 1, op=MPI.MAX)
    Kf = comm.allreduce(max(interface_coloring.values(), default=-1) + 1, op=MPI.MAX)
    colors = np.empty(num_islands, dtype=np.int32)
    for k, c in interior_coloring.items():
        colors[k] = c
    for k, c in interface_coloring.items():
        colors[k] = Ki + q * Kf + c
    return colors, Ki + num_rank_colors * Kf


class ColorSolver:
    """Exact solve on the union of the islands of one color."""

    def __init__(self, V, selected, bc_dofs):
        """Build the submesh of ``selected`` cells and factor its operator."""
        msh = V.mesh
        tdim = msh.topology.dim
        cmap = msh.topology.index_map(tdim)
        imap = V.dofmap.index_map

        # Make the selection consistent on every rank holding a cell
        marker = dolfinx.la.vector(cmap, dtype=np.float64)
        marker.array[selected] = 1.0
        marker.scatter_reverse(dolfinx.la.InsertMode.add)
        marker.scatter_forward()
        is_selected = marker.array > 0
        owned_selected = np.flatnonzero(is_selected[: cmap.size_local]).astype(np.int32)
        self.num_cells = msh.comm.allreduce(owned_selected.size, op=MPI.SUM)
        if self.num_cells == 0:
            return

        sub, entity_map, _, _ = dolfinx.mesh.create_submesh(msh, tdim, owned_selected)
        Vc = dolfinx.fem.functionspace(sub, V.ufl_element())

        # Sub -> parent dof map through interpolation of global dof numbers,
        # robust to dof permutations on edges
        idx = dolfinx.fem.Function(V, dtype=np.float64)
        idx.x.array[:] = imap.local_to_global(
            np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
        )
        idx_c = dolfinx.fem.Function(Vc, dtype=np.float64)
        smap = sub.topology.index_map(tdim)
        sub_cells = np.arange(smap.size_local + smap.num_ghosts, dtype=np.int32)
        parent_cells = entity_map.sub_topology_to_topology(sub_cells, inverse=False)
        idx_c.interpolate(idx, cells0=parent_cells, cells1=sub_cells)
        self.sub_to_parent = imap.global_to_local(
            np.rint(idx_c.x.array).astype(np.int64)
        )
        assert np.all(self.sub_to_parent >= 0)

        # A parent dof is interior to the islands iff no unselected cell
        # touches it and it is not a physical Dirichlet dof
        outside = dolfinx.la.vector(imap, dtype=np.float64)
        unselected = np.flatnonzero(~is_selected[: cmap.size_local])
        outside.array[V.dofmap.list[unselected].reshape(-1)] = 1.0
        outside.scatter_reverse(dolfinx.la.InsertMode.add)
        outside.scatter_forward()
        outside.array[bc_dofs] = 1.0
        sub_bc = np.flatnonzero(outside.array[self.sub_to_parent] > 0).astype(np.int32)
        self.interior = np.flatnonzero(outside.array[self.sub_to_parent] == 0)

        dtype = PETSc.ScalarType
        u, v = ufl.TrialFunction(Vc), ufl.TestFunction(Vc)
        a = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, dtype=dtype)
        bc = dolfinx.fem.dirichletbc(dtype(0), sub_bc, Vc)
        self.A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
        self.A.assemble()

        self.ksp = PETSc.KSP().create(sub.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        self.ksp.setErrorIfNotConverged(True)
        self.ksp.setUp()

        self.sub_bc = sub_bc
        self.num_owned_sub = Vc.dofmap.index_map.size_local
        self.rhs = dolfinx.fem.Function(Vc, dtype=dtype)
        self.e = dolfinx.fem.Function(Vc, dtype=dtype)

    def solve(self, r, w):
        """Add the correction for residual ``r`` (ghosted, parent) to ``w``.

        ``w`` is a parent work vector, accumulated over owned sub dofs;
        the caller reverse-scatters it.
        """
        self.rhs.x.array[:] = r[self.sub_to_parent]
        self.rhs.x.array[self.sub_bc] = 0
        self.ksp.solve(self.rhs.x.petsc_vec, self.e.x.petsc_vec)
        n = self.num_owned_sub
        w[self.sub_to_parent[:n]] += self.e.x.array[:n]

    def destroy(self):
        """Destroy the PETSc objects."""
        if self.num_cells > 0:
            self.ksp.destroy()
            self.A.destroy()


class MacroVankaPC:
    """Symmetric multiplicative sweep over the colored macro-patches."""

    def __init__(self, V, A, solvers, bc_dofs):
        """Sweep ``solvers`` 0, ..., C-1, ..., 0 with operator ``A``."""
        self.A = A
        self.solvers = [s for s in solvers if s.num_cells > 0]
        C = len(self.solvers)
        self.order = list(range(C)) + list(range(C - 2, -1, -1))
        imap = V.dofmap.index_map
        self.n = imap.size_local
        self.r = dolfinx.la.vector(imap, dtype=PETSc.ScalarType)
        self.w = dolfinx.la.vector(imap, dtype=PETSc.ScalarType)
        self.bc_owned = bc_dofs[bc_dofs < self.n]
        self.Ax = A.createVecLeft()

    def setUp(self, pc):
        """Nothing to set up; the factorizations are built in ``ColorSolver``."""
        pass

    def apply(self, pc, r_in, y):
        """Apply one symmetric sweep, ``y = M^{-1} r_in``."""
        y.zeroEntries()
        for k, c in enumerate(self.order):
            if k == 0:
                self.r.array[: self.n] = r_in.array_r
            else:
                self.A.mult(y, self.Ax)
                self.r.array[: self.n] = r_in.array_r - self.Ax.array_r
            self.r.scatter_forward()
            self.w.array[:] = 0
            self.solvers[c].solve(self.r.array, self.w.array)
            self.w.scatter_reverse(dolfinx.la.InsertMode.add)
            y.array[:] += self.w.array[: self.n]
        # Identity on the Dirichlet rows, matching A
        y.array[self.bc_owned] = r_in.array_r[self.bc_owned]

    def free(self):
        """Destroy the work vector."""
        self.Ax.destroy()


def make_ksp(comm, A, ksp_type, pc_type, ctx=None):
    """Create a KSP without touching the global options database."""
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    if ctx is not None:
        pc.setPythonContext(ctx)
    return ksp


def main(n, degree, cells_per_island, coloring):
    """Solve Poisson and compare MacroVanka with Jacobi."""
    comm = MPI.COMM_WORLD
    dtype = PETSc.ScalarType
    msh0 = dolfinx.mesh.create_unit_square(comm, n, n)
    msh = ghost_by_vertex(msh0)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(0, tdim)
    msh.topology.create_connectivity(tdim - 1, tdim)

    V = dolfinx.fem.functionspace(msh, ("Lagrange", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = -ufl.div(ufl.grad(u_ex))
    a = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, dtype=dtype)
    L = dolfinx.fem.form(ufl.inner(f, v) * ufl.dx, dtype=dtype)
    facets = dolfinx.mesh.locate_entities_boundary(
        msh, tdim - 1, lambda x: np.full(x.shape[1], True)
    )
    bc = dolfinx.fem.dirichletbc(
        dtype(0), dolfinx.fem.locate_dofs_topological(V, tdim - 1, facets), V
    )
    bc_dofs = bc.dof_indices()[0]

    A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L)
    dolfinx.fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc])

    # Islands, colors and one submesh solver per color
    vertex_island, cell_islands = build_islands(msh, cells_per_island)
    colors, num_colors = color_islands(msh, cell_islands, coloring)
    v_to_c = msh.topology.connectivity(0, tdim)
    owned_v = np.flatnonzero(vertex_island >= 0)
    solvers, stats = [], []
    for c in range(num_colors):
        verts = owned_v[colors[vertex_island[owned_v]] == c]
        cells = np.unique(np.concatenate([v_to_c.links(vi) for vi in verts] or [[]]))
        solvers.append(ColorSolver(V, cells.astype(np.int32), bc_dofs))
        stats.append(
            (
                comm.allreduce(int(np.sum(colors == c)), op=MPI.SUM),
                solvers[-1].num_cells,
            )
        )

    # Every free dof must be interior to some island
    covered = dolfinx.la.vector(V.dofmap.index_map, dtype=np.float64)
    for s in solvers:
        if s.num_cells > 0:
            interior = s.interior[s.interior < s.num_owned_sub]
            covered.array[s.sub_to_parent[interior]] += 1
    covered.scatter_reverse(dolfinx.la.InsertMode.add)
    nown = V.dofmap.index_map.size_local
    free = np.ones(nown, dtype=bool)
    free[bc_dofs[bc_dofs < nown]] = False
    assert np.all(covered.array[:nown][free] > 0), "Dof not covered by any island"

    if comm.rank == 0:
        print(
            f"P{degree}, n={n}, ranks={comm.size}, dofs={V.dofmap.index_map.size_global}"
        )
        print(f"{num_colors} colors ({coloring}), (islands, cells) per color: {stats}")

    vanka = MacroVankaPC(V, A, solvers, bc_dofs)

    # Smoother: Richardson from a random guess, b = 0
    rng = np.random.default_rng(comm.rank)
    zero = A.createVecRight()
    zero.zeroEntries()
    x0 = A.createVecRight()

    def richardson(pc_type, ctx=None, scale=1.0):
        ksp = make_ksp(comm, A, "richardson", pc_type, ctx)
        opts = PETSc.Options()
        opts["smoother_ksp_richardson_scale"] = scale
        ksp.setOptionsPrefix("smoother_")
        ksp.setFromOptions()
        opts.delValue("smoother_ksp_richardson_scale")
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        ksp.setTolerances(rtol=0.0, atol=0.0, max_it=10)
        ksp.setConvergenceHistory()
        ksp.setInitialGuessNonzero(True)
        x0.array[:] = rng.random(x0.getLocalSize())
        x0.array[bc.dof_indices()[0][: bc.dof_indices()[1]]] = 0
        ksp.solve(zero, x0)
        hist = ksp.getConvergenceHistory()
        ksp.destroy()
        return hist[1:] / hist[0]

    # Damp Jacobi by 1/lambda_max(D^-1 A)
    ksp = make_ksp(comm, A, "cg", "jacobi")
    ksp.setComputeEigenvalues(True)
    ksp.setTolerances(rtol=1e-30, max_it=40)
    x0.array[:] = 0
    ksp.solve(b, x0)
    lmax = np.max(ksp.computeEigenvalues().real)
    ksp.destroy()

    hist_v = richardson("python", vanka)
    hist_j = richardson("jacobi", scale=1.0 / lmax)
    if comm.rank == 0:
        print("Richardson ||r_k||/||r_0||   MacroVanka      Jacobi")
        for k, (hv, hj) in enumerate(zip(hist_v, hist_j, strict=True), start=1):
            print(f"  sweep {k:2d}               {hv:10.3e}  {hj:10.3e}")

    # Preconditioner: CG
    uh = dolfinx.fem.Function(V, dtype=dtype)
    err = dolfinx.fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx, dtype=dtype)
    for name, pc_type, ctx in [
        ("MacroVanka", "python", vanka),
        ("Jacobi", "jacobi", None),
    ]:
        ksp = make_ksp(comm, A, "cg", pc_type, ctx)
        ksp.setTolerances(rtol=1e-10, max_it=10000)
        ksp.setErrorIfNotConverged(True)
        uh.x.array[:] = 0
        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        e = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(err), op=MPI.SUM).real)
        if comm.rank == 0:
            print(
                f"CG + {name:10s}: {ksp.getIterationNumber():4d} its, L2 error {e:.3e}"
            )
        ksp.destroy()

    vanka.free()
    for s in solvers:
        s.destroy()
    for obj in (A, b, zero, x0):
        obj.destroy()
    PETSc.garbage_cleanup(comm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--cells-per-island", type=int, default=32)
    parser.add_argument(
        "--coloring", choices=["rank-offset", "local"], default="rank-offset"
    )
    args = parser.parse_args()
    main(args.n, args.degree, args.cells_per_island, args.coloring)
