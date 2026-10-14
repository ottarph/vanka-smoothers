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
import ctypes
import os
import resource
import time

import petsc4py
import petsc4py.lib
from mpi4py import MPI
from petsc4py import PETSc

import networkx as nx
import numpy as np

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl


# From https://gist.github.com/jorgensd/1e3d89b8c46a05fce36eb2fe88cfe9a6
def ghost_by_vertex(msh):
    """Rebuild ``msh`` so a cell is ghosted wherever it shares a vertex."""
    tdim = msh.topology.dim
    num_owned = msh.topology.index_map(tdim).size_local
    cell_to_vertex = msh.topology.connectivity(tdim, 0)
    dest, offset = msh.topology.index_map(0).index_to_dest_ranks()

    # Each owned cell goes to its own rank first, then to every other rank
    # sharing one of its vertices (vectorized over cell-vertex pairs)
    rank, size = msh.comm.rank, msh.comm.size
    nv = cell_to_vertex.links(0).size if num_owned > 0 else 0
    v = cell_to_vertex.array[: num_owned * nv]
    reps = offset[v + 1] - offset[v]
    first = np.repeat(np.cumsum(reps) - reps, reps)
    sharing = dest[np.repeat(offset[v], reps) + np.arange(reps.sum()) - first]
    cell = np.repeat(np.repeat(np.arange(num_owned, dtype=np.int64), nv), reps)
    key = np.unique(cell * size + sharing)
    key = key[key % size != rank]
    extra_cell, extra_rank = key // size, key % size
    counts = 1 + np.bincount(extra_cell, minlength=num_owned)
    offsets = np.zeros(num_owned + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    data = np.full(offsets[-1], rank, dtype=np.int32)
    is_extra = np.ones(offsets[-1], dtype=bool)
    is_extra[offsets[:-1]] = False
    data[is_extra] = extra_rank
    layout = dolfinx.graph.adjacencylist(data, offsets)

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

    # One island per owned vertex: plain vertex-star patches
    if cells_per_island <= 0:
        vertex_island = np.full(num_vertices, -1, dtype=np.int32)
        vertex_island[: vmap.size_local] = np.arange(vmap.size_local, dtype=np.int32)
        return vertex_island, vertex_island[cv]

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


def color_islands(msh, cell_islands, strategy, greedy_strategy="largest_first"):
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
        return nx.coloring.greedy_color(graph, strategy=greedy_strategy)

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


_MATCOLORING_WEIGHTS = {"random": 0, "lexical": 1, "lf": 2, "sl": 3}


def _libpetsc():
    """The libpetsc that petsc4py loaded (MatColoring has no petsc4py binding)."""
    arch = petsc4py.lib.getPathArchPETSc()[1]
    path = os.path.join(petsc4py.get_config()["PETSC_DIR"], arch, "lib", "libpetsc.so")
    return ctypes.CDLL(path)


def petsc_color_islands(msh, cell_islands, mc_type="greedy", weight="sl", distance=1):
    """Distributed coloring of the island graph with PETSc MatColoring.

    Islands sharing a cell are adjacent. With vertex ghosting all cells
    of an island, and so all its neighbours, are local to the island's
    rank, so each rank assembles its own rows of the adjacency matrix.

    With ``distance=1`` islands of one color share no cell. With
    ``distance=2`` they share no vertex either: islands touching at a
    vertex ``u`` both share a cell with the island owning ``u``.

    Returns:
        Color of each local island and the number of colors.
    """
    comm = msh.comm
    tdim = msh.topology.dim
    vmap = msh.topology.index_map(0)
    nv = cell_islands.shape[1]
    cv = msh.topology.connectivity(tdim, 0).array.reshape(-1, nv)
    # Global island numbering: rank r owns rows [offset, offset + num_islands)
    # of the N x N island adjacency matrix
    num_islands = int(cell_islands.max()) + 1 if cell_islands.size > 0 else 0
    offset = comm.exscan(num_islands) or 0
    N = comm.allreduce(num_islands, op=MPI.SUM)

    # Global island of every local vertex. Every local vertex is in a local
    # cell; ghosts (-1 in cell_islands) are then overwritten by their owner.
    gid = dolfinx.la.vector(vmap, dtype=np.float64)
    gid.array[cv.reshape(-1)] = cell_islands.reshape(-1) + offset
    gid.scatter_forward()
    # g[c, i]: global island of vertex i of local cell c
    g = np.rint(gid.array).astype(np.int64)[cv]

    # Two islands are adjacent iff they share a cell, i.e. a cell has vertices
    # in both. Each cell gives all nv x nv (row, col) pairs of its vertex
    # islands, the diagonal included. Vertex ghosting makes every cell of an
    # island local to its rank, so the pairs from local cells give complete
    # rows for the islands owned here; pairs with a row owned elsewhere are
    # dropped (the owner finds them too). After the scatter every vertex has
    # an island, so ``cols >= 0`` is only a guard.
    rows = np.repeat(g, nv, axis=1).reshape(-1)  # g[c, i] repeated nv times
    cols = np.tile(g, (1, nv)).reshape(-1)  # g[c, :] repeated nv times
    keep = (rows >= offset) & (rows < offset + num_islands) & (cols >= 0)

    # Encode each pair as one integer (local row) * N + (global col), so
    # np.unique removes duplicate pairs and sorts by row, then column: this
    # is CSR order. Row starts are found by searching for each local row.
    key = np.unique((rows[keep] - offset) * N + cols[keep])
    indptr = np.searchsorted(key // N, np.arange(num_islands + 1)).astype(PETSc.IntType)
    indices = (key % N).astype(PETSc.IntType)

    # Only the sparsity pattern matters for coloring; the values are ones
    G = PETSc.Mat().createAIJ(
        ((num_islands, N), (num_islands, N)),
        csr=(indptr, indices, np.ones(indices.size)),
        comm=comm,
    )

    # MatColoring has no petsc4py binding, so call the C API through ctypes
    # on the libpetsc petsc4py loaded. PetscInt follows the PETSc build (32 or
    # 64 bit); ISColoringValue is unsigned short (PETSC_IS_COLORING_VALUE_TYPE).
    # Opaque PETSc objects (MatColoring, ISColoring) are plain pointers.
    lib = _libpetsc()
    pint = ctypes.c_int64 if np.dtype(PETSc.IntType) == np.int64 else ctypes.c_int32
    mc, iscoloring = ctypes.c_void_p(), ctypes.c_void_p()
    n, ncolors = pint(), pint()
    values = ctypes.POINTER(ctypes.c_ushort)()

    # Distance-d coloring: islands within d edges of the graph get different
    # colors. ISColoringGetColors returns one color per owned row, i.e. per
    # local island, in local order. Each call returns a PetscErrorCode.
    calls = [
        lambda: lib.MatColoringCreate(ctypes.c_void_p(G.handle), ctypes.byref(mc)),
        lambda: lib.MatColoringSetType(mc, mc_type.encode()),
        lambda: lib.MatColoringSetDistance(mc, pint(distance)),
        lambda: lib.MatColoringSetWeightType(
            mc, ctypes.c_int(_MATCOLORING_WEIGHTS[weight])
        ),
        lambda: lib.MatColoringApply(mc, ctypes.byref(iscoloring)),
        lambda: lib.ISColoringGetColors(
            iscoloring, ctypes.byref(n), ctypes.byref(ncolors), ctypes.byref(values)
        ),
    ]
    for call in calls:
        if call() != 0:
            raise RuntimeError("PETSc MatColoring failed.")
    # Copy the colors out before the ISColoring that owns them is destroyed
    colors = np.ctypeslib.as_array(values, shape=(n.value,)).astype(np.int32)
    lib.ISColoringDestroy(ctypes.byref(iscoloring))
    lib.MatColoringDestroy(ctypes.byref(mc))
    G.destroy()
    assert colors.size == num_islands
    num_colors = int(colors.max()) + 1 if colors.size > 0 else 0
    return colors, comm.allreduce(num_colors, op=MPI.MAX)


class ColorSolver:
    """Exact solve on the islands of one color."""

    _count = 0

    def __init__(
        self,
        V,
        selected,
        bc_dofs,
        solve_dofs=None,
        local_islands=False,
        exterior_bc=False,
    ):
        """Build the submesh of ``selected`` cells and factor its operator.

        Args:
            V: Parent function space.
            selected: Local cells of the islands of this color.
            bc_dofs: Physical Dirichlet dofs (local).
            solve_dofs: Local parent dofs to solve for, the union of the
                island interiors. All other submesh dofs get zero Dirichlet
                data, so each island has its own boundary. If ``None``, every
                dof that no unselected cell touches is solved, i.e. the
                boundary is that of the union of the islands.
            local_islands: Pass this rank's island cells, ghosts included,
                to ``create_submesh``. A ghost cell selected only by the
                island's rank becomes owned there, so each island lies on
                one rank and the color matrix has no coupling between ranks;
                it is then factored per rank (block Jacobi, exact).
                Developer builds reject this owner change.
            exterior_bc: The islands of this color share no vertex, so the
                submesh boundary is exactly the island boundaries. Zero
                Dirichlet data is set on its exterior facets; ``solve_dofs``
                is not needed.
        """
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
        cells = (
            np.unique(selected).astype(np.int32) if local_islands else owned_selected
        )
        self.num_cells = 0
        if msh.comm.allreduce(cells.size, op=MPI.SUM) == 0:
            return

        sub, entity_map, _, _ = dolfinx.mesh.create_submesh(msh, tdim, cells)
        self.num_cells = sub.topology.index_map(tdim).size_global
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

        self.num_owned_sub = Vc.dofmap.index_map.size_local
        if exterior_bc:
            # Islands share no vertex: the submesh boundary is exactly the
            # island boundaries, physical boundary included
            fdim = tdim - 1
            sub.topology.create_connectivity(fdim, tdim)
            facets = dolfinx.mesh.exterior_facet_indices(sub.topology)
            sub_bc = dolfinx.fem.locate_dofs_topological(Vc, fdim, facets)
            sub_unknown = np.ones(self.sub_to_parent.size, dtype=bool)
            sub_unknown[sub_bc] = False
        else:
            if solve_dofs is None:
                # Union boundary: solve dofs that no unselected cell touches
                outside = dolfinx.la.vector(imap, dtype=np.float64)
                unselected = np.flatnonzero(~is_selected[: cmap.size_local])
                outside.array[V.dofmap.list[unselected].reshape(-1)] = 1.0
                outside.scatter_reverse(dolfinx.la.InsertMode.add)
                outside.scatter_forward()
                outside.array[bc_dofs] = 1.0
                unknown = outside.array == 0
            else:
                # Island boundaries: solve the island interiors only
                solve = dolfinx.la.vector(imap, dtype=np.float64)
                solve.array[solve_dofs] = 1.0
                solve.scatter_reverse(dolfinx.la.InsertMode.add)
                solve.scatter_forward()
                unknown = solve.array > 0
            sub_unknown = unknown[self.sub_to_parent]
        self.interior = np.flatnonzero(sub_unknown)
        owned_interior = self.interior[self.interior < self.num_owned_sub]
        if msh.comm.allreduce(owned_interior.size, op=MPI.SUM) == 0:
            self.num_cells = 0
            return

        dtype = PETSc.ScalarType
        u, v = ufl.TrialFunction(Vc), ufl.TestFunction(Vc)
        a = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, dtype=dtype)
        if exterior_bc:
            # Dirichlet assembly keeps the zeroed boundary rows and columns in
            # the sparsity pattern, but each boundary dof belongs to one
            # island, so the pattern stays block-diagonal by island
            bc = dolfinx.fem.dirichletbc(dtype(0), sub_bc, Vc)
            self.A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
            self.A.assemble()
            self.parent_rows = self.sub_to_parent[: self.num_owned_sub]
            self.owned_bc = sub_bc[sub_bc < self.num_owned_sub]
        else:
            # Same-colored islands may touch, and a Dirichlet dof shared by two
            # would couple them in the sparsity pattern (LU fills it in), so
            # copy out the unknowns. Removing stored zeros by value instead
            # (MatEliminateZeros) can leave a nonsymmetric pattern: an exact
            # zero may have a rounding-sized transpose.
            A_sub = dolfinx.fem.petsc.assemble_matrix(a)
            A_sub.assemble()
            rows = Vc.dofmap.index_map.local_to_global(owned_interior.astype(np.int32))
            is_unknown = PETSc.IS().createGeneral(
                rows.astype(PETSc.IntType), comm=sub.comm
            )
            self.A = A_sub.createSubMatrix(is_unknown, is_unknown)
            A_sub.destroy()
            is_unknown.destroy()
            self.parent_rows = self.sub_to_parent[owned_interior]
            self.owned_bc = np.empty(0, dtype=np.int32)
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        # ``w[parent_rows] += e`` in ``solve`` drops repeated indices
        assert np.unique(self.parent_rows).size == self.parent_rows.size

        # No coupling between ranks -> the diagonal blocks are exact
        rstart, rend = self.A.getOwnershipRange()
        _, cols, vals = self.A.getValuesCSR()
        off_rank = np.count_nonzero(((cols < rstart) | (cols >= rend)) & (vals != 0))
        self.rank_local = msh.comm.allreduce(int(off_rank), op=MPI.SUM) == 0

        self.ksp = PETSc.KSP().create(sub.comm)
        self.ksp.setOperators(self.A)
        # self.ksp.setType("cg")
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        options = PETSc.Options()
        prefix = f"vanka_color{ColorSolver._count}_"
        ColorSolver._count += 1
        opts = {}
        if self.rank_local:
            pc.setType("bjacobi")
            opts = {"sub_ksp_type": "preonly", "sub_pc_type": "lu"}
        else:
            pc.setType("cholesky")
            # pc.setType("gamg")
            pc.setFactorSolverType("mumps")
        for k, val in opts.items():
            options[prefix + k] = val
        self.ksp.setOptionsPrefix(prefix)
        self.ksp.setFromOptions()
        self.ksp.setErrorIfNotConverged(True)
        self.ksp.setUp()
        self.ksp.setUpOnBlocks()  # factor the blocks now, not at the first solve
        for k in opts:
            options.delValue(prefix + k)
        self.rhs, self.e = self.A.createVecs()

    def solve(self, r, w):
        """Add the correction for residual ``r`` (ghosted, parent) to ``w``.

        ``w`` is a parent work vector, accumulated over owned sub dofs;
        the caller reverse-scatters it.
        """
        self.rhs.array[:] = r[self.parent_rows]
        self.rhs.array[self.owned_bc] = 0
        self.ksp.solve(self.rhs, self.e)
        w[self.parent_rows] += self.e.array_r

    def destroy(self):
        """Destroy the PETSc objects."""
        if self.num_cells > 0:
            for obj in (self.ksp, self.A, self.rhs, self.e):
                obj.destroy()


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


def create_problem(comm, n, degree, ghost_layers=1, dim=2):
    """Poisson on the unit square (cube), ghosted by ``ghost_layers`` vertex layers.

    Returns:
        Mesh, space, bilinear form, ``A``, ``b``, Dirichlet BC and exact
        solution.
    """
    dtype = PETSc.ScalarType
    if dim == 2:
        msh = dolfinx.mesh.create_unit_square(comm, n, n)
    elif dim == 3:
        msh = dolfinx.mesh.create_unit_cube(comm, n, n, n)
    else:
        raise ValueError(f"Unsupported dimension {dim}.")
    for _ in range(ghost_layers):
        msh = ghost_by_vertex(msh)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(0, tdim)
    msh.topology.create_connectivity(tdim - 1, tdim)

    V = dolfinx.fem.functionspace(msh, ("Lagrange", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.sin(ufl.pi * x[0])
    for i in range(1, dim):
        u_ex *= ufl.sin(ufl.pi * x[i])
    f = -ufl.div(ufl.grad(u_ex))
    a = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, dtype=dtype)
    L = dolfinx.fem.form(ufl.inner(f, v) * ufl.dx, dtype=dtype)
    facets = dolfinx.mesh.locate_entities_boundary(
        msh, tdim - 1, lambda x: np.full(x.shape[1], True)
    )
    bc = dolfinx.fem.dirichletbc(
        dtype(0), dolfinx.fem.locate_dofs_topological(V, tdim - 1, facets), V
    )

    A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L)
    dolfinx.fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc])
    return msh, V, a, A, b, bc, u_ex


def create_macro_vanka(
    V,
    A,
    bc_dofs,
    cells_per_island,
    coloring,
    boundary="island",
    local_islands=False,
    matcoloring=("greedy", "sl"),
    separate=False,
):
    """Build islands, colors and one submesh solver per color.

    Args:
        V: Function space.
        A: Assembled operator.
        bc_dofs: Physical Dirichlet dofs (local).
        cells_per_island: Target owned cells per island; 0 gives one
            vertex star per island.
        coloring: ``"rank-offset"`` or ``"local"`` (see ``color_islands``),
            or ``"petsc"`` (see ``petsc_color_islands``).
        boundary: ``"island"`` puts zero Dirichlet data on each island's
            boundary, ``"color"`` on the boundary of the union of the
            islands of a color.
        local_islands: Keep each island on its own rank, see ``ColorSolver``.
        matcoloring: PETSc ``MatColoring`` type and weight for ``"petsc"``.
        separate: Color so that islands of one color share no vertex
            (distance-2 coloring; requires ``"petsc"``). Each color's
            submesh boundary is then the island boundaries, and
            ``boundary`` makes no difference.

    Returns:
        The PC context, the color solvers and per-color
        ``(islands, cells)`` statistics.
    """
    if boundary not in ("island", "color"):
        raise ValueError(f"Unknown boundary type '{boundary}'.")
    if separate and coloring != "petsc":
        raise ValueError("Separated islands require the 'petsc' coloring.")
    msh = V.mesh
    comm = msh.comm
    tdim = msh.topology.dim
    cmap = msh.topology.index_map(tdim)
    imap = V.dofmap.index_map
    dofs = V.dofmap.list
    _, cell_islands = build_islands(msh, cells_per_island)
    if coloring == "petsc":
        colors, num_colors = petsc_color_islands(
            msh, cell_islands, *matcoloring, distance=2 if separate else 1
        )
    else:
        colors, num_colors = color_islands(msh, cell_islands, coloring)

    # Cells of each island: the stars of its vertices
    num_cells, nv = cell_islands.shape
    isl = cell_islands.reshape(-1)
    cell = np.repeat(np.arange(num_cells), nv)
    pairs = np.unique(np.column_stack([isl[isl >= 0], cell[isl >= 0]]), axis=0)
    split = np.searchsorted(pairs[:, 0], np.arange(colors.size + 1))
    island_cells = [pairs[split[k] : split[k + 1], 1] for k in range(colors.size)]

    # Global number of cells containing each dof. An island's cells are
    # all local, so a dof is interior to it iff it counts them all.
    cells_of_dof = dolfinx.la.vector(imap, dtype=np.float64)
    np.add.at(cells_of_dof.array, dofs[: cmap.size_local].reshape(-1), 1.0)
    cells_of_dof.scatter_reverse(dolfinx.la.InsertMode.add)
    cells_of_dof.scatter_forward()
    is_bc = np.zeros(cells_of_dof.array.size, dtype=bool)
    is_bc[bc_dofs] = True

    def island_interior(cells):
        d, count = np.unique(dofs[cells], return_counts=True)
        d = d[count == cells_of_dof.array[d]]
        return d[~is_bc[d]]

    empty = np.empty(0, dtype=np.int64)
    solvers, stats = [], []
    for c in range(num_colors):
        ks = np.flatnonzero(colors == c)
        cells = np.unique(np.concatenate([island_cells[k] for k in ks] or [empty]))
        solve = None
        if boundary == "island" and not separate:
            solve = np.concatenate(
                [island_interior(island_cells[k]) for k in ks] or [empty]
            )
        solvers.append(
            ColorSolver(
                V, cells.astype(np.int32), bc_dofs, solve, local_islands, separate
            )
        )
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
    return MacroVankaPC(V, A, solvers, bc_dofs), solvers, stats


def main(
    n,
    degree,
    dim,
    cells_per_island,
    coloring,
    boundary,
    local_islands,
    matcoloring,
    separate,
):
    """Solve Poisson and compare MacroVanka with Jacobi."""
    comm = MPI.COMM_WORLD
    dtype = PETSc.ScalarType
    t0 = time.perf_counter()
    _, V, _, A, b, bc, u_ex = create_problem(comm, n, degree, dim=dim)
    t_problem = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
    bc_dofs = bc.dof_indices()[0]
    t0 = time.perf_counter()
    vanka, solvers, stats = create_macro_vanka(
        V,
        A,
        bc_dofs,
        cells_per_island,
        coloring,
        boundary,
        local_islands,
        matcoloring,
        separate,
    )
    t_setup = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
    num_local = sum(s.rank_local for s in solvers if s.num_cells > 0)
    if comm.rank == 0:
        print(
            f"{dim}D P{degree}, n={n}, ranks={comm.size}, "
            f"dofs={V.dofmap.index_map.size_global}, problem setup {t_problem:.1f} s"
        )
        print(f"{len(stats)} colors ({coloring}), (islands, cells) per color: {stats}")
        print(
            f"setup {t_setup:.3f} s; color solves: {num_local} per-rank (block Jacobi), "
            f"{len(vanka.solvers) - num_local} MUMPS"
        )

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
        t0 = time.perf_counter()
        ksp.solve(b, uh.x.petsc_vec)
        t_solve = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
        uh.x.scatter_forward()
        e = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(err), op=MPI.SUM).real)
        if comm.rank == 0:
            print(
                f"CG + {name:10s}: {ksp.getIterationNumber():4d} its, "
                f"{t_solve:.3f} s, L2 error {e:.3e}"
            )
        ksp.destroy()

    # Peak resident memory (kB on Linux), summed over ranks
    rss = comm.allreduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, op=MPI.SUM)
    if comm.rank == 0:
        print(f"peak memory {rss / 1024**2:.2f} GB (all ranks)")

    vanka.free()
    for s in solvers:
        s.destroy()
    for obj in (A, b, zero, x0):
        obj.destroy()
    PETSc.garbage_cleanup(comm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n", type=int, default=32, help="cells per direction")
    parser.add_argument("--degree", type=int, default=2, help="Lagrange degree")
    parser.add_argument(
        "--dim", type=int, choices=[2, 3], default=2, help="unit square or unit cube"
    )
    parser.add_argument(
        "--cells-per-island",
        type=int,
        default=0,
        help="target owned cells per island, from a partition of the per-rank "
        "dual graph; 0 gives one vertex star per island",
    )
    parser.add_argument(
        "--coloring",
        choices=["rank-offset", "local", "petsc"],
        default="rank-offset",
        help="island coloring: networkx per rank with extra colors for islands at "
        "rank boundaries (rank-offset), networkx per rank only (local; same-colored "
        "islands may overlap across ranks), or distributed PETSc MatColoring (petsc)",
    )
    parser.add_argument(
        "--matcoloring-type",
        choices=["greedy", "jp"],
        default="greedy",
        help="PETSc MatColoring algorithm for --coloring petsc: speculative greedy "
        "with conflict repair, or Jones-Plassmann",
    )
    parser.add_argument(
        "--matcoloring-weight",
        choices=["random", "lexical", "lf", "sl"],
        default="sl",
        help="PETSc MatColoring vertex priority for --coloring petsc: random, "
        "index order, largest-first, or smallest-last (approximate in parallel)",
    )
    parser.add_argument(
        "--boundary",
        choices=["island", "color"],
        default="island",
        help="zero Dirichlet data on each island's boundary, or on the boundary "
        "of the union of a color's islands (couples islands that touch)",
    )
    parser.add_argument(
        "--local-islands",
        action="store_true",
        help="pass each rank's island cells, ghosts included, to create_submesh so "
        "every island is owned by one rank and the color solves are per-rank "
        "(block Jacobi); needs a Release build of DOLFINx",
    )
    parser.add_argument(
        "--separate-islands",
        action="store_true",
        help="color so that islands of one color share no vertex (distance-2 "
        "coloring, needs --coloring petsc); the color submesh boundary is then "
        "the island boundaries, set from its exterior facets",
    )
    args = parser.parse_args()
    main(
        args.n,
        args.degree,
        args.dim,
        args.cells_per_island,
        args.coloring,
        args.boundary,
        args.local_islands,
        (args.matcoloring_type, args.matcoloring_weight),
        args.separate_islands,
    )
