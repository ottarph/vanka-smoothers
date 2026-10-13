# Copyright (C) 2026 Jørgen S. Dokken
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""PETSc ``PCPATCH`` star/Vanka smoothers on a DOLFINx mesh, compared with MacroVanka.

``PCPATCH`` needs a ``DMPlex``. It is built here from the DOLFINx topology
(owned and ghost cells, edges, vertices), with a point SF from the DOLFINx
index maps, so plex point ownership matches DOLFINx exactly. The dof
section numbers dofs per plex point, and its section SF maps them to the
rows of the DOLFINx-assembled PETSc matrix. Patch operators are summed from
FFCx element tensors in a ``PCPatchSetComputeOperator`` callback.

Ghosting: ``PCPATCH`` assembles a patch over the cells in the star of
every point of the patch. For ``star`` patches that is the vertex star
(one vertex ghost layer). For ``vanka`` patches (closure of the star) it
reaches two vertex layers, so the mesh is ghosted by vertex twice.
"""

import argparse
import time

from mpi4py import MPI
from petsc4py import PETSc

import cffi
import numpy as np

import dolfinx
import dolfinx.fem.petsc
import ufl
from vanka_poisson import create_macro_vanka, create_problem


def create_plex(msh):
    """Create a DMPlex with the local DOLFINx topology and ownership.

    Points are numbered as cells, then vertices, then entities of
    dimension 1, ..., tdim - 1, each in DOLFINx local order.

    Returns:
        The plex, and the first plex point of each entity dimension.
    """
    comm = msh.comm
    topology = msh.topology
    tdim = topology.dim
    for d in range(1, tdim):
        topology.create_entities(d)
    for d in range(1, tdim + 1):
        topology.create_connectivity(d, d - 1)

    sizes = {}
    for d in range(tdim + 1):
        imap = topology.index_map(d)
        sizes[d] = imap.size_local + imap.num_ghosts
    offsets, num_points = {}, 0
    for d in [tdim, 0, *range(1, tdim)]:
        offsets[d] = num_points
        num_points += sizes[d]

    plex = PETSc.DMPlex().create(comm=comm)
    plex.setDimension(tdim)
    plex.setChart(0, num_points)
    for d in range(1, tdim + 1):
        cone_sizes = np.diff(topology.connectivity(d, d - 1).offsets)
        for e, k in enumerate(cone_sizes):
            plex.setConeSize(offsets[d] + e, k)
    plex.setUp()
    for d in range(1, tdim + 1):
        conn = topology.connectivity(d, d - 1)
        for e in range(sizes[d]):
            plex.setCone(offsets[d] + e, (offsets[d - 1] + conn.links(e)).astype(PETSc.IntType))
    plex.symmetrize()
    plex.stratify()

    # Point SF: each ghost entity points at the owner's plex point
    all_offsets = comm.allgather(offsets)
    leaves, remotes = [], []
    for d in [tdim, 0, *range(1, tdim)]:
        imap = topology.index_map(d)
        starts = np.array(comm.allgather(imap.local_range[0]), dtype=np.int64)
        owners = imap.owners
        leaves.append(offsets[d] + imap.size_local + np.arange(imap.num_ghosts))
        root_offset = np.array([all_offsets[r][d] for r in owners], dtype=np.int64)
        remotes.append(np.column_stack([owners, root_offset + imap.ghosts - starts[owners]]))
    sf = PETSc.SF().create(comm=comm)
    sf.setGraph(
        num_points,
        np.concatenate(leaves).astype(PETSc.IntType),
        np.concatenate(remotes).reshape(-1).astype(PETSc.IntType),
    )
    plex.setPointSF(sf)
    return plex, offsets


def attach_dofs(plex, offsets, V):
    """Attach a dof section and section SF matching ``V`` to ``plex``.

    Returns:
        Map from DOLFINx local dof to section index.
    """
    msh = V.mesh
    comm = msh.comm
    topology = msh.topology
    tdim = topology.dim
    layout = V.dofmap.dof_layout
    dofs = V.dofmap.list
    num_cells = dofs.shape[0]

    # Section: dofs per point, offsets in point order
    num_points = plex.getChart()[1]
    point_dofs = np.zeros(num_points, dtype=np.int64)
    for d in range(tdim + 1):
        imap = topology.index_map(d)
        size = imap.size_local + imap.num_ghosts
        point_dofs[offsets[d] : offsets[d] + size] = len(layout.entity_dofs(d, 0))
    section = PETSc.Section().create(comm=PETSc.COMM_SELF)
    section.setChart(0, num_points)
    for p in np.flatnonzero(point_dofs):
        section.setDof(p, point_dofs[p])
    section.setUp()
    point_offset = np.concatenate([[0], np.cumsum(point_dofs)[:-1]])

    # DOLFINx dof -> section index, through the cell-local entity dofs
    imap = V.dofmap.index_map
    num_local = imap.size_local + imap.num_ghosts
    sec_of_dof = np.full(num_local, -1, dtype=np.int64)
    for d in range(tdim + 1):
        if len(layout.entity_dofs(d, 0)) == 0:
            continue
        if d == tdim:
            entities = np.arange(num_cells).reshape(-1, 1)
        else:
            entities = topology.connectivity(tdim, d).array.reshape(num_cells, -1)
        for i in range(entities.shape[1]):
            pos = layout.entity_dofs(d, i)
            if len(pos) == 0:
                continue
            first = point_offset[offsets[d] + entities[:, i]]
            sec_of_dof[dofs[:, pos]] = first[:, None] + np.arange(len(pos))
    assert np.all(sec_of_dof >= 0) and np.unique(sec_of_dof).size == num_local
    assert section.getOffset(num_points - 1) == point_offset[-1]
    dof_of_sec = np.empty_like(sec_of_dof)
    dof_of_sec[sec_of_dof] = np.arange(num_local)

    # Section SF: section index -> owned row of the DOLFINx matrix
    starts = np.array(comm.allgather(imap.local_range[0]), dtype=np.int64)
    n_own = imap.size_local
    owner = np.full(num_local, comm.rank, dtype=np.int64)
    owner[dof_of_sec >= n_own] = imap.owners[dof_of_sec[dof_of_sec >= n_own] - n_own]
    root = imap.local_to_global(dof_of_sec.astype(np.int32)) - starts[owner]
    sf = PETSc.SF().create(comm=comm)
    sf.setGraph(n_own, None, np.column_stack([owner, root]).reshape(-1).astype(PETSc.IntType))
    plex.setLocalSection(section)
    plex.setSectionSF(sf)
    return sec_of_dof


def element_tensors(a, V):
    """Tabulate the element matrix of ``a`` on every local (owned + ghost) cell."""
    assert not V.element.needs_dof_transformations
    msh = V.mesh
    dtype = PETSc.ScalarType
    ffi = cffi.FFI()
    kernel = getattr(a.ufcx_form.form_integrals[0], f"tabulate_tensor_{np.dtype(dtype).name}")
    num_cells, nd = V.dofmap.list.shape
    Ae = np.zeros((num_cells, nd, nd), dtype=dtype)
    x, x_dofs = msh.geometry.x, msh.geometry.dofmaps[0]
    ctype = {np.float64: "double", np.complex128: "double _Complex"}[dtype]
    for c in range(num_cells):
        coords = np.ascontiguousarray(x[x_dofs[c]])
        kernel(
            ffi.from_buffer(f"{ctype}[]", Ae[c]),
            ffi.NULL,
            ffi.NULL,
            ffi.from_buffer("double[]", coords),
            ffi.NULL,
            ffi.NULL,
            ffi.NULL,
        )
    return Ae


def compute_patch_operator(pc, point, x, mat, cell_is, dofs, dofs_all, Ae, nd):
    """Sum element tensors of the patch cells; negative dofs are skipped."""
    cells = cell_is.getIndices()
    for c, idx in zip(cells, np.asarray(dofs).reshape(-1, nd), strict=True):
        mat.setValues(idx, idx, Ae[c], addv=PETSc.InsertMode.ADD_VALUES)
    mat.assemble()


class PatchSetup:
    """Shared data for configuring ``PCPATCH`` on a KSP."""

    def __init__(self, a, V, bc_dofs):
        """Build the plex, dof section, cell numbering and element tensors."""
        self.plex, offsets = create_plex(V.mesh)
        sec_of_dof = attach_dofs(self.plex, offsets, V)
        num_cells, self.nd = V.dofmap.list.shape
        self.cell_numbering = PETSc.Section().create(comm=PETSc.COMM_SELF)
        self.cell_numbering.setChart(0, num_cells)
        for c in range(num_cells):
            self.cell_numbering.setDof(c, 1)
        self.cell_numbering.setUp()
        self.cell_node_map = sec_of_dof[V.dofmap.list].astype(PETSc.IntType)
        self.num_local = sec_of_dof.size
        n_own = V.dofmap.index_map.size_local
        self.ghost_bcs = sec_of_dof[bc_dofs].astype(PETSc.IntType)
        self.global_bcs = bc_dofs[bc_dofs < n_own].astype(PETSc.IntType)
        self.Ae = element_tensors(a, V)

    def configure(self, ksp, construct_type, local_type):
        """Make ``ksp``'s PC a ``PCPATCH``; return its options (unprefixed)."""
        pc = ksp.getPC()
        pc.setType("patch")
        pc.setDM(self.plex)
        pc.setPatchCellNumbering(self.cell_numbering)
        pc.setPatchDiscretisationInfo(
            [self.plex],
            np.array([1], dtype=PETSc.IntType),
            [self.cell_node_map],
            np.array([0, self.num_local], dtype=PETSc.IntType),
            self.ghost_bcs,
            self.global_bcs,
        )
        pc.setPatchComputeOperator(compute_patch_operator, args=(self.Ae, self.nd))
        return {
            "pc_patch_construct_type": construct_type,
            "pc_patch_construct_dim": 0,
            "pc_patch_local_type": local_type,
            "pc_patch_symmetrise_sweep": local_type == "multiplicative",
            "pc_patch_save_operators": True,
            "pc_patch_dense_inverse": True,
            "pc_patch_sub_mat_type": "seqdense",
        }


def main(n, degree):
    """Compare PCPATCH star/Vanka, MacroVanka and Jacobi on P``degree`` Poisson."""
    comm = MPI.COMM_WORLD
    dtype = PETSc.ScalarType
    msh, V, a, A, b, bc, u_ex = create_problem(comm, n, degree, ghost_layers=2)
    bc_dofs = bc.dof_indices()[0]
    n_own = V.dofmap.index_map.size_local
    patch = PatchSetup(a, V, bc_dofs)

    # Sanity: element tensors through the dofmap reproduce diag(A)
    diag = dolfinx.la.vector(V.dofmap.index_map, dtype=dtype)
    num_owned_cells = msh.topology.index_map(msh.topology.dim).size_local
    np.add.at(
        diag.array,
        V.dofmap.list[:num_owned_cells].reshape(-1),
        np.diagonal(patch.Ae[:num_owned_cells], axis1=1, axis2=2).reshape(-1),
    )
    diag.scatter_reverse(dolfinx.la.InsertMode.add)
    free = np.ones(n_own, dtype=bool)
    free[bc_dofs[bc_dofs < n_own]] = False
    assert np.allclose(diag.array[:n_own][free], A.getDiagonal().array_r[free])

    macro, macro_setup = {}, {}
    for k in (0, 2, 32):
        t0 = time.perf_counter()
        macro[k] = create_macro_vanka(V, A, bc_dofs, k, "local")
        macro_setup[k] = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)

    def configure(ksp, method):
        """Set the PC on ``ksp``; return its options (unprefixed)."""
        pc = ksp.getPC()
        if method == "Jacobi":
            pc.setType("jacobi")
            return {}
        if method.startswith("MacroVanka"):
            pc.setType("python")
            pc.setPythonContext(macro[int(method.split("/")[1])][0])
            return {}
        _, construct, local = method.split()
        return patch.configure(ksp, construct, local)

    counter = iter(range(1000))

    def make(method, ksp_type, extra=None, eigenvalues=False):
        """Create and set up a KSP.

        Options live only until setUp, where PCPATCH reads its sub-solver
        options.
        """
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType(ksp_type)
        opts = configure(ksp, method) | (extra or {})
        prefix = f"cmp{next(counter)}_"
        options = PETSc.Options()
        for k, v in opts.items():
            options[prefix + k] = v
        ksp.setOptionsPrefix(prefix)
        ksp.setFromOptions()
        ksp.setComputeEigenvalues(eigenvalues)
        t0 = time.perf_counter()
        ksp.setUp()
        t_setup = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
        for k in opts:
            options.delValue(prefix + k)
        return ksp, t_setup

    rng = np.random.default_rng(comm.rank)
    x0 = A.createVecRight()
    zero = A.createVecRight()
    zero.zeroEntries()
    uh = dolfinx.fem.Function(V, dtype=dtype)
    err = dolfinx.fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx, dtype=dtype)
    bc_owned = bc_dofs[bc_dofs < n_own]

    methods = [
        "Jacobi",
        "PCPATCH star additive",
        "PCPATCH star multiplicative",
        "PCPATCH vanka additive",
        "PCPATCH vanka multiplicative",
        "MacroVanka/0",
        "MacroVanka/2",
        "MacroVanka/32",
    ]
    rows = []
    for method in methods:
        # Additive methods need damping 1/lambda_max(PA) as smoothers
        scale = 1.0
        if method == "Jacobi" or "additive" in method:
            ksp, _ = make(method, "cg", eigenvalues=True)
            ksp.setTolerances(rtol=1e-30, max_it=40)
            x0.array[:] = 0
            ksp.solve(b, x0)
            scale = 1.0 / np.max(ksp.computeEigenvalues().real)
            ksp.destroy()

        # Smoother: 10 Richardson sweeps on A x = 0 from a random guess
        ksp, _ = make(method, "richardson", {"ksp_richardson_scale": scale})
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        ksp.setTolerances(rtol=0.0, atol=0.0, max_it=10)
        ksp.setConvergenceHistory()
        ksp.setInitialGuessNonzero(True)
        x0.array[:] = rng.random(x0.getLocalSize())
        x0.array[bc_owned] = 0
        ksp.solve(zero, x0)
        hist = ksp.getConvergenceHistory()
        rate = (hist[-1] / hist[0]) ** (1 / (len(hist) - 1))
        ksp.destroy()

        # Preconditioner: CG
        ksp, t_setup = make(method, "cg")
        ksp.setTolerances(rtol=1e-10, max_it=10000)
        ksp.setErrorIfNotConverged(True)
        uh.x.array[:] = 0
        t0 = time.perf_counter()
        ksp.solve(b, uh.x.petsc_vec)
        t_solve = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
        uh.x.scatter_forward()
        e = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(err), op=MPI.SUM).real)
        if method.startswith("MacroVanka"):
            t_setup = macro_setup[int(method.split("/")[1])]
        rows.append((method, rate, ksp.getIterationNumber(), t_setup, t_solve, e))
        ksp.destroy()

    if comm.rank == 0:
        print(f"P{degree}, n={n}, ranks={comm.size}, dofs={V.dofmap.index_map.size_global}")
        for k, (_, _, stats) in macro.items():
            print(f"MacroVanka/{k}: {len(stats)} colors, (islands, cells) per color: {stats}")
        header = ("method", "rate", "CG its", "setup s", "CG s", "L2 err")
        print("{:30s} {:>8s} {:>7s} {:>8s} {:>7s} {:>9s}".format(*header))
        for method, rate, its, ts, tc, e in rows:
            print(f"{method:30s} {rate:8.3f} {its:7d} {ts:8.3f} {tc:7.3f} {e:9.2e}")

    for vanka, solvers, _ in macro.values():
        vanka.free()
        for s in solvers:
            s.destroy()
    patch.plex.destroy()
    for obj in (A, b, x0, zero):
        obj.destroy()
    PETSc.garbage_cleanup(comm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()
    main(args.n, args.degree)
