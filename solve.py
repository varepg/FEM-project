import calfem.core as cfc
import calfem.utils as cfu
import numpy as np
from gripper import GripperGeometry, GripperMesh
from utils import get_C, get_eq
from scipy import linalg
from typing import Tuple
from numpy.typing import NDArray


def stat_temp_dist(
        gripper: GripperGeometry,
        T_inf: float,
        h: float
    ) -> Tuple[GripperMesh, NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64], NDArray[np.float64]]:
    """Solves the stationary temperature distribution of the thermo-mechanical
    gripper.

    ## Parameters:
    - `gripper`: `GripperGeometry` object
    - `T_inf`: surrounding temperature in K
    - `h`: heat flux in W/m2

    ## Returns:
    - `mesh`: `GripperMesh` object
    - `a`: node temperatures
    - `r`: reaction force vector
    - `K`: stiffness matrix with convection
    - `f`: force vector
    """

    mesh = gripper.mesh(el_type=2,              # triangular elements
                        el_size_factor=0.01,
                        dofs_per_node=1)        # node temperature

    # get FE formulation of stationary heat equation
    K, f = get_eq(gripper, mesh, T_inf, h)

    # non-existent temperature boundary conditions
    bc = np.array([], 'i')

    # solve for node temperatures and 
    a, r = cfc.solveq(K, f, bc)

    return mesh, a, r, K, f


def transient_temp_step(
        gripper: GripperGeometry,
        mesh: GripperMesh,
        a_old: NDArray[np.float64],
        dt: float,
        K: NDArray[np.float64],
        f: NDArray[np.float64],
        C: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    """Performs one timestep for the transient heat distribution of the thermo-
    mechanical gripper.

    ## Parameters:
    - `gripper`: `GripperGeometry` object
    - `mesh`: `GripperMesh` object
    - `a_old`: node temperatures in the previous step
    - `dt`: size of timestep in s
    - `K`: stiffness matrix with convection
    - `f`: force vector
    - `C`: capacity matrix

    ## Returns:
    - `a_new`: node temperatures in the next step
    """

    A = C+dt*K
    dtf = dt*f

    b = (C@a_old).reshape(np.size(a_old), 1) + dtf
    a_new = linalg.solve(A, b)

    return a_new


def transient_90percent_stat(
        gripper: GripperGeometry,
        T_inf: float, 
        h: float,
        dt: float
    ) -> Tuple[NDArray[np.float64], int]:
    """Solves the transient heat distribution of the thermo-mechanical gripper,
    stopping when the max node temperature has increased to 90% of the increase
    in the stationary case.

    ## Parameters:
    - `gripper`: `GripperGeometry` object
    - `T_inf`: surrounding temperature in K
    - `h`: heat flux in W/m2
    - `dt`: size of timestep in s

    ## Returns:
    - `a`: node temperatures at 90% of stationary increase
    - `nbr_steps`: number of timesteps taken
    """
    mesh, a_stat, _, K, f = stat_temp_dist(gripper, T_inf, h)
    ndofs = np.size(mesh.dofs)

    idx_max = np.argmax(a_stat)
    pt9a_stat_max = 0.9*(a_stat[idx_max] - T_inf) + T_inf

    a = T_inf * np.ones((ndofs, 1))
    a_new = a

    C = get_C(gripper, mesh)

    while a_new[idx_max] < pt9a_stat_max:
        a_new = transient_temp_step(gripper, mesh, a[:,-1], dt, K, f, C)
        a = np.hstack((a, a_new))

    nbr_steps = np.shape(a)[1]

    return a, nbr_steps


def get_displacement(
        gripper: GripperGeometry,
        mesh: GripperMesh,
        stat_a: NDArray,
        T_inf: float
    ) -> Tuple[GripperMesh, NDArray[np.float64], map, NDArray[np.float64]]:
    """Solves the displacement in one quarter of the gripper as a result of the
    stationary thermal distribution. 

    ## Parameters:
    - `gripper`: `GripperGeometry` object
    - `mesh`: `GripperMesh` object with one dof per node
    - `stat_a`: node temperatures in stationary distribution
    - `T_inf`: surrounding temperature in K

    ## Returns:
    - `mesh_2d`: `GripperMesh` object with two dofs per node
    - `u`: node displacements in x- and y-direction
    - `values`: map containing material-specific values
    - `el_dT`: average element temperatures
    """
    mesh_2d = gripper.mesh(el_type=mesh.el_type,             # triangular elements
                        el_size_factor=mesh.el_size_factor,
                        dofs_per_node=2)

    n_nodes = mesh_2d.n_nodes
    n_dofs = mesh_2d.n_dofs
    K = np.zeros([n_dofs,n_dofs])
    f = np.zeros([n_dofs, 1])
    ptype = 2                       # 1 = plane stress, 2 = plane strain
    ep = [ptype, 1]                 # [ptype, thickness]


    values = {
        gripper.marker.nylon: [gripper.alpha_exp_n, gripper.E_n, gripper.nu_n],
        gripper.marker.copper: [gripper.alpha_exp_c, gripper.E_c, gripper.nu_c]
    }

    el_dT = []
    element_props = zip(mesh_2d.edof, mesh.edof, mesh_2d.ex, mesh_2d.ey,
                        mesh_2d.el_markers)
    for eltopo, old_eltopo, elx, ely, el_marker in element_props:
        # Getting material-specific values
        alpha, E, nu = values[el_marker]
        D = cfc.hooke(2, E, nu)[np.ix_([0, 1, 3], [0, 1, 3])]

        # Assembling K
        Ke = cfc.plante(elx, ely, ep, D)
        K = cfc.assem(eltopo, K, Ke)

        # Finding the element's average temperature difference
        delta_T = np.mean((
            stat_a[old_eltopo[0] - 1],
            stat_a[old_eltopo[1] - 1],
            stat_a[old_eltopo[2] - 1])) - T_inf
        el_dT.append(delta_T)

        # Assembling f
        epsilon_therm = alpha*delta_T*D@np.array([[1, 1, 0]]).T
        fe = cfc.plantf(elx, ely, ep, epsilon_therm.T)
        for i in range(len(eltopo)):
            f[eltopo[i] - 1] += fe[i]

    # Setting boundary values at x = 0
    bc = np.array([],'i')
    bc_val = np.array([],'f')

    bc, bc_val = cfu.apply_bc(mesh_2d.bdofs, bc, bc_val, gripper.marker.qh)
    bc, bc_val = cfu.apply_bc(mesh_2d.bdofs, bc, bc_val, gripper.marker.q0_and_clamped)

    # Setting the boundary values at the symmetry lines
    for el in mesh_2d.bdofs[gripper.marker.clamped_x]:
        if el%2 == 1:
            bc = np.append(bc, el)
    for el in mesh_2d.bdofs[gripper.marker.clamped_y]:
        if el%2 == 0:
            bc = np.append(bc, el)

    u, _ = cfc.solveq(K, f, bc)

    return mesh_2d, u, values, el_dT


def get_stress(
        gripper: GripperGeometry,
        mesh: GripperMesh,
        u: NDArray[np.float64],
        values: dict,
        el_dTs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    """Solves the nodal von Mises stress in one quarter of the gripper as a
    result of the displacement caused by the stationary thermal distribution. 

    ## Parameters:
    - `gripper`: `GripperGeometry` object
    - `mesh`: `GripperMesh` object with one dof per node
    - `u`: node displacements in x- and y-direction
    - `values`: map containing material-specific values
    - `el_dTs`: average element temperatures

    ## Returns:
    - `nodal_stresses`: nodal von Mises stresses
    """
    ptype = 2
    ep = [ptype, 1]
    u_edof = cfc.extract_eldisp(mesh.edof, u)

    stresses = []
    element_props = zip(mesh.ex, mesh.ey, u_edof, mesh.el_markers, el_dTs)

    for elx, ely, el_disp, el_marker, delta_T in element_props:
        alpha, E, nu = values[el_marker]
        D = cfc.hooke(2, E, nu)[np.ix_([0, 1, 3], [0, 1, 3])]
        es, et = cfc.plants(elx, ely, ep, D, el_disp)

        sigx, sigy, tauxy = es[0]
        sigx -= alpha*E*delta_T/(1-2*nu) #remove thermal stress
        sigy -= alpha*E*delta_T/(1-2*nu) #remove thermal stress

        sigz = (E * nu / ((1 + nu)*(1 - 2*nu))*(et[0][0] + et[0][1]) 
               - alpha*E*delta_T / (1 - 2*nu))

        stress = np.sqrt(sigx**2 + sigy**2 + sigz**2 - sigx*sigy - sigx*sigz 
                         - sigy*sigz + 3*tauxy**2)
        stresses = np.append(stresses, stress)
        
        nodal_stresses = np.zeros((mesh.n_nodes, 1))

    for node in range(mesh.n_nodes):
        x, y = mesh.dofs[node]
        indexes = np.where(mesh.edof == x)[0]
        mean_stress = np.mean(stresses[indexes])
        nodal_stresses[node] = mean_stress

    return nodal_stresses
