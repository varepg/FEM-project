import calfem.core as cfc
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
                        el_size_factor=0.05,
                        dofs_per_node=1)        # node temperature

    # get FE formulation of stationary heat equation
    K, f = get_eq(gripper, mesh, T_inf, h)

    # non-existent temperature boundary conditions
    bc = np.array([], 'i')

    # solve for node temperatures and 
    a, r = cfc.solveq(K, f, bc)

    return mesh, a, r, K, f


def transient_temp_dist(gripper: GripperGeometry, T_inf: float, h: float, dt: float):
    mesh, a_stat, _, K, f = stat_temp_dist(gripper, T_inf, h)
    ndofs = np.size(mesh.dofs)

    C = get_C(gripper, mesh)

    idx_max = np.argmax(a_stat)
    pt9a_stat_max = 0.9*a_stat[idx_max]

    a = T_inf * np.ones((ndofs, 1))
    a_new = a

    A = C+dt*K
    dtf = dt*f

    while a_new[idx_max] < pt9a_stat_max:
        b = (C@a[:,-1]).reshape(ndofs, 1) + dtf
        a_new = linalg.solve(A, b)
        a = np.hstack((a, a_new))

    nbr_steps = np.shape(a)[1]

    return a, nbr_steps
