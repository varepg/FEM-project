import calfem.core as cfc
import numpy as np
from gripper import GripperGeometry, GripperMesh
from utils import get_eq
from typing import Tuple
from numpy.typing import NDArray


def stat_temp_dist(
        gripper: GripperGeometry,
        T_inf: float,
        h: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64], GripperMesh]:

    """Solves the stationary temperature distribution of the thermo-mechanical
    gripper.

    ## Parameters:
    - `gripper`: `GripperGeometry` object
    - `T_inf`: surrounding temperature in K
    - `h`: heat flux in W/m2
    
    ## Returns:
    - `a`: node temperatures
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

    return a, r, K, f, mesh