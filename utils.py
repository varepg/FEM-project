import calfem.core as cfc
import calfem.utils as cfu
import plantml
from gripper import GripperGeometry, GripperMesh
from numpy.typing import NDArray
from typing import Tuple

import numpy as np

def get_Le(coords, node1, node2):
    x1 = coords[node1 - 1][0]
    x2 = coords[node2 - 1][0]
    y1 = coords[node1 - 1][1]
    y2 = coords[node2 - 1][1]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def get_eq(
        gripper: GripperGeometry,
        mesh: GripperMesh,
        T_inf: float,
        h: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    
    nDofs = np.size(mesh.dofs)
    ex, ey = cfc.coordxtr(mesh.edof, mesh.coords, mesh.dofs)
    K = np.zeros([nDofs,nDofs])
    Kc = np.zeros([nDofs,nDofs])
    f = np.zeros([nDofs,1])

    
    const_matrix = {
        gripper.marker.nylon: gripper.k_n*np.identity(2),
        gripper.marker.copper: gripper.k_c*np.identity(2)
    }
    
    for eltopo, elx, ely, elMarker in zip(mesh.edof, ex, ey, mesh.el_markers):
        Ke = cfc.flw2te(elx, ely, [1], const_matrix[elMarker])
        K = cfc.assem(eltopo, K, Ke)

    for element in mesh.edof:
        Kce = np.zeros((2, 2))
        in_boundary_qn = [False, False, False]
        in_boundary_qh = [False, False, False]
        for i in range(3):
            if element[i] in mesh.bdofs[gripper.marker.qn]:
                in_boundary_qn[i] = True
            if element[i] in mesh.bdofs[gripper.marker.qh]:
                in_boundary_qh[i] = True
        for i in range(3):
            for j in range(i + 1, 3):
                if in_boundary_qn[i] and in_boundary_qn[j]:
                    Le = get_Le(mesh.coords, element[i], element[j])
                    Kce = gripper.alpha_conv_c*Le/6*np.array([[2, -1], [-1, 2]])
                    f[element[i]-1] += gripper.alpha_conv_c*Le*T_inf/2
                    f[element[j]-1] += gripper.alpha_conv_c*Le*T_inf/2
                    Kc = cfc.assem(np.array([element[i], element[j]]), Kc, Kce)
                if in_boundary_qh[i] and in_boundary_qh[j]:
                    Le = get_Le(mesh.coords, element[i], element[j])
                    f[element[i]-1] += h*Le/2
                    f[element[j]-1] += h*Le/2
    K = K + Kc
    return K, f