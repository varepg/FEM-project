import calfem.vis_mpl as cfv
import numpy as np
import solve
import utils
import calfem.core as cfc

from gripper import GripperGeometry, GripperMesh
from vis_extra import draw_nodal_values_shaded_clim_cmap
from numpy.typing import NDArray


def plot_temp_dist(a: NDArray[np.float64], gripper: GripperGeometry, mesh: GripperMesh, clim=None) -> None:
    cfv.figure(fig_size=(8,8))
    cfv.draw_mesh(
        coords=mesh.coords,
        edof=mesh.edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        title="Gripper"
    )

    draw_nodal_values_shaded_clim_cmap(a, mesh.coords, mesh.edof, title="Temperature", clim=clim, cmap="YlOrRd")
    cfv.colorbar()
    # cfv.draw_geometry(gripper)
    cfv.show_and_wait()


def main() -> None:
    # boundary convection at L_h
    h = 1e5

    # Surrounding temperature
    T_inf = 18

    # define geometry
    gripper = GripperGeometry()

    # solve and plot stationary temperature distribution
    mesh, a, _, _, _ = solve.stat_temp_dist(gripper, T_inf, h)
    plot_temp_dist(a, gripper, mesh)
    # timesstep
    dt = 0.1
    

    # solve transient temperature distribution
    #     (stops iterating when the max node temperature reaches 90% of the max
    #     stationary node temperature, under the asumption that they are the
    #     same node)
    a_transient, nbr_steps = solve.transient_90percent_stat(gripper, T_inf, h, dt)
    
    print(f"Time to 90% stationary temperature: {nbr_steps*dt}")

    # plot 5 snapshots of the first 3% of the time
    step_size = int(0.03*nbr_steps/4) 
    max_idx = 5*step_size

    T_max = np.max(a_transient[:,max_idx])

    for i in range(0, max_idx+1, step_size):
        plot_temp_dist(a_transient[:,i], gripper, mesh, clim=(T_inf, T_max))
    
    # Get displacement
    mesh3, u, r3, K3, f3, values, el_dT = solve.get_displacement(gripper, mesh, a, T_inf)
    
    
    cfv.figure(fig_size=(8,8))
    cfv.draw_displacements(100 * u, mesh3.coords, mesh3.edof, mesh3.dofs_per_node, mesh3.el_type, draw_undisplaced_mesh=True)
    cfv.show_and_wait()

    # Get von Mises stress

    von_mises_el = solve.get_stress(gripper, mesh3, u, values, el_dT)

    von_mises_nodes = solve.get_nodal_stress(gripper, mesh3, von_mises_el)
    
    cfv.figure(fig_size=(8,8))
    cfv.draw_nodal_values_shaded(
        von_mises_nodes,
        mesh3.coords,
        mesh.edof,
        title="von Mises stress",
        dofs_per_node = mesh3.dofs_per_node)
    cfv.colorbar()
    cfv.show_and_wait()

def test():

    gripper = GripperGeometry()

    mesh = gripper.mesh(el_type=2,              # triangular elements
                        el_size_factor=0.5,
                        dofs_per_node=1)

    mesh2 = gripper.mesh(el_type=2,              # triangular elements
                        el_size_factor=0.5,
                        dofs_per_node=2)
    
    print(np.where(mesh2.edof == 3)[0])

    print(mesh2.edof)

    print(np.shape(mesh2.edof))
    E = 10000
    nu = 0.4
    D1 = utils.get_D_plain_strain(E, nu)
    D2 = cfc.hooke(2, E, nu)
    print(f"D1 - D2: {D1 - D2}")
    #print(f"D2: {D2}")


    
    # print(f"mesh coords: {mesh.coords - mesh2.coords}")

    # print(f"mesh2 coords")

    # y_dofs = mesh.dofs + n_nodes
    # dofs_s = np.hstack((mesh.dofs, y_dofs))
    # edof_s = np.zeros((len(mesh.edof), 6))
    # for i in range(len(dofs_s)):
    #     for j in range(3):
    #         [a,b] = dofs_s[mesh.edof[i][j] - 1]
    #         edof_s[i][2*j] = a
    #         edof_s[i][2*j + 1] = b
    # coords_s = mesh.coords

    # y_dofs = mesh.dofs + n_nodes
    # new_dofs = np.hstack((mesh.dofs, y_dofs))
    # edof_s = np.zeros((len(mesh.edof), 6), 'int')
    # for i in range(len(mesh.edof)):
    #     for j in range(3):
    #         [a,b] = new_dofs[mesh.edof[i][j] - 1]
    #         edof_s[i][2*j] = a
    #         edof_s[i][2*j + 1] = b

    





if __name__ == "__main__":
    main()