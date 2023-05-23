import calfem.vis_mpl as cfv
import numpy as np
import solve

from gripper import GripperGeometry, GripperMesh
from vis_extra import draw_nodal_values_shaded_clim_cmap
from numpy.typing import NDArray


def plot_temp_dist(
        a: NDArray[np.float64],
        mesh: GripperMesh,
        clim=None,
        title = None
    )-> None:
    """Plots temperature distribution for a quarter of the gripper."""

    cfv.figure(fig_size=(8,8))

    cfv.draw_mesh(coords=mesh.coords,
                  edof=mesh.edof,
                  dofs_per_node=mesh.dofs_per_node,
                  el_type=mesh.el_type,
                  filled=False,
                  title="Gripper")

    draw_nodal_values_shaded_clim_cmap(a,
                                       mesh.coords,
                                       mesh.edof,
                                       title=title,
                                       clim=clim,
                                       cmap="YlOrRd")

    cfv.colorbar()
    cfv.show_and_wait()


def plot_displacement(
        u: NDArray[np.float64],
        gripper: GripperGeometry,
        mesh: GripperMesh
    ) -> None:
    """Plots displacements for the full gripper."""

    L = gripper.L
    flip_y = np.array([([1, -1]*int(u.size/2))]).T
    flip_x = np.array([([-1, 1]*int(u.size/2))]).T

    draw_quarter(u, mesh.coords, mesh)
    draw_quarter(np.multiply(flip_y, u), [0, L]+[1, -1]*mesh.coords, mesh)
    draw_quarter(np.multiply(flip_y*flip_x, u),[2*L, L]+[-1, -1]*mesh.coords, mesh)
    draw_quarter(np.multiply(flip_x, u), [2*L, 0]+[-1, 1]*mesh.coords, mesh)

    cfv.show_and_wait()


def draw_quarter(
        u: NDArray[np.float64],
        coords: NDArray[np.float64],
        mesh: GripperMesh
    ) -> None:
    """Plots fisplacements for one quarter of the gripper."""

    magnification = 5.0
    cfv.draw_displacements(u, coords,
                           mesh.edof,
                           mesh.dofs_per_node,
                           mesh.el_type,
                           draw_undisplaced_mesh=True,
                           magnfac=magnification,
                           title="Displacement")


def main() -> None:
    # boundary convection at L_h
    h = 1e5

    # Surrounding temperature
    T_inf = 18

    # define geometry
    gripper = GripperGeometry()

    # solve and plot stationary temperature distribution
    mesh, a, _, _, _ = solve.stat_temp_dist(gripper, T_inf, h)
    plot_temp_dist(a, mesh, title="Stationary temperature")
    # timesstep
    dt = 0.1

    # solve transient temperature distribution
    #     (stops iterating when the max node temperature increase reaches 90% of
    #     the max stationary node temperature increase, under the asumption that
    #     they are the same node)
    a_transient, nbr_steps = solve.transient_90percent_stat(gripper, T_inf, h, dt)

    print(f"Time to 90% stationary temperature: {nbr_steps*dt}")

    # plot 5 snapshots of the first 3% of the time
    step_size = int(0.03*nbr_steps/4) 
    max_idx = 4*step_size

    T_max = np.max(a_transient[:,max_idx])

    for i in range(0, max_idx+1, step_size):
        plot_temp_dist(a_transient[:,i],
                       mesh,
                       clim=(T_inf, T_max),
                       title=f"Transient temperature, t = {i*dt:.2}s")

    # Get displacement
    mesh_2d, u, values, el_dT = solve.get_displacement(gripper, mesh, a, T_inf)
    plot_displacement(u, gripper, mesh_2d)

    # Get von Mises stress
    von_mises_nodes = solve.get_stress(gripper, mesh_2d, u, values, el_dT)

    cfv.draw_nodal_values_shaded(
        von_mises_nodes,
        mesh_2d.coords,
        mesh.edof,
        title="von Mises stress",
        dofs_per_node = mesh_2d.dofs_per_node)
    cfv.colorbar()
    cfv.show_and_wait()


if __name__ == "__main__":
    main()