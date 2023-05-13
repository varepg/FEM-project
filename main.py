import calfem.vis_mpl as cfv
import numpy as np
import solve

from gripper import GripperGeometry, GripperMesh
from vis_extra import draw_nodal_values_shaded_clim_cmap
from numpy.typing import NDArray


def plot_temp_dist(a: NDArray[np.float64], mesh: GripperMesh, clim=None) -> None:
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
    #cfv.draw_geometry(g)
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
    plot_temp_dist(a, mesh)

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
        plot_temp_dist(a_transient[:,i], mesh, clim=(T_inf, T_max))


if __name__ == "__main__":
    main()