import calfem.vis_mpl as cfv
import numpy as np
import solve

from gripper import GripperGeometry, GripperMesh
from numpy.typing import NDArray


def plot_stat_temp_dist(a: NDArray[np.float64], mesh: GripperMesh) -> None:
    cfv.figure(fig_size=(8,8))
    cfv.draw_mesh(
        coords=mesh.coords,
        edof=mesh.edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        title="Gripper"
    )

    cfv.draw_nodal_values_shaded(a, mesh.coords, mesh.edof, title="Temperature")
    cfv.colorbar()
    #cfv.draw_geometry(g)
    cfv.show_and_wait()


def main() -> None:
    # boundary convection at L_h
    h = 1e5

    # Surrounding temperature
    T_inf = 18 + 273.15

    # define geometry
    gripper = GripperGeometry()

    # solve and plot stationary temperature distribution
    mesh, a, _, _, _ = solve.stat_temp_dist(gripper, T_inf, h)
    plot_stat_temp_dist(a, mesh)

    # timesstep
    dt = 1.0

    # solve and plot 5 snapshots of transient temperature distribution
    #     (stops iterating when the max node temperature reaches 90% of the max
    #     stationary node temperature, under the asumption that they are the
    #     same node)
    a_transient, nbr_steps = solve.transient_temp_dist(gripper, T_inf, h, dt)
    for i in range(0, nbr_steps, nbr_steps//5):
        plot_stat_temp_dist(a_transient[:,i], mesh)


if __name__ == "__main__":
    main()