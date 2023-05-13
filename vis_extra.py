import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import numpy as np


def draw_nodal_values_shaded_clim_cmap(
        values,
        coords,
        edof,
        title=None,
        dofs_per_node=None,
        el_type=None,
        draw_elements=False,
        clim=None,
        cmap=None
    ) -> None:
    """Draws element nodal values as shaded triangles. Element topologies
    supported are triangles, 4-node quads and 8-node quads. Forked from
    calfem.vis_mpl to add support for clim and cmap."""

    edof_tri = cfv.topo_to_tri(edof)

    ax = plt.gca()
    ax.set_aspect('equal')

    if clim:
        (vmin, vmax)  = clim
    else:
        (vmin, vmax) = (None, None)

    x, y = coords.T
    v = np.asarray(values)

    plt.tripcolor(
        x,
        y,
        edof_tri - 1,
        v.ravel(),
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax)

    if draw_elements:
        if dofs_per_node != None and el_type != None:
            cfv.draw_mesh(coords, edof, dofs_per_node,
                      el_type, color=(0.2, 0.2, 0.2))
        else:
            cfv.info("dofs_per_node and el_type must be specified to draw the mesh.")

    if title != None:
        ax.set(title=title)