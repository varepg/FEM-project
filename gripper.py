from __future__ import annotations
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.core as cfc
from enum import IntEnum, auto
from numpy.typing import NDArray
import numpy as np


class GripperMarker(IntEnum):
    """Markers for materials and boundary conditions."""
    q0 = auto()
    qh = auto()
    qn = auto()
    nylon = auto()
    copper = auto()
    q0_and_clamped = auto()
    clamped_y = auto()
    clamped_x = auto()


class GripperGeometry(cfg.Geometry):
    """Geometry of thermal gripper.

    Constants (subscript `c` for copper and `n` for nylon):
    - Length: `L`
    - Young's modulus: `E_c`, `E_n`
    - Poisson's ratio: `mu_c`, `mu_n`
    - Expansion coefficient: `alpha_exp_c`, `alpha_exp_n`
    - Denisty: `rho_c`, `rho_n`
    - Specific heat: `cp_c`, `cp_n`
    - Thermal conductivity: `k_c`, `k_n`
    - Convection coefficient: `alpha_conv_c`
    """
    L = 0.005
    E_c = 128e9
    E_n = 3.00e9
    nu_c = 0.36
    nu_n = 0.39
    alpha_exp_c = 17.6e-6
    alpha_exp_n = 80e-6
    rho_c = 8930
    rho_n = 1100
    cp_c = 386
    cp_n = 1500
    k_c = 385
    k_n = 0.26
    alpha_conv_c = 40

    marker: GripperMarker

    def __init__(self) -> None:
        super().__init__()
        self.marker = GripperMarker
        self._setup_points()
        self._setup_lines()
        self._setup_surfaces()

    def mesh(self,
             el_type: int,
             el_size_factor: float,
             dofs_per_node: int
        ) -> GripperMesh:
        """Generate GripperMesh from geometry."""
        return GripperMesh(self, el_type, el_size_factor, dofs_per_node)

    def _setup_points(self) -> None:
        """Generates points ordered from 0-19."""

        points = [[0, 0], [0.35, 0], [0.45, 0], [0.9, 0.25], [1, 0.25],     #0-4
          [1, 0.3], [0.9, 0.3], [0.45, 0.05], [0.45, 0.35], [0.4, 0.4],     #5-9
          [0.1, 0.4], [0.1, 0.5], [0, 0.5], [0, 0.4], [0, 0.3],             #10-14
          [0.1, 0.3], [0.1, 0.15], [0.15, 0.15], [0.15, 0.3], [0.35, 0.3]]  #15-19

        #quadrant 1
        for xp, yp in points:
            self.point([xp*self.L, yp*self.L])  #0 - 19

    def _setup_lines(self) -> None:
        """Generates lines between the points, with boundary markers.

        Boundary conditions:
        - lines connected to marker q0 have q = 0
        - lines connected to marker qh have q = h
        - lines connected to marker qn have q = alpha(T - T_inf)
        """

        self.spline([0, 1], marker=self.marker.q0)                # line 0
        self.spline([1, 2], marker=self.marker.q0)                # line 1

        self.spline([2, 3], marker = self.marker.qn)              # line 2
        self.spline([3, 4], marker = self.marker.qn)              # line 3

        self.spline([4, 5], marker = self.marker.clamped_x)       # line 4

        for i in range(5, 11):
            self.spline([i, i+1], marker = self.marker.qn)        # line 5-10
        
        self.spline([11, 12], marker = self.marker.clamped_y)     # line 11

        self.spline([12, 13], marker=self.marker.qh)              # line 12
        self.spline([13, 14], marker=self.marker.q0_and_clamped)  # line 13

        for i in range(14, 19):
            self.spline([i, i+1])                                 # line 14-18

        self.spline([19, 1])                                      # line 19
        self.spline([14, 0], marker=self.marker.q0_and_clamped)   # line 20

    def _setup_surfaces(self) -> None:
        """Generates surfaces for copper and nylon parts respectively."""
        self.surface([0, 19, 18, 17, 16, 15, 14, 20], marker = self.marker.nylon)
        self.surface([x for x in range(1, 20)], marker = self.marker.copper)


class GripperMesh(cfm.GmshMesh):
    """Mesh for thermal gripper.

    Constructed using `GripperGeometry.mesh`.
    """
    coords: float
    edof: NDArray
    dofs: NDArray
    bdofs: dict
    el_markers: list[GripperMarker]
    ex: NDArray
    ey: NDArray
    n_nodes: int
    n_dofs: int
    n_elm: int

    def __init__(self,
                 geometry: GripperGeometry,
                 el_type: int,
                 el_size_factor: float,
                 dofs_per_node: int
        ) -> None:

        super().__init__(
            geometry,
            el_type,
            el_size_factor,
            dofs_per_node,
        )

        (self.coords, self.edof, self.dofs,
        self.bdofs, self.el_markers) = self.create()

        (self.ex, self.ey) = cfc.coordxtr(self.edof, self.coords, self.dofs)

        self.n_nodes = np.shape(self.dofs)[0]
        self.n_dofs = np.size(self.dofs)
        self.n_elm = np.shape(self.edof)[0]
