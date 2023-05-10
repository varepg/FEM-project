from __future__ import annotations
import calfem.geometry as cfg
import calfem.mesh as cfm
from enum import Enum, auto
from numpy.typing import NDArray


class GripperMarker(Enum):
    q0 = auto()
    qh = auto()
    qn = auto()
    nylon = auto()
    copper = auto()


class GripperGeometry(cfg.Geometry):
    """Geometry of thermal gripper.

    Constants (subscript `c` for copper and `n` for nylon):
    - Length: `L`
    - Young's modulus: `E_c`, `E_n`
    - Poisson's ratio: `mu_c`, `mu_n`
    - Expansion coefficient: `alpha_c`, `alpha_n`
    - Denisty: `rho_c`, `rho_n`
    - Specific heat: `cp_c`, `cp_n`
    - Thermal conductivity: `k_c`, `k_n`
    """
    L = 0.005
    E_c = 128e9
    E_n = 3.00e9
    mu_c = 0.36
    mu_n = 0.39
    alpha_c = 17.6e-6
    alpha_n = 80e-6
    rho_c = 8930
    rho_n = 1100
    cp_c = 386
    cp_n = 1500
    k_c = 385
    k_n = 0.26

    marker: GripperMarker

    def __init__(self) -> None:
        super().__init__()
        self.marker = GripperMarker()
        self._setup_points()
        self._setup_lines()
        self._setup_surfaces()

    def mesh(self,
             el_type: int,
             el_size_factor: float,
             dofs_per_node: int
        ) -> GripperMesh:

        return GripperMesh(self, el_type, el_size_factor, dofs_per_node)


    def _setup_points(self) -> None:
        """Generates points ordered from 0-19."""

        self.point([0.0, 0.0])                 # point 0
        self.point([0.35*self.L, 0.0])         # point 1
        self.point([0.45*self.L, 0.0])         # point 2
        self.point([0.9*self.L, 0.25*self.L])  # point 3
        self.point([self.L, 0.25*self.L])      # point 4
        self.point([self.L, 0.3*self.L])       # point 5
        self.point([0.9*self.L, 0.3*self.L])   # point 6
        self.point([0.45*self.L, 0.05*self.L]) # point 7
        self.point([0.45*self.L, 0.35*self.L]) # point 8
        self.point([0.4*self.L, 0.4*self.L])   # point 9
        self.point([0.1*self.L, 0.4*self.L])   # point 10
        self.point([0.1*self.L, 0.5*self.L])   # point 11
        self.point([0.0, 0.5*self.L])          # point 12
        self.point([0.0, 0.4*self.L])          # point 13
        self.point([0.0, 0.3*self.L])          # point 14
        self.point([0.1*self.L, 0.3*self.L])   # point 15
        self.point([0.1*self.L, 0.15*self.L])  # point 16
        self.point([0.15*self.L, 0.15*self.L]) # point 17
        self.point([0.15*self.L, 0.3*self.L])  # point 18
        self.point([0.35*self.L, 0.3*self.L])  # point 19

    def _setup_lines(self) -> None:
        """Generates lines between the points, with boundary markers.

        Boundary conditions:
        - lines connected to marker q0 have q = 0
        - lines connected to marker qh have q = h
        - lines connected to marker qn have q = alpha(T - T_inf)
        """

        self.spline([0, 1], marker=self.marker.q0)         # line 0
        self.spline([1, 2], marker=self.marker.q0)         # line 1

        for i in range(2, 12):
            self.spline([i, i+1], marker = self.marker.qn) # line 2 - 12

        self.spline([12, 13], marker=self.marker.qh)       # line 13
        self.spline([13, 14], marker=self.marker.q0)       # line 14

        for i in range(14, 19):
            self.spline([i, i+1])                          # line 14 - 18

        self.spline([19, 1])                               # line 19
        self.spline([14, 0], marker=self.marker.q0)        # line 20

    def _setup_surfaces(self) -> None:
        self.surface([0, 19, 18, 17, 16, 15, 14, 20], marker = self.marker.nylon)
        self.surface([x for x in range(1, 20)], marker = self.marker.copper)


class GripperMesh(cfm.GmshMesh):
    coords: float
    edof: NDArray
    dofs: NDArray
    bdofs: dict
    el_markers: list[GripperMarker]

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
