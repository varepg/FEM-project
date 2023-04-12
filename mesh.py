import calfem.geometry as cfg
import calfem.vis as cfv


L = 0.005 # meter

POINTS = [
    [0, 0],
    [0.35*L, 0],
    [0.45*L, 0],
    [0.9*L, 0.25*L],
    [L, 0.25*L],
    [L, 0.3*L],
    [0.9*L, 0.3*L],
    [0.45*L, 0.05*L],
    [0.45*L, 0.35*L],
    [0.4*L, 0.4*L],
    [0.1*L, 0.4*L],
    [0.1*L, 0.5*L],
    [0, 0.5*L],
    [0, 0.3*L],
    [0.1*L, 0.3*L],
    [0.1*L, 0.15*L],
    [0.15*L, 0.15*L],
    [0.15*L, 0.3*L],
    [0.35*L, 0.3*L]
]


g = cfg.Geometry()
map(lambda p: g.point(p), POINTS)

for i in range(0, 18):
    g.spline([i, i+1])
g.spline([18, 1])
g.spline([13, 0])


nylon = g.surface([0, 18, 17, 16, 15, 14, 13, 19])
# copper = g.surface([x for x in range(1, 19)])

cfv.draw_geometry(g)
cfv.show_and_wait()