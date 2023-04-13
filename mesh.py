import calfem.geometry as cfg
import calfem.vis as cfv

g = cfg.Geometry()
L = 0.005 #meter

g.point([0.0, 0.0]) # point 0
g.point([0.35*L, 0.0]) # point 1
g.point([0.45*L, 0.0]) # point 2
g.point([0.9*L, 0.25*L]) # point 3
g.point([L, 0.25*L]) # point 4
g.point([L, 0.3*L]) # point 5
g.point([0.9*L, 0.3*L]) # point 6
g.point([0.45*L, 0.05*L]) # point 7
g.point([0.45*L, 0.35*L]) # point 8
g.point([0.4*L, 0.4*L]) # point 9
g.point([0.1*L, 0.4*L]) # point 10
g.point([0.1*L, 0.5*L]) # point 11
g.point([0.0, 0.5*L]) # point 12
g.point([0.0, 0.4*L]) # point 13
g.point([0.0, 0.3*L]) # point 14
g.point([0.1*L, 0.3*L]) # point 15
g.point([0.1*L, 0.15*L]) # point 16
g.point([0.15*L, 0.15*L]) # point 17
g.point([0.15*L, 0.3*L]) # point 18
g.point([0.35*L, 0.3*L]) # point 19

for i in range(0, 19):
    g.spline([i, i+1]) # line 0 - 18
g.spline([19, 1]) # line 19
g.spline([14, 0]) # line 20

nylon = g.surface([0, 19, 18, 17, 16, 15, 14, 20])
copper = g.surface([x for x in range(1, 20)])

cfv.drawGeometry(g)
cfv.showAndWait()