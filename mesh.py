import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import plantml
import gripper
import utils
from gripper import GripperGeometry

import numpy as np


def main():
    g = cfg.Geometry()
    L = 0.005 #met

    #generating the points
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

    # Creating marker names instead of numbers
    q0 = 1; qh = 2; qn = 3; nylon = 4; copper = 5

    #generating the lines between points, with boundary markers
    g.spline([0, 1], marker=q0) # line 0
    g.spline([1, 2], marker=q0) # line 1
    for i in range(2, 12):
        g.spline([i, i+1], marker = qn) # line 2 - 12
    g.spline([12, 13], marker=qh) # line 13
    g.spline([13, 14], marker=q0) # line 14
    for i in range(14, 19):
        g.spline([i, i+1]) # line 14 - 18
    g.spline([19, 1]) # line 19
    g.spline([14, 0], marker=q0) # line 20

    # lines connected to marker q0 have q = 0
    # lines connected to marker qh have q = h
    # lines connected to marker qn have q = alpha(T - T_inf)

    #creating the surfaces
    g.surface([0, 19, 18, 17, 16, 15, 14, 20], marker = nylon) #nylon
    g.surface([x for x in range(1, 20)], marker = copper) #copper

    #creating the mesh
    mesh = cfm.GmshMesh(g)

    mesh.el_type = 2          # Degrees of freedom per node.
    mesh.dofs_per_node = 1     # Factor that changes element sizes.
    mesh.el_size_factor = 0.02 # Element size Factor

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    print(bdofs)
    # coords is a ndofs x 2 -matrix with coordinates for each node
    # edof as a nelm x 3 - matrix which gives the three nodes each element is connected to
    # dofs is a ndofs x 1 - matrix which gives the index of the nodes in coords
    # bdofs is a map showing which splines belong to which markers
    # elementmarkers shows for each element which surface it belongs

    # Material constants
    E_c = 128; E_n = 3.00 # Young's modulus
    mu_c = 0.36; mu_n = 0.39 # Poisson's ratio
    alpha_c = 17.6*10**(-6); alpha_n = 80*10**(-6) #Expansion coefficient
    rho_c = 8930; rho_n = 1100 # Density
    cp_c = 386; cp_n = 1500 # Specific heat
    k_c = 385; k_n = 0.26 # Thermal conductivity
    
    #Constants given by the problem
    t = 0.005 #thickness, not needed
    alpha = 40 # convection coefficient
    h = 10**5 # convection constant (ska det vara minus här?)
    T_inf = 18 + 273.15 # Surrounding temperature

    #Adding 
    # Create dictionary for the different element properties

    const_matrix = {}
    const_matrix[nylon] = k_n*np.identity(2)
    const_matrix[copper] = k_c*np.identity(2)

    #Solving the problem
    nDofs = np.size(dofs) #total number of dofs
    ex, ey = cfc.coordxtr(edof, coords, dofs) # x- and y-coordinates of the elements

    K = np.zeros([nDofs,nDofs]) # Global stiffness matrix
    Kc = np.zeros([nDofs,nDofs]) # Global convection matrix
    f = np.zeros([nDofs, 1]) # Global f matrix

    # Assemble the part of K that comes from thermal conductivity
    for eltopo, elx, ely, elMarker in zip(edof, ex, ey, elementmarkers):
        #print(eltopo)
        #print(elx)
        #print(ely)
        Ke = cfc.flw2te(elx, ely, [1], const_matrix[elMarker])
        K = cfc.assem(eltopo, K, Ke)
    cfv.plt.spy(K)
    cfv.plt.show()
    elements_in_fh = []
    # Assemble the part of K (Kc) that comes from thermal convection
    # Also create force vector for h-convection
    for element in edof:
        #print(element)
        Kce = np.zeros((3, 3))
        if element[0] in bdofs[qn]:
            if element[1] in bdofs[qn]:
                x1 = coords[element[0] - 1][0]
                x2 = coords[element[1] - 1][0]
                y1 = coords[element[0] - 1][1]
                y2 = coords[element[1] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                Kce += alpha*Le/6*np.array([[2, 1, 0], [1, 2, 0], [0, 0, 0]])
                f[element[0]-1] += alpha*Le*T_inf/2
                f[element[1]-1] += alpha*Le*T_inf/2
            if element[2] in bdofs[qn]: #corner elements can exist
                x1 = coords[element[0] - 1][0]
                x2 = coords[element[2] - 1][0]
                y1 = coords[element[0] - 1][1]
                y2 = coords[element[2] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                Kce += alpha*Le/6*np.array([[2, 0, 1], [0, 0, 0], [1, 0, 2]])
                f[element[0]-1] += alpha*Le*T_inf/2
                f[element[2]-1] += alpha*Le*T_inf/2
        if element[1] in bdofs[qn]: #corner elements can exist
            if element[2] in bdofs[qn]: #corner elements can exist
                x1 = coords[element[0] - 1][0]
                x2 = coords[element[2] - 1][0]
                y1 = coords[element[0] - 1][1]
                y2 = coords[element[2] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                Kce += alpha*Le/6*np.array([[0, 0, 0], [0, 2, 1], [0, 1, 2]])
                f[element[1]-1] += alpha*Le*T_inf/2
                f[element[2]-1] += alpha*Le*T_inf/2
        Kc = cfc.assem(element, Kc, Kce)
        if element[0] in bdofs[qh]:
            if element[1] in bdofs[qh]:
                x1 = coords[element[0] - 1][0]
                x2 = coords[element[1] - 1][0]
                y1 = coords[element[0] - 1][1]
                y2 = coords[element[1] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                f[element[0]-1] += h*Le/2
                f[element[1]-1] += h*Le/2
                elements_in_fh.append(element[0])
                elements_in_fh.append(element[1])
            if element[2] in bdofs[qh]:
                x1 = coords[element[0] - 1][0]
                x2 = coords[element[2] - 1][0]
                y1 = coords[element[0] - 1][1]
                y2 = coords[element[2] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                f[element[0]-1] += h*Le/2
                f[element[2]-1] += h*Le/2
                elements_in_fh.append(element[0])
                elements_in_fh.append(element[2])
        if element[1] in bdofs[qh]:
            if element[2] in bdofs[qh]:
                x1 = coords[element[1] - 1][0]
                x2 = coords[element[2] - 1][0]
                y1 = coords[element[1] - 1][1]
                y2 = coords[element[2] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                f[element[1]-1] += h*Le/2
                f[element[2]-1] += h*Le/2
                elements_in_fh.append(element[1])
                elements_in_fh.append(element[2])

    grip = gripper.GripperGeometry()
    mesh = grip.mesh(2, 0.05, 1)
    K1, f1 = utils.get_eq(grip, mesh, T_inf, h)

    cfv.plt.spy(Kc)
    cfv.plt.show()
    # Applying boundary conditions
    bc = np.array([], 'i')
    #bcVal = np.array([], 'f')

    #bc, bcVal = cfu.applybc(bdofs, bc, bcVal, q0, 0.0)
    #bc, bcVal = cfu.applybc(bdofs, bc, bcVal, qh, h)
    #bc, bcVal = cfu.applybc(bdofs, bc, bcVal, qn, alpha*T_inf)

    Kt = K + Kc
    cfv.plt.spy(Kt)
    cfv.plt.show()
    print(elements_in_fh)
    #print(f)
    #print("lala")
    #print(f1)
    #print(np.linalg.det(Kt))
    
    
    #print(fb)
    a, r = cfc.solveq(Kt, f, bc)
    #print(a)
    #print(bcVal)

    #ed = cfc.extract_eldisp(edof, a)

    #for i in range(np.shape(ex)[0]):
        #es, et = cfc.flw2ts(ex[i, :], ey[i, :], elprop[elementmarkers[i]], ed[i, :])

    # cfv.figure(fig_size=(8,8))
    # # Draw the mesh.
    # cfv.drawMesh(
    #     coords=coords,
    #     edof=edof,
    #     dofs_per_node=mesh.dofsPerNode,
    #     el_type=mesh.elType,
    #     filled=True,
    #     title="Gripper"
    #         )

    # cfv.draw_nodal_values_shaded(a, coords, edof, title="Temperature")
    # cfv.colorbar()
    # cfv.draw_geometry(g)
    # cfv.showAndWait()

    ## transient heat
    C = np.zeros([nDofs,nDofs]) # Global transient matrix

    transient_scalar = {}
    transient_scalar[nylon] = cp_n*rho_n
    transient_scalar[copper] = cp_c*rho_c


    for eltopo, elx, ely, elMarker in zip(edof, ex, ey, elementmarkers):
        #print(eltopo)
        #print(elx)
        #print(ely)
        Ce = plantml.plantml(elx, ely, transient_scalar[elMarker])
        C = cfc.assem(eltopo, C, Ce)
    
    a0 = T_inf*np.ones((nDofs, 1))
    a_n = a0
    dt = 0.1
    #print(np.linalg.det(Kc))

def main2():
    # Material constants
    E_c = 128; E_n = 3.00 # Young's modulus
    nu_c = 0.36; nu_n = 0.39 # Poisson's ratio
    alpha_c = 17.6*10**(-6); alpha_n = 80*10**(-6) #Expansion coefficient
    rho_c = 8930; rho_n = 1100 # Density
    cp_c = 386; cp_n = 1500 # Specific heat
    k_c = 385; k_n = 0.26 # Thermal conductivity
    t = 1

    # Problem constants
    ptype = 1
    ep = [ptype,t]

    # boundary convection at L_h
    h = 1e5

    # Surrounding temperature
    T_inf = 18

    # define geometry
    gripper = GripperGeometry()

    mesh = gripper.mesh(el_type=2,              # triangular elements
                        el_size_factor=0.02,
                        dofs_per_node=2)         # node temperature
    
    nDofs = np.size(mesh.dofs)
    ex, ey = cfc.coordxtr(mesh.edof, mesh.coords, mesh.dofs)
    K = np.zeros([nDofs,nDofs])

    for eltopo, elx, ely in zip(mesh.edof, ex, ey):
        Ke = cfc.planqe(elx, ely, ep, D)
        cfc.assem(eltopo, K, Ke)



    

main2()