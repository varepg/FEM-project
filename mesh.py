import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu

import numpy as np


def main():
    g = cfg.Geometry()
    L = 0.005 #meter

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
    q0 = 10; qh = 20; qn = 30; nylon = 40; copper = 50

    #generating the lines between points, with boundary markers
    g.spline([0, 1], marker=q0) # line 0
    g.spline([1, 2], marker=q0) # line 1
    for i in range(2, 12):
        g.spline([i, i+1], marker = qn) # line 2 - 12
    g.spline([12, 13], marker=qh) # line 13
    g.spline([13, 14], marker=q0) # line 14
    for i in range(14, 19):
        g.spline([i, i+1], marker = qn) # line 14 - 18
    g.spline([19, 1], marker = qn) # line 19
    g.spline([14, 0], marker=q0) # line 20

    # lines connected to marker q0 have q = 0
    # lines connected to marker qh have q = h
    # lines connected to marker qn have q = alpha(T - T_inf)

    #creating the surfaces
    g.surface([0, 19, 18, 17, 16, 15, 14, 20], marker = nylon) #nylon
    g.surface([x for x in range(1, 20)], marker = copper) #copper

    #creating the mesh
    mesh = cfm.GmshMesh(g)

    mesh.elType = 2          # Degrees of freedom per node.
    mesh.dofsPerNode = 1     # Factor that changes element sizes.
    mesh.elSizeFactor = 0.15 # Element size Factor

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    # coords is a ndofs x 2 -matrix with coordinates for each node
    # edof as a nelm x 3 - matrix which gives the three nodes each element is connected to
    # dofs is a ndofs x 1 - matrix which gives the index of the nodes in coords
    # bdofs is a map showing which splines belong to which markers
    # elementmarkers shows for each element which surface it belongs

    cfv.figure()

    # Draw the mesh.
    cfv.drawMesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofsPerNode,
        el_type=mesh.elType,
        filled=True,
        title="Gripper"
            )

    # Material constants
    E_c = 128; E_n = 3.00 # Young's modulus
    mu_c = 0.36; mu_n = 0.39 # Poisson's ratio
    alpha_c = 17.6*10**(-6); alpha_n = 80*10**(-6) #Expansion coefficient
    rho_c = 8930; rho_n = 1100 # Density
    cp_c = 386; cp_n = 1500 # Specific heat
    k_c = 385; k_n = 0.26 # Thermal conductivity
    
    #Constants given by the problem
    t = 0.005 #thickness
    alpha = 40 # convection coefficient
    h = -10**5 # convection constant
    T_inf = 18 # Surrounding temperature

    #Adding 
    # Create dictionary for the different element properties

    elprop = {}
    elprop[nylon] = k_n*np.identity(2)
    elprop[copper] = k_c*np.identity(2)

    #Solving the problem
    nDofs = np.size(dofs) #total number of dofs
    ex, ey = cfc.coordxtr(edof, coords, dofs) # x- and y-coordinates of the elements

    K = np.zeros([nDofs,nDofs]) # Global stiffness matrix
    Kc = np.zeros([nDofs,nDofs]) # Global convection matrix
    fb = np.zeros((nDofs, 1)) # Global fb matrix
    fc = np.zeros((nDofs, 1)) # Global fc matrix

    # Assemble the part of K that comes from thermal conductivity
    for eltopo, elx, ely, elMarker in zip(edof, ex, ey, elementmarkers):
        #print(eltopo)
        #print(elx)
        #print(ely)
        Ke = cfc.flw2te(elx, ely, [t], elprop[elMarker])
        K = cfc.assem(eltopo, K, Ke)
    
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
                fc[element[0]] += alpha*Le*T_inf/2
                fc[element[1]] += alpha*Le*T_inf/2
            if element[2] in bdofs[qn]: #corner elements can exist
                x1 = coords[element[0] - 1][0]
                x2 = coords[element[2] - 1][0]
                y1 = coords[element[0] - 1][1]
                y2 = coords[element[2] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                Kce += alpha*Le/6*np.array([[2, 0, 1], [0, 0, 0], [1, 0, 2]])
                fc[element[0]] += alpha*Le*T_inf/2
                fc[element[2]] += alpha*Le*T_inf/2
        if element[1] in bdofs[qn]: #corner elements can exist
            if element[2] in bdofs[qn]: #corner elements can exist
                x1 = coords[element[0] - 1][0]
                x2 = coords[element[2] - 1][0]
                y1 = coords[element[0] - 1][1]
                y2 = coords[element[2] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                Kce += alpha*Le/6*np.array([[0, 0, 0], [0, 2, 1], [0, 1, 2]])
                fc[element[1]] += alpha*Le*T_inf/2
                fc[element[2]] += alpha*Le*T_inf/2
        Kc = cfc.assem(element, Kc, Kce)

        if element[0] in bdofs[qh]:
            if element[1] in bdofs[qh]:
                x1 = coords[element[0] - 1][0]
                x2 = coords[element[1] - 1][0]
                y1 = coords[element[0] - 1][1]
                y2 = coords[element[1] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                fb[element[0]] += h*Le/2
                fb[element[1]] += h*Le/2
            if element[2] in bdofs[qh]:
                x1 = coords[element[0] - 1][0]
                x2 = coords[element[2] - 1][0]
                y1 = coords[element[0] - 1][1]
                y2 = coords[element[2] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                fb[element[0]] += h*Le/2
                fb[element[2]] += h*Le/2
        if element[1] in bdofs[qh]:
            if element[2] in bdofs[qh]:
                x1 = coords[element[1] - 1][0]
                x2 = coords[element[2] - 1][0]
                y1 = coords[element[1] - 1][1]
                y2 = coords[element[2] - 1][1]
                Le = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                fb[element[1]] += h*Le/2
                fb[element[2]] += h*Le/2

    # Applying boundary conditions
    bc = np.array([], 'i')
    #bcVal = np.array([], 'f')

    #bc, bcVal = cfu.applybc(bdofs, bc, bcVal, q0, 0.0)
    #bc, bcVal = cfu.applybc(bdofs, bc, bcVal, qh, h)
    #bc, bcVal = cfu.applybc(bdofs, bc, bcVal, qn, alpha*T_inf)

    Kt = K + Kc
    f = fb + fc
    
    #print(fb)
    a = cfc.solveq(K, f, bc)
    #print(bc)
    #print(bcVal)

    #cfc.extract_ed

    

    cfv.draw_geometry(g)
    cfv.showAndWait()

main()