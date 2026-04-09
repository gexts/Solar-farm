from math import cos, sin
import numpy as np

def site_construction_ns_ft(W_r, A, H, n_row, W_p, beta, L_c):
    """
    Construct the solar site.

    Parameters
    ----------
    W_r : float
        The width of the PV array.
    A : float
        The distance between two rows of the PV array.
    H : float
        The height of the PV array.
    n_row : int
        The number of rows of the PV array.
    W_p : float
        The width of the PV panel.
    beta : float
        The tilt angle of the PV array.
    L_c : float
        The distance between the PV array and the building.

    Returns
    -------
    Coordinate_p : numpy.ndarray
        The coordinates of the PV array.
    """
    # Create the coordinate for each corner of the panel
    Coordinate_p = []
    # for i in range(1, n_row+1): # each row has four corners
    #     Coordinate_p.append([L_c, L_c + (i-1)*(A*cos(beta)+W_r), H])
    #     Coordinate_p.append([L_c + W_p, L_c + (i-1)*(A*cos(beta)+W_r), H])
    #     Coordinate_p.append([L_c + W_p, L_c + A*cos(beta) + (i-1)*(A*cos(beta)+W_r), H + A*sin(beta)])
    #     Coordinate_p.append([L_c, L_c + A*cos(beta) + (i-1)*(A*cos(beta)+W_r), H + A*sin(beta)])

    for i in range(1, n_row+1): # each row has four corners
        Coordinate_p.append([L_c, L_c + A /2 + (i-1)*(A+W_r) - abs(cos(beta)*A/2), H - A /2 * sin(beta)])
        Coordinate_p.append([L_c + W_p, L_c + A /2 + (i-1)*(A+W_r) - abs(cos(beta)*A/2), H - A /2 * sin(beta)])
        Coordinate_p.append([L_c + W_p, L_c + A /2 + (i-1)*(A+W_r) + abs(cos(beta)*A/2), H + A /2 * sin(beta)])
        Coordinate_p.append([L_c, L_c + A /2 + (i-1)*(A+W_r) + abs(cos(beta)*A/2), H + A /2 * sin(beta)])

    Coordinate_p = np.array(Coordinate_p)
    
    return Coordinate_p



def site_construction_ns_sat(W_r,A,H,n_row,W_p,beta,L_c,beta_n, num_int):
    
    # Create the coordinate for each corner of the panel
    Coordinate_p = [[0 for j in range(24*num_int*3)] for i in range(4*n_row)] # 864=288*3
    for j in range(1, 24*num_int+1):
        for i in range(1, n_row+1): # each row has four corners
            # Coordinate_p[i-1+(i-1)*3][0+(j-1)*3] = L_c
            # Coordinate_p[i-1+(i-1)*3][1+(j-1)*3] = L_c + (i-1)*(A*cos(beta_n)+W_r) + A/2*(cos(beta_n)-cos(beta[j-1]))
            # Coordinate_p[i-1+(i-1)*3][2+(j-1)*3] = H - A /2 * (sin(beta[j-1])-sin(beta_n))
            # Coordinate_p[i-1+1+(i-1)*3][0+(j-1)*3] = L_c + W_p
            # Coordinate_p[i-1+1+(i-1)*3][1+(j-1)*3] = L_c  + (i-1)*(A*cos(beta_n)+W_r) + A/2*(cos(beta_n)-cos(beta[j-1]))
            # Coordinate_p[i-1+1+(i-1)*3][2+(j-1)*3] = H - A /2 * (sin(beta[j-1])-sin(beta_n))
            # Coordinate_p[i-1+2+(i-1)*3][0+(j-1)*3] = L_c + W_p
            # Coordinate_p[i-1+2+(i-1)*3][1+(j-1)*3] = L_c + A*cos(beta_n) + (i-1)*(A*cos(beta_n)+W_r) - A/2*(cos(beta_n)-cos(beta[j-1]))
            # Coordinate_p[i-1+2+(i-1)*3][2+(j-1)*3] = H + A*sin(beta_n) + A /2 * (sin(beta[j-1])-sin(beta_n))
            # Coordinate_p[i-1+3+(i-1)*3][0+(j-1)*3] = L_c
            # Coordinate_p[i-1+3+(i-1)*3][1+(j-1)*3] = L_c + A*cos(beta_n) + (i-1)*(A*cos(beta_n)+W_r) - A/2*(cos(beta_n)-cos(beta[j-1]))
            # Coordinate_p[i-1+3+(i-1)*3][2+(j-1)*3] = H + A*sin(beta_n) + A /2 * (sin(beta[j-1])-sin(beta_n))
    
            # lower left corner
            Coordinate_p[i-1+(i-1)*3][0+(j-1)*3] = L_c
            Coordinate_p[i-1+(i-1)*3][1+(j-1)*3] = L_c + A /2 + (i-1)*(A+W_r) - abs(cos(beta[j-1])*A/2)
            Coordinate_p[i-1+(i-1)*3][2+(j-1)*3] = H - A /2 * (sin(beta[j-1]))
            # lower right corner
            Coordinate_p[i-1+1+(i-1)*3][0+(j-1)*3] = L_c + W_p
            Coordinate_p[i-1+1+(i-1)*3][1+(j-1)*3] = L_c + A /2 + (i-1)*(A+W_r) - abs(cos(beta[j-1])*A/2)
            Coordinate_p[i-1+1+(i-1)*3][2+(j-1)*3] = H - A /2 * (sin(beta[j-1]))
            # upper right corner
            Coordinate_p[i-1+2+(i-1)*3][0+(j-1)*3] = L_c + W_p
            Coordinate_p[i-1+2+(i-1)*3][1+(j-1)*3] = L_c + A /2 + (i-1)*(A+W_r) + abs(cos(beta[j-1])*A/2)
            Coordinate_p[i-1+2+(i-1)*3][2+(j-1)*3] = H + A /2 * (sin(beta[j-1]))
            # upper left corner
            Coordinate_p[i-1+3+(i-1)*3][0+(j-1)*3] = L_c
            Coordinate_p[i-1+3+(i-1)*3][1+(j-1)*3] = L_c + A /2 + (i-1)*(A+W_r) + abs(cos(beta[j-1])*A/2)
            Coordinate_p[i-1+3+(i-1)*3][2+(j-1)*3] = H + A /2 * (sin(beta[j-1]))

    Coordinate_p = np.array(Coordinate_p)

    return Coordinate_p



def site_construction_ew_ft(W_r, A, H, n_row, W_p, beta, L_c):

    # Create the coordinate for each corner of the panel
    Coordinate_p = []
    # for i in range(1, n_row+1): # each row has four corners
    #     Coordinate_p.append([L_c, L_c + (i-1)*(A*cos(beta)+W_r), H])
    #     Coordinate_p.append([L_c + W_p, L_c + (i-1)*(A*cos(beta)+W_r), H])
    #     Coordinate_p.append([L_c + W_p, L_c + A*cos(beta) + (i-1)*(A*cos(beta)+W_r), H + A*sin(beta)])
    #     Coordinate_p.append([L_c, L_c + A*cos(beta) + (i-1)*(A*cos(beta)+W_r), H + A*sin(beta)])

    # for i in range(1, n_row+1): # each row has four corners
    #     Coordinate_p.append([L_c + (i-1)*(W_p*cos(beta)+W_r), L_c, H])
    #     Coordinate_p.append([L_c + (i-1)*(W_p*cos(beta)+W_r)+W_p*cos(beta), L_c, H + W_p*sin(beta)])
    #     Coordinate_p.append([L_c + (i-1)*(W_p*cos(beta)+W_r)+W_p*cos(beta), L_c + A, H + W_p*sin(beta)])
    #     Coordinate_p.append([L_c + (i-1)*(W_p*cos(beta)+W_r), L_c + A, H])

    for i in range(1, n_row+1): # each row has four corners
        Coordinate_p.append([L_c + W_p /2 + (i-1)*(W_p+W_r) - abs(cos(beta)*W_p/2), L_c, H + W_p /2 * sin(beta)])
        Coordinate_p.append([L_c + W_p /2 + (i-1)*(W_p+W_r) + abs(cos(beta)*W_p/2), L_c, H - W_p /2 * sin(beta)])
        Coordinate_p.append([L_c + W_p /2 + (i-1)*(W_p+W_r) + abs(cos(beta)*W_p/2), L_c + A, H - W_p /2 * sin(beta)])
        Coordinate_p.append([L_c + W_p /2 + (i-1)*(W_p+W_r) - abs(cos(beta)*W_p/2), L_c + A, H + W_p /2 * sin(beta)])

    Coordinate_p = np.array(Coordinate_p)
    
    return Coordinate_p



def site_construction_ew_sat(W_r,A,H,n_row,W_p,beta,L_c,beta_n, num_int):
    # Create the coordinate for each corner of the p
    Coordinate_p = [[0 for j in range(24*num_int*3)] for i in range(4*n_row)] # 864=288*3
    for j in range(1, 24*num_int+1):
        for i in range(1, n_row+1): # each row has four corners
            # # lower left corner
            # Coordinate_p[i-1+(i-1)*3][0+(j-1)*3] = L_c + (i-1)*(W_p*cos(beta_n)+W_r)+ W_p /2 *(cos(beta_n)-cos(beta[j-1]))
            # Coordinate_p[i-1+(i-1)*3][1+(j-1)*3] = L_c 
            # Coordinate_p[i-1+(i-1)*3][2+(j-1)*3] = H - W_p /2 * (sin(beta[j-1])-sin(beta_n))
            # # lower right corner
            # Coordinate_p[i-1+1+(i-1)*3][0+(j-1)*3] = L_c + (i-1)*(W_p*cos(beta_n)+W_r) + W_p /2 *(cos(beta_n)-cos(beta[j-1]))+ W_p*cos(beta_n)
            # Coordinate_p[i-1+1+(i-1)*3][1+(j-1)*3] = L_c
            # Coordinate_p[i-1+1+(i-1)*3][2+(j-1)*3] = H + W_p /2 * (sin(beta[j-1])-sin(beta_n))
            # # upper right corner
            # Coordinate_p[i-1+2+(i-1)*3][0+(j-1)*3] = L_c + (i-1)*(W_p*cos(beta_n)+W_r) + W_p /2 *(cos(beta_n)-cos(beta[j-1]))+ W_p*cos(beta_n)
            # Coordinate_p[i-1+2+(i-1)*3][1+(j-1)*3] = L_c + A
            # Coordinate_p[i-1+2+(i-1)*3][2+(j-1)*3] = H + W_p /2 * (sin(beta[j-1])-sin(beta_n))
            # # upper left corner
            # Coordinate_p[i-1+3+(i-1)*3][0+(j-1)*3] = L_c + (i-1)*(W_p*cos(beta_n)+W_r)+ W_p /2 *(cos(beta_n)-cos(beta[j-1]))
            # Coordinate_p[i-1+3+(i-1)*3][1+(j-1)*3] = L_c + A
            # Coordinate_p[i-1+3+(i-1)*3][2+(j-1)*3] = H - W_p /2 * (sin(beta[j-1])-sin(beta_n))

            # lower left corner
            Coordinate_p[i-1+(i-1)*3][0+(j-1)*3] = L_c + W_p /2 + (i-1)*(W_p+W_r) - abs(cos(beta[j-1])*W_p/2)
            Coordinate_p[i-1+(i-1)*3][1+(j-1)*3] = L_c 
            Coordinate_p[i-1+(i-1)*3][2+(j-1)*3] = H + W_p /2 * (sin(beta[j-1])-sin(beta_n))
            # lower right corner
            Coordinate_p[i-1+1+(i-1)*3][0+(j-1)*3] = L_c + W_p /2 + (i-1)*(W_p+W_r) + abs(cos(beta[j-1])*W_p/2)
            Coordinate_p[i-1+1+(i-1)*3][1+(j-1)*3] = L_c
            Coordinate_p[i-1+1+(i-1)*3][2+(j-1)*3] = H - W_p /2 * (sin(beta[j-1])-sin(beta_n))
            # upper right corner
            Coordinate_p[i-1+2+(i-1)*3][0+(j-1)*3] = L_c + W_p /2 + (i-1)*(W_p+W_r) + abs(cos(beta[j-1])*W_p/2)
            Coordinate_p[i-1+2+(i-1)*3][1+(j-1)*3] = L_c + A
            Coordinate_p[i-1+2+(i-1)*3][2+(j-1)*3] = H - W_p /2 * (sin(beta[j-1])-sin(beta_n))
            # upper left corner
            Coordinate_p[i-1+3+(i-1)*3][0+(j-1)*3] = L_c + W_p /2 + (i-1)*(W_p+W_r) - abs(cos(beta[j-1])*W_p/2)
            Coordinate_p[i-1+3+(i-1)*3][1+(j-1)*3] = L_c + A
            Coordinate_p[i-1+3+(i-1)*3][2+(j-1)*3] = H + W_p /2 * (sin(beta[j-1])-sin(beta_n))

    Coordinate_p = np.array(Coordinate_p)

    return Coordinate_p