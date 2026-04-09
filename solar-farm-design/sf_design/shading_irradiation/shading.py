import numpy as np
from math import cos, sin, tan

def shading_a(Coordinate_p, gamma_s, theta_z, n_row, num_int):
    """
    Calculate the coordinates of the shading area.

    Parameters
    ----------
    Coordinate_p : numpy.ndarray
        The coordinates of the PV array.
    gamma_s : numpy.ndarray
        The azimuth angle of the sun.
    theta_z : numpy.ndarray
        The zenith angle of the sun.
    n_row : int
        The number of rows of the PV array.
    num_int : int
        Number of intervals.

    Returns
    -------
    Coordinate_s : numpy.ndarray
        The coordinates of the shading area.
    """

    Coordinate_s = np.zeros((365*24*num_int, n_row*8))
    for i in range(1, 366):
        for j in range(1, 24*num_int+1):
            for k in range(1, n_row*4+1):
                Coordinate_s[(i-1)*24*num_int+j-1, (k-1)*2] = Coordinate_p[k-1, 0] \
                    + Coordinate_p[k-1, 2]*tan(theta_z[j-1, i-1])*sin(gamma_s[j-1, i-1])
                Coordinate_s[(i-1)*24*num_int+j-1, (k-1)*2+1] = Coordinate_p[k-1, 1] \
                    + Coordinate_p[k-1, 2]*tan(theta_z[j-1, i-1])*cos(gamma_s[j-1, i-1])
    return Coordinate_s

def shading_tilt(Coordinate_p,gamma_s,theta_z,n_row,num_int):
    """
    Calculate the coordinates of the shading area with tilted panels.
    
    Parameters
    ----------
    Coordinate_p : numpy.ndarray
        The coordinates of the PV array.
    gamma_s : numpy.ndarray
        The azimuth angle of the sun.
    theta_z : numpy.ndarray
        The zenith angle of the sun.
    n_row : int
        The number of rows of the PV array.
    num_int : int  
        Number of intervals.


    Returns
    -------
    Coordinate_s : numpy.ndarray
        The coordinates of the shading area.
    """

    Coordinate_s = np.zeros((365*24*num_int, n_row*8))
    for i in range(1, 366):
        for j in range(1, 24*num_int+1):
            for k in range(1, n_row*4+1):
                Coordinate_s[(i-1)*24*num_int+j-1,(k-1)*2] = Coordinate_p[k-1, (j-1)*3]\
                    +Coordinate_p[k-1,2+(j-1)*3]*tan(theta_z[j-1,i-1])*sin(gamma_s[j-1,i-1])
                Coordinate_s[(i-1)*24*num_int+j-1,(k-1)*2+1] = Coordinate_p[k-1,1+(j-1)*3]\
                    +Coordinate_p[k-1,2+(j-1)*3]*tan(theta_z[j-1,i-1])*cos(gamma_s[j-1,i-1])
    return Coordinate_s