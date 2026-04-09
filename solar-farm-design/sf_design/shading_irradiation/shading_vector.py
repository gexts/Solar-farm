import numpy as np
from matplotlib.path import Path


def shading_vector(time,Coordinate_s,n_sA,n_sW,dA,n_row):
    """
    
    Parameters
    ----------
    time : int
        The time step.
    Coordinate_s : numpy.ndarray
        The coordinates of the shading points.
    n_sA : int
        The number of shading points in the A direction.
    n_sW : int
        The number of shading points in the W direction.
    dA : float
        The distance between two shading points.
    n_row : int
        The number of rows of the PV array.

    Returns
    -------
    shading : numpy.ndarray
        The shading vector.
    """

    ground_x = np.zeros((n_sA*n_sW))
    ground_y = np.zeros((n_sA*n_sW))

    for i in range(1, n_sA+1):
        for j in range(1, n_sW+1):
            ground_x[(i-1)*n_sW+j-1] = j/(1/dA)
            ground_y[(i-1)*n_sW+j-1] = i/(1/dA)

    xv = np.zeros((n_row,5))
    yv = np.zeros((n_row,5))

    for i in range(1,n_row+1):
        xv[i-1,0] = Coordinate_s[time,(i-1)*8]
        xv[i-1,1] = Coordinate_s[time,(i-1)*8+2]
        xv[i-1,2] = Coordinate_s[time,(i-1)*8+4]
        xv[i-1,3] = Coordinate_s[time,(i-1)*8+6]
        xv[i-1,4] = Coordinate_s[time,(i-1)*8]
        yv[i-1,0] = Coordinate_s[time,(i-1)*8+1]
        yv[i-1,1] = Coordinate_s[time,(i-1)*8+3]
        yv[i-1,2] = Coordinate_s[time,(i-1)*8+5]
        yv[i-1,3] = Coordinate_s[time,(i-1)*8+7]
        yv[i-1,4] = Coordinate_s[time,(i-1)*8+1]

    in_ = np.zeros((n_row,len(ground_x)))
    # on_ = np.zeros((n_row,len(ground_x)))
    for i in range(1,n_row+1):

        path = Path(np.vstack((xv[i-1, :], yv[i-1, :])).T)
        in_[i-1, :] = path.contains_points(np.vstack((ground_x, ground_y)).T)
        # # On the edge of the polygon
        # on_[i-1, :] = np.logical_and(path.contains_points(np.vstack((ground_x, ground_y)).T),
        #                            np.logical_not(in_[i-1, :]))
        # # Combine the two
        # in_[i-1, :] = np.logical_or(in_[i-1, :], on_[i-1, :])

    shading = np.zeros((n_sW,n_sA))
    for i in range(1,n_sA+1):
        for j in range(1,n_sW+1):
            if in_[:,(i-1)*n_sW+j-1].any() == 1:
                shading[j-1,i-1] = 1
            else:
                shading[j-1,i-1] = 0

    return shading

