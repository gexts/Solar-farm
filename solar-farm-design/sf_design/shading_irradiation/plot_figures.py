import logging
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_shading_percentage(Coordinate_p, shading_percentage, 
                            n_row, dA, A_study, W_study, A_i, A_f, W_i, W_f, ax=None):
    """
    Parameters
    ----------
    Coordinate_p : numpy.ndarray
        The coordinates of the PV array.
    shading_percentage : numpy.ndarray
        The shading percentage.
    n_row : int
        The number of rows of the PV array.
    dA : float
        The distance between two shading points.
    A_study : float
        The length of the study area in the A direction.
    W_study : float
        The length of the study area in the W direction.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The axes.

    Returns
    -------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The axes.
    """

    x = np.arange(0,W_study+dA,dA)
    y = np.arange(0,(A_study+dA)//dA*dA,dA)
    X,Y = np.meshgrid(x,y)

    # Colorbar axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    im = ax.imshow(shading_percentage, cmap='Wistia', origin='lower',
                extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
                vmin=0, vmax=1)
    plt.colorbar(im, cax=cax, orientation='vertical')
    cax.tick_params(width=1.2, labelsize=28)
    cax.set_ylabel('Shading Percentage', fontsize=28)

    # Plot the PV array
    ax = plot_panels(Coordinate_p, n_row, ax=ax)
    ax = plot_boundaries(A_i, A_f, W_i, W_f, ax=ax)

    ax.set_xlabel('Study Length [m]', fontsize=28)
    ax.set_ylabel('Study Length [m]', fontsize=28)
    ax.tick_params(width=1.2, labelsize=28)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
    ax.set_xticks(np.arange(0, W_study+1, 5))
    ax.set_yticks(np.arange(0, A_study+1, 5))
    ax.set_aspect('equal')
    
    return ax

def plot_radiation_percentage(Coordinate_p, radiation_percentage, 
                              n_row, dA, A_study, W_study, A_i, A_f, W_i, W_f, ax=None):
    """
    Parameters
    ----------
    Coordinate_p : numpy.ndarray
        The coordinates of the PV array.
    radiation_percentage : numpy.ndarray
        The radiation percentage.
    n_row : int
        The number of rows of the PV array.
    dA : float
        The distance between two shading points.
    A_study : float
        The length of the study area in the A direction.
    W_study : float
        The length of the study area in the W direction.
    ax: matplotlib.axes._subplots.Axes3DSubplot
        The axes.

    Returns
    -------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The axes.        
    """
    x = np.arange(0,W_study+dA,dA)
    y = np.arange(0,(A_study+dA)//dA*dA,dA)
    X,Y = np.meshgrid(x,y)

    # Colorbar axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    im = ax.imshow(radiation_percentage, cmap='spring', origin='lower',
                extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
                vmin=0.6, vmax=1)
    plt.colorbar(im, cax=cax, orientation='vertical')
    cax.tick_params(width=1.2, labelsize=28)
    cax.set_ylabel('Radiation Percentage', fontsize=28)

    # Plot the PV array
    ax = plot_panels(Coordinate_p, n_row, ax=ax)
    ax = plot_boundaries(A_i, A_f, W_i, W_f, ax=ax)

    ax.set_xlabel('Study Length [m]', fontsize=28)
    ax.set_ylabel('Study Length [m]', fontsize=28)
    ax.tick_params(width=1.2, labelsize=28)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)

    ax.set_xticks(np.arange(0, W_study+1, 5))
    ax.set_yticks(np.arange(0, A_study+1, 5))
    ax.set_aspect('equal')

    return ax

def plot_radiation_par(Coordinate_p, radiation_par, 
                       n_row, dA, A_study, W_study, 
                       A_i, A_f, W_i, W_f, ax=None):
    """
    Parameters
    ----------
    Coordinate_p : numpy.ndarray
        The coordinates of the PV array.
    radiation_par : numpy.ndarray
        The radiation PAR.
    n_row : int
        The number of rows of the PV array.
    dA : float
        The distance between two shading points.
    A_study : float
        The length of the study area in the A direction.
    W_study : float
        The length of the study area in the W direction.
    ax: matplotlib.axes._subplots.Axes3DSubplot
        The axes.

    Returns
    -------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The axes.        
    """

    x = np.arange(0,W_study+dA,dA)
    y = np.arange(0,(A_study+dA)//dA*dA,dA)
    X,Y = np.meshgrid(x,y)

    # Colorbar axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    im = ax.imshow(radiation_par, cmap='spring', origin='lower',
                extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
                vmin=300, vmax=700)
    plt.colorbar(im, cax=cax, orientation='vertical')
    cax.tick_params(width=1.2, labelsize=28)
    cax.set_ylabel('PAR [μmol/(m$^2$·s)]', fontsize=28)

    # Plot the PV array
    ax = plot_panels(Coordinate_p, n_row, ax=ax)
    ax = plot_boundaries(A_i, A_f, W_i, W_f, ax=ax)

    ax.set_xlabel('Study Length [m]', fontsize=28)
    ax.set_ylabel('Study Length [m]', fontsize=28)
    ax.tick_params(width=1.2, labelsize=28)
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
    
    ax.set_xticks(np.arange(0, W_study+1, 5))
    ax.set_yticks(np.arange(0, A_study+1, 5))
    ax.set_aspect('equal')
    
    return ax

def plot_area_agri(Coordinate_p, area_agri, 
                   n_row, dA, A_study, W_study, 
                   A_i, A_f, W_i, W_f, ax=None):
    """
    Parameters
    ----------
    Coordinate_p : numpy.ndarray
        The coordinates of the PV array.
    area_agri : numpy.ndarray
        The area of agriculture.
    n_row : int
        The number of rows of the PV array.
    dA : float
        The distance between two shading points.
    A_study : float
        The length of the study area in the A direction.
    W_study : float
        The length of the study area in the W direction.
    A_i : float
        The initial value of A.
    A_f : float
        The final value of A.
    W_i : float
        The initial value of W.
    W_f : float
        The final value of W.
    ax: matplotlib.axes._subplots.Axes3DSubplot
        The axes.

    Returns
    -------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The axes.        
    """
    x = np.arange(0,W_study+dA,dA)
    y = np.arange(0,A_study+dA,dA)
    X,Y = np.meshgrid(x,y)

    # Colorbar axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    im = ax.contourf(area_agri, levels=[-0.5, 0.5, 1.5], 
                         colors=['b' , '#00FF00'], # colors=['lightgray' , 'pink'],
                         origin='lower',
                         extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
    
    cbar = plt.colorbar(im, boundaries=[0, 1, 2], values=[0, 1], cax=cax, orientation='vertical')
    cax.tick_params(width=1.2, labelsize=28)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Unsuitable', 'Suitable'])

    # Plot the PV array
    ax = plot_panels(Coordinate_p, n_row, ax=ax)
    ax = plot_boundaries(A_i, A_f, W_i, W_f, ax=ax)

    ax.set_xlabel('Study Length [m]', fontsize=28)
    ax.set_ylabel('Study Length [m]', fontsize=28)
    ax.tick_params(width=1.2, labelsize=28)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)

    ax.set_xticks(np.arange(0, W_study+1, 5))
    ax.set_yticks(np.arange(0, A_study+1, 5))
    ax.set_aspect('equal')
    # ax.set_xlim([0, W_study])
    # ax.set_yticks(np.arange(0, A_study, 5))

    # ax.set_title('Area of Agriculture')

    return ax

def plot_combined(Coordinate_p, shading_percentage, radiation_percentage,
                  radiation_par, area_agri, n_row, dA, A_study, W_study, axs=None):
    """
    Combined plot of shading percentage, radiation percentage, radiation PAR,
    and area of agriculture.

    Parameters
    ----------
    Coordinate_p : numpy.ndarray
        The coordinates of the PV array.
    shading_percentage : numpy.ndarray
        The shading percentage.
    radiation_percentage : numpy.ndarray
        The radiation percentage.
    radiation_par : numpy.ndarray
        The radiation PAR.
    area_agri : numpy.ndarray
        The area of agriculture.
    n_row : int
        The number of rows of the PV array.
    dA : float
        The distance between two shading points.
    A_study : float
        The length of the study area in the A direction.
    W_study : float
        The length of the study area in the W direction.
    axs : numpy.ndarray
        The axes.

    Returns
    -------
    axs : numpy.ndarray
        The axes.
    """

    x = np.arange(dA,W_study+dA,dA)
    y = np.arange(dA,(A_study+dA)//dA*dA,dA)
    X,Y = np.meshgrid(x,y)

    for i, ax in enumerate(axs.flatten()):
        im = None
        if i == 0:
            # Plot the shading percentage
            im = ax.imshow(shading_percentage, cmap='Wistia', origin='lower',
                        extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
            ax.set_title('Shading Percentage')
        elif i == 1:
            # Plot the radiation percentage
            im = ax.imshow(radiation_percentage, cmap='spring', origin='lower',
                extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
            ax.set_title('Radiation Percentage')
        elif i == 2:
            # Plot the radiation PAR
            im = ax.imshow(radiation_par, cmap='spring', origin='lower',
                extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
            ax.set_title('Radiation PAR')
        elif i == 3:
            # Plot the area of agriculture
            cmap = cm.get_cmap('Pastel1', 2)
            im = ax.imshow(area_agri, cmap=cmap, origin='lower',
                        extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
            ax.set_title('Area of Agriculture')
        else:
            raise ValueError('The number of subplots is not correct.')
        
        if im is not None:
            # Plot the PV array
            ax = plot_panels(Coordinate_p, n_row, ax=ax)

            # Add the colorbar
            fig = ax.figure
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            if i != 3:
                fig.colorbar(im, cax=cax, orientation='vertical', fraction=0.1)
            else:
                cbar = fig.colorbar(im, cax=cax, orientation='vertical', fraction=0.1,
                                    boundaries=[0, 1, 2], values=[0, 1])
                cbar.set_ticks([0.5, 1.5])
                cbar.set_ticklabels(['Unsuitable', 'Suitable'])

    return axs
                           

def plot_panels(Coordinate_p, n_row, ax=None):
    """
    Parameters
    ----------
    Coordinate_p : numpy.ndarray
        The coordinates of the PV array.
    n_row : int
        The number of rows of the PV array.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The axes.

    Returns
    -------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The axes.        
    """

    for i in range(1, n_row+1):
        x = [Coordinate_p[(i-1)*4,0],Coordinate_p[(i-1)*4+1,0],
            Coordinate_p[(i-1)*4+2,0],Coordinate_p[(i-1)*4+3,0],Coordinate_p[(i-1)*4,0]]
        y = [Coordinate_p[(i-1)*4,1],Coordinate_p[(i-1)*4+1,1],
            Coordinate_p[(i-1)*4+2,1],Coordinate_p[(i-1)*4+3,1],Coordinate_p[(i-1)*4,1]]
        ax.plot(x,y,'-o',color='k',linewidth=1)


    ## for one day, coordinate_p changes per hour
    # for j in range(1, 25):
    #     for i in range(1, n_row+1):
    #         x = [Coordinate_p[(i-1)*4,0+(j-1)*3],Coordinate_p[(i-1)*4+1,1+(j-1)*3],
    #         Coordinate_p[(i-1)*4+2,2+(j-1)*3],Coordinate_p[(i-1)*4+3,3+(j-1)*3],Coordinate_p[(i-1)*4,0+(j-1)*3]]
    #         y = [Coordinate_p[(i-1)*4,1],Coordinate_p[(i-1)*4+1,1],
    #         Coordinate_p[(i-1)*4+2,1],Coordinate_p[(i-1)*4+3,1],Coordinate_p[(i-1)*4,1]]
    #         ax.plot(x,y,'-o',color='b',linewidth=1)

    return ax


def plot_boundaries(A_i, A_f, W_i, W_f, ax=None):

    ax.plot([W_i, W_f], [A_i, A_i], 'r--')
    ax.plot([W_i, W_f], [A_f, A_f], 'r--')
    ax.plot([W_i, W_i], [A_i, A_f], 'r--')
    ax.plot([W_f, W_f], [A_i, A_f], 'r--')
    # ax.set_xlabel('Study Length [m]')
    # ax.set_ylabel('Study Length [m]')
    # ax.set_title('Study Area')

    return ax