from math import cos, sin
from math import pi
import numpy as np

def solar_angle(phi,L_loc,L_st, num_int):
    """
    Calculate the solar angle.

    Parameters
    ----------
    phi : float
        The latitude of the site.
    L_loc : float
        The longitude of the site.
    L_st : float
        The longitude of the standard meridian.

    Returns
    -------
    theta_z: numpy.ndarray
        The zenith angle of the sun.
    gamma_s: numpy.ndarray
        The azimuth angle of the sun.
    """
    
    # get solar angle
    # dt = 1 # [hr] Time step
    dt = 1/num_int # [hr] Time step, 5Min
    ntime = int(24/dt) # number of time step
    t_s = np.arange(0,24+dt,dt) # Time step vector
    n = np.arange(1,366) # nth day of the year
    B = (n-1)*360/365*pi/180 # [rad] 
    E = 229.2*(0.000075+0.001868*np.cos(B)-0.032077*np.sin(B)-0.014615*np.cos(2*B)-0.04089*np.sin(2*B))
    T_st = np.zeros((ntime,365))
    T_solar = np.zeros((ntime,365))
    for i in range(ntime):
        for j in range(365):
            T_st[i,j] = i/num_int # old: /1, /12
            T_solar[i,j] = T_st[i,j]-(4*(L_st-L_loc)-E[j])/60  

    omega = (15*T_solar-180)*pi/180 # [rad] Hour angle
    delta = (0.006918-0.399912*np.cos(B)+0.070257*np.sin(B)-0.006758*np.cos(2*B)+0.000907*np.sin(2*B)-0.002697*np.cos(3*B)+0.00148*np.sin(3*B))
    # [rad] Declination

    # use for loop to calculate theta_z
    theta_z = np.zeros((ntime,365))
    for i in range(ntime):
        for j in range(365):
            theta_z[i,j] = np.arccos(cos(phi)*cos(delta[j])*cos(omega[i,j])+sin(phi)*sin(delta[j]))
    # [rad] Zenith angle
    
    # use for loop to calculate gamma_s
    gamma_s = np.zeros((ntime,365))
    for i in range(ntime):
        for j in range(365):
            gamma_s[i,j] = np.sign(omega[i,j])*np.abs(np.arccos((cos(theta_z[i,j])*sin(phi)-sin(delta[j]))/(sin(theta_z[i,j])*cos(phi))))
    
    # [rad] Solar azimuth angle
    alpha_s = (pi/2-theta_z) # Solar altitude angle

    # Set theta_z and gamma_s to nan when alpha_s<0
    # theta_z = np.where(alpha_s<0, np.nan, theta_z)
    # gamma_s = np.where(alpha_s<0, np.nan, gamma_s)

    # for j in range(365):
    #     for i in range(2,ntime-2):
    #         if alpha_s[i,j]<0:
    #             theta_z[i-1,j] = np.nan 
    #             theta_z[i+1,j] = np.nan
    #             theta_z[i-2,j] = np.nan
    #             theta_z[i+2,j] = np.nan
    #             gamma_s[i-1,j] = np.nan
    #             gamma_s[i+1,j] = np.nan
    #             gamma_s[i-2,j] = np.nan
    #             gamma_s[i+2,j] = np.nan
    
    for j in range(365):
        for i in range(2*num_int,ntime-2*num_int):
            if alpha_s[i,j]<0:
                theta_z[i-num_int:i+num_int,j] = np.nan 
                theta_z[i-2*num_int:i+2*num_int,j] = np.nan
                gamma_s[i-num_int:i+num_int,j] = np.nan
                gamma_s[i-2*num_int:i+2*num_int,j] = np.nan

    return theta_z, gamma_s

if __name__ == "__main__":
    phi = 42.45*pi/180 #[rad] Latitude
    L_loc = -76.50 #[deg] Longitude
    L_st = -75 #standard longitude
    theta_z, gamma_s = solar_angle(phi,L_loc,L_st)
    print(theta_z)