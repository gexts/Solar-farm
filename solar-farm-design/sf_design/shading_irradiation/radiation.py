import numpy as np


def radiation(phi,L_st,L_loc,weather,num_int):
    """
    Calculate direct and diffused radiation.

    Parameters
    ----------
    phi : float
        Latitude of the location [deg].
    L_st : float
        Longitude of the standard meridian [deg].
    L_loc : float
        Longitude of the location [deg].
    weather : numpy.ndarray
        Weather data.
    num_int : int
        Number of integration.

    Returns
    -------
    I_t : numpy.ndarray
        Total radiation [W/m2].
    I_d : numpy.ndarray
        Diffuse radiation [W/m2].
    """
    
    # dt = 1 # [hr] Time step
    dt = 1/num_int # [hr] Time step, 5Min
    ntime = int(24/dt) # number of time step in one day
    t_s = np.arange(0,24+dt,dt) # Time step vector
    n = np.arange(1,366) # nth day of the year vector
    G_sc = 2450 #1367 [W/m2]
    #Range factors have to be looked up according to sprectural range.
    B = (n-1)*360/365*np.pi/180 # [rad] 
    G_on = G_sc*(1.000110+0.034221*np.cos(B)+0.01280*np.sin(B)+0.000719*np.cos(2*B)+0.000077*np.sin(2*B))
    # [micromol/m2-s] the extraterrestrial radiation incident on the plane normal to the radiation on the nth day of the year
    E = 229.2*(0.000075+0.001868*np.cos(B)-0.032077*np.sin(B)-0.014615*np.cos(2*B)-0.04089*np.sin(2*B))
    T_st = np.zeros((ntime,365))
    T_solar = np.zeros((ntime,365))
    for i in range(ntime):
        for j in range(365):
            T_st[i,j] = i/num_int
            T_solar[i,j] = T_st[i,j]-(4*(L_st-L_loc)-E[j])/60
    omega_deg = (15*T_solar-180) # [deg] Hour angle
    omega = (15*T_solar-180)*np.pi/180 # [rad] Hour angle
    delta = (0.006918-0.399912*np.cos(B)+0.070257*np.sin(B)-0.006758*np.cos(2*B)+0.000907*np.sin(2*B)-0.002697*np.cos(3*B)+0.00148*np.sin(3*B))
    # [rad] Declination
    theta_z = np.zeros((ntime,365))
    for i in range(ntime):
        for j in range(365):
            theta_z[i,j] = np.arccos(np.cos(phi)*np.cos(delta[j])*np.cos(omega[i,j])+np.sin(phi)*np.sin(delta[j]))
    # [rad] Zenith angle
    gamma_s = np.zeros((ntime,365))
    for i in range(ntime):
        for j in range(365):
            gamma_s[i,j] = np.sign(omega[i,j])*np.abs(np.arccos((np.cos(theta_z[i,j])*np.sin(phi)-np.sin(delta[j]))/(np.sin(theta_z[i,j])*np.cos(phi))))
    # [rad] Solar azimuth angle
    gamma_s_deg = gamma_s/np.pi*180
    alpha_s = (np.pi/2-theta_z) # Solar altitude angle
    alpha_s_deg = alpha_s/np.pi*180
    omega_s = np.arccos(-np.tan(phi)*np.tan(delta))*180/np.pi # [degree] sunset hour angle
    G_o = np.zeros((ntime,365))
    for j in range(365):
        for i in range(ntime):
            G_o[i,j] = G_on[j]*(np.cos(phi)*np.cos(delta[j])*np.cos(omega[i,j])+np.sin(phi)*np.sin(delta[j]))
    # [micromol/m2-s]
    I_o = np.zeros((ntime,365))
    for j in range(365):
        for i in range(ntime):
            if G_o[i,j] > 0:
                I_o[i,j] = I_o[i,j]+G_o[i,j]*dt*3600
    # [micromol/m2-hr] hourly total extraterrestrial
    I = np.zeros((ntime,365))
    for j in range(365):
        for i in range(ntime):
            I[i,j] = weather[j*ntime+i]*(0.469-0.08)/0.217*3600*dt 
    # assuming H is about half of H_o, use experimental data ig applicable
    K_T = I/I_o # daily clearness index
    rd = np.zeros((ntime,365))
    for j in range(365):
        for i in range(ntime):
            if K_T[i,j] > 0.8:
                rd[i,j] = rd[i,j]+0.165
            else:
                if K_T[i,j] > 0.22:
                    rd[i,j] = rd[i,j]+0.9511-0.1604*K_T[i,j]+4.388*K_T[i,j]**2-16.638*K_T[i,j]**3+12.336*K_T[i,j]**4
                else:
                    rd[i,j] = rd[i,j]+1-0.09*K_T[i,j]
    # Diffuse ratio
    I_d = np.zeros((ntime,365))
    I_d = I*rd/(dt*3600) # [micromol/m2-s] hourly diffuse radiation
    I_t = np.zeros((ntime,365))
    I_t = I/(dt*3600)

    # Caculate Reflection ratio: 
    # 1st, caculate DNI using GHI(I_t) = DHI(I_d) + DNI * cos(alpha_s) 
    # 2nd, find the ratio of reflected radiation intensity to the original radiation, which is a [ntime, 365] matrix, as a function of alpha_s.
    # Using Fresnel equation'Schlick approximation, the model is:
    # refractive index_water = 1.333333, refractive index_air = 1.000277
    # new definition: DNI, I_reflect(target function), f_reflectratio
    # I_reflect = DNI * f_reflectratio
    DNI=(I_t-I_d)/np.cos(alpha_s)
    eta = 1.000277 / 1.333333
    f_0 = ((eta-1)/(eta+1))**2
    f_reflectratio = f_0 + (1-f_0)*(1-np.sin(alpha_s))**5
    I_reflect = np.zeros((ntime, 365))
    I_reflect = DNI * f_reflectratio

    # Calulate the refraction ratio
    # Basic model: Fresnel equation
    # The energy of incoming light all goes into the reflection and refraction
    I_refract = I_reflect * (1-f_reflectratio)/f_reflectratio

    return I_t, I_d, I_reflect, I_refract
            
def radiation_p(day_i, day_f, I_d, I_t, n_sA, n_sW, shading_list, num_int):
    """
    Calculate radiation percentage and PAR.
    Not used in the current version of the model.

    Parameters
    ----------
    day_i : int
        Initial day.
    day_f : int
        Final day.
    I_d : array
        Diffuse radiation.
    I_t : array
        Total radiation.
    n_sA : int
        Number of solar azimuth angles.
    n_sW : int
        Number of solar zenith angles.
    shading_list : list
        List of shading vectors.
    num_int : int
        Number of intervals.

    Returns
    -------
    radiation_percentage : numpy.ndarray
        Radiation percentage.
    radiation_par : numpy.ndarray
        PAR.
    """
    
    radiation_t = np.zeros((n_sW,n_sA))
    radiation_t_par = np.zeros((n_sW,n_sA))
    count = 0
    for i in range(day_i,day_f+1):
        for j in range(1,24*num_int+1):
            shading_v = shading_list[i-day_i][j-1]

            if not np.isnan(I_d[j-1,i-1]):
                count += 1
                for k in range(n_sW):
                    for l in range(n_sA):
                        if shading_v[k,l] == 1:
                            radiation_t[k,l] += I_d[j-1,i-1]/I_t[j-1,i-1]
                            radiation_t_par[k,l] += I_d[j-1,i-1]
                        else:
                            radiation_t[k,l] += 1
                            radiation_t_par[k,l] += I_t[j-1,i-1]
    radiation_percentage = radiation_t.T/(count)
    radiation_par = radiation_t_par.T/(count)

    return radiation_percentage, radiation_par