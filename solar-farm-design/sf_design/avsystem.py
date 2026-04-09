import os
import sys
import logging
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
# import sf_design.power_generation.solar_gen as solar_gen
import sf_design.shading_irradiation.site_construction as sc
import sf_design.shading_irradiation.solar_angle as sa
import sf_design.shading_irradiation.shading as sh
import sf_design.shading_irradiation.shading_vector as sv
import sf_design.shading_irradiation.radiation as rd
import sf_design.shading_irradiation.plot_figures as pf
from math import radians
import pickle

class AVSystem(PVSystem):
    """
    The AVSystem class defines a standard set of agrivoltaic system attributes and modeling functions.
    This is an inherited class from the PVSystem class in the pvlib package.
    
    """

    def define_parameters(self, **kwargs):
    
        # %% Define agrivoltaic system parameters

        # Distance of aisle between solar panels in m
        self.W_r = kwargs.get('W_r', 4)
        
        # Length of the solar panel in m
        self.A = kwargs.get('A', 20)

        # PV panel height in m, distance from the axis to the ground
        self.H = kwargs.get('H', 1)

        # Number of rows of the PV array
        self.n_row = kwargs.get('n_row', 4)

        # PV panel width in m
        self.W_p = kwargs.get('W_p', 1)

        # Clearance on each side
        self.L_c = kwargs.get('L_c', 5)

        # Start day of simulation
        self.day_i = kwargs.get('day_i', 60)

        # End day of simulation
        self.day_f = kwargs.get('day_f', 274)

        logging.info('Defined agrivoltaic system.')

    def set_timeinterval(self, num_int):
        self.num_int = num_int

    def set_meshgrid(self):
        
       pass

    def set_surface_angle(self):
        
        pass

    def construct_site(self):

        pass

    def calc_shading_percentage(self, n_proc=12):
        """
        Calculate shading percentage
        """
        start = time.time()
        shading_list = list()

        if n_proc <= 1:
            for i in range(self.day_i, self.day_f+1):
                shading_sublist = list()
                for j in range(1, 24*self.num_int+1):
                    shading = sv.shading_vector(
                        (i-1)*24*self.num_int+j,
                        self.coordinate_s,
                        self.n_sA,
                        self.n_sW,
                        self.dA,
                        self.n_row)
                    shading_sublist.append(shading)
                shading_list.append(shading_sublist)

                elapsed = time.time() - start
                logging.info(f'Calculating day: {i}/{self.day_f}. Elapsed time: {elapsed:.4f} s.')
        else:
            # Start multiprocessing pool
            pool = mp.Pool(n_proc)

            # List of multiprocessing results
            shading_res_list = list()

            for i in range(self.day_i, self.day_f+1):
                shading_res_sublist = list()
                for j in range(1, 24*self.num_int+1):
                    res = pool.apply_async(sv.shading_vector, args=(
                        (i-1)*24*self.num_int+j, self.coordinate_s,
                        self.n_sA, self.n_sW, self.dA, self.n_row))
                    shading_res_sublist.append(res)

                shading_res_list.append(shading_res_sublist)

            for i in range(self.day_i, self.day_f+1):
                shading_sublist = list()
                for j in range(1, 24*self.num_int+1):
                    shading = shading_res_list[i-self.day_i][j-1].get()
                    shading_sublist.append(shading)
                shading_list.append(shading_sublist)

                elapsed = time.time() - start
                logging.info(f'Calculating day: {i}/{self.day_f}. Elapsed time: {elapsed:.4f} s.')

            pool.close()
            pool.join()

        # Convert shading_list to shading_array
        self.shading_array = np.array(shading_list)

        # Total shading of all days and hours
        shading_t = self.shading_array.sum(axis=(0, 1))

        # Total number of hours with sun
        count_b = np.sum(~np.isnan(self.gamma_s[:, self.day_i: self.day_f+1]))

        # Average shading percentage
        self.shading_percentage = shading_t.T/count_b

        self.average_shading = np.mean(self.shading_percentage[self.n_Ai:self.n_Af, self.n_Wi:self.n_Wf])

        logging.info('Calculated shading percentage.')

    def calc_irradiance_components(self, weather):
        """
        Calculate diffused and total irradiance
        """

        weather_array = self._prepare_weather_array(weather)
        radiation_out = rd.radiation(radians(self.site_location.latitude), self.L_st,
                                     self.site_location.longitude, weather_array, self.num_int)
        self.I_t, I_d_temp = radiation_out[:2]
        self.I_d = I_d_temp * 1

        # Calculate direct irradiance
        self.I_n = (self.I_t - self.I_d)/self.cos_theta

        logging.info('Calculated irradiance components.')

    @staticmethod
    def _prepare_weather_array(weather):
        """
        Normalize weather inputs for the shading/irradiance routines.

        The legacy radiation model expects a 1D sequence of GHI values.
        Examples in this repository pass either a pandas object or a numpy
        array, so we accept both and reduce them to a numeric 1D array.
        """

        if isinstance(weather, pd.DataFrame):
            if 'GHI' in weather.columns:
                return weather['GHI'].to_numpy(dtype=float)
            if weather.shape[1] == 1:
                return weather.iloc[:, 0].to_numpy(dtype=float)
            raise ValueError('weather DataFrame must contain a GHI column or a single data column.')

        if isinstance(weather, pd.Series):
            return weather.to_numpy(dtype=float)

        weather_array = np.asarray(weather, dtype=float)
        if weather_array.ndim == 1:
            return weather_array
        if weather_array.ndim == 2 and 1 in weather_array.shape:
            return weather_array.reshape(-1)

        raise ValueError('weather must be a 1D sequence of GHI values or a single-column pandas object.')

    def calc_irradiance_percentage(self, PAR_require, 
                                   save_results=False, results_dir=None):
        """
        Calculate irradiance percentage and PAR

        Parameters
        ----------
        PAR_require : float
            PAR requirement in W/m^2
        results_dir : str
            Directory to save results

        Returns
        -------
        None
        """

        radiation_pct_array = np.zeros((self.day_f-self.day_i+1, 24*self.num_int, self.n_sW, self.n_sA))
        radiation_par_array = np.zeros((self.day_f-self.day_i+1, 24*self.num_int, self.n_sW, self.n_sA))

        for i in range(self.day_i, self.day_f+1):
            for j in range(1, 24*self.num_int+1):

                # Shading vector
                shading = self.shading_array[i-self.day_i, j-1, :, :]

                if not np.isnan(self.I_d[j-1, i-1]):

                    # Radiation percentage for one hour
                    # Radiation percentage: If shaded, I_d/I_t, else 1
                    radiation_pct = np.where(
                        shading == 1, self.I_d[j-1, i-1]/self.I_t[j-1, i-1], 1)
                    radiation_pct_array[i-self.day_i, j-1, :, :] = radiation_pct

                    # Radiation PAR for one hour
                    # Radiation: If shaded, I = I_d (diffused), else I = I_t (total)
                    radiation_par = np.where(
                        shading == 1, self.I_d[j-1, i-1], self.I_t[j-1, i-1])
                    radiation_par_array[i-self.day_i, j-1, :, :] = radiation_par
                    
        #print shading_percentage
        if save_results:
            with open(os.path.join(results_dir, "radiation_t_par_array.pkl"), "wb") as f:
                pickle.dump(radiation_par_array, f)

        # Total radiaiton percentage and PAR for all days and hours for each pixel
        radiation_pct_sum_hours = radiation_pct_array.sum(axis=(0, 1)) # Total radiation percentage (I/I_t) in all hours
        radiation_par_sum_hours = radiation_par_array.sum(axis=(0, 1)) # Total PAR in all hours

        # Average radiation percentage and PAR for each pixel
        n_hour_with_sun = np.sum(~np.isnan(self.I_d[:, self.day_i:self.day_f+1])) # Number of hours with sun
        self.radiation_percentage = radiation_pct_sum_hours.T/n_hour_with_sun # Averaged total radiation of all hours
        self.radiation_par = radiation_par_sum_hours.T/n_hour_with_sun # Averaged total PAR of all hours

        # Calculate total radiation percentage and PAR for the calculation area
        total_radiation = self.radiation_percentage[self.n_Ai:self.n_Af, self.n_Wi:self.n_Wf].sum() # Sum of averaged total radiation in calculation area
        total_par = self.radiation_par[self.n_Ai:self.n_Af, self.n_Wi:self.n_Wf].sum() # Sum of averaged total PAR in calculation area
        n_pixel_calc = (self.n_Af-self.n_Ai)*(self.n_Wf-self.n_Wi) # Number of pixels in calculation area
        n_pixel_calc_meet_req = np.sum(self.radiation_par[self.n_Ai:self.n_Af, self.n_Wi:self.n_Wf] > PAR_require) # Number of pixels that meet PAR requirement in calculation area

        # Calculate area that meets PAR requirement
        self.area_agri = np.where(self.radiation_par >= PAR_require, 1, 0) # Area that averaged total PAR meets PAR requirement

        # Calculate average radiation percentage and PAR
        self.average_radiation = total_radiation/n_pixel_calc # Averaged total radiation of calculation area (average of all pixels and all hours)
        self.average_par = total_par/n_pixel_calc # Averaged total PAR of calculation area (average of all pixels and all hours)
        self.area_percentage = n_pixel_calc_meet_req/n_pixel_calc # Percentage of area that meets PAR requirement

        logging.info('Calculated irradiance and PAR.')
        
    def plot_shading_percentage(self, fig_width=8, fig_height=6):
        """
        Plot shading percentage

        Parameters
        ----------
        None

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax = pf.plot_shading_percentage(self.coordinate_p, self.shading_percentage,
                                    self.n_row, self.dA, self.A_study, self.W_study, 
                                    self.A_i, self.A_f, self.W_i, self.W_f, ax=ax)
        
        return fig
    
    def plot_irradiance_percentage(self, fig_width=8, fig_height=6):
        """
        Plot irradiance percentage

        Parameters
        ----------
        None

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax = pf.plot_radiation_percentage(self.coordinate_p, self.radiation_percentage,
                                        self.n_row, self.dA, self.A_study, self.W_study,
                                         self.A_i, self.A_f, self.W_i, self.W_f, ax=ax)
        
        return fig
    
    def plot_irradiance_par(self, fig_width=8, fig_height=6):
        """
        Plot irradiance PAR

        Parameters
        ----------
        None

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax = pf.plot_radiation_par(self.coordinate_p, self.radiation_par,
                                self.n_row, self.dA, self.A_study, self.W_study,
                                 self.A_i, self.A_f, self.W_i, self.W_f, ax=ax)
        
        return fig
    
    def plot_area_agri(self, fig_width=8, fig_height=6):
        """
        Plot area suitable for agriculture

        Parameters
        ----------
        None

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax = pf.plot_area_agri(self.coordinate_p, self.area_agri,
                                self.n_row, self.dA, self.A_study, self.W_study, 
                                self.A_i, self.A_f, self.W_i, self.W_f, ax=ax)
        
        return fig
    
    def plot_combined(self, fig_width=8, fig_height=6):
        """
        Plot combined figure

        Parameters
        ----------
        None

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """

        fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        ax = pf.plot_combined(self.coordinate_p, self.shading_percentage, 
                              self.radiation_percentage, self.radiation_par, 
                              self.area_agri, self.n_row, self.dA, self.A_study, 
                              self.W_study, axs=axs)
        
        return fig
    
    def report_results(self):

        logging.info('Average shading percentage: {:.2f}%'.format(self.average_shading*100))
        logging.info('Average radiation percentage: {:.2f}%'.format(self.average_radiation*100))
        logging.info('Average PAR: {:.2f} W/m^2'.format(self.average_par))
        logging.info('Area suitable for agriculture: {:.2f}%'.format(self.area_percentage*100))

    def save_results(self, results_dir):

        with open(os.path.join(results_dir, "shading_percentage.pkl"), "wb") as f:
            pickle.dump(self.shading_percentage, f)
        with open(os.path.join(results_dir, "radiation_percentage.pkl"), "wb") as f:
            pickle.dump(self.radiation_percentage, f)
        with open(os.path.join(results_dir, "radiation_par.pkl"), "wb") as f:
            pickle.dump(self.radiation_par, f)
        with open(os.path.join(results_dir, "area_agri.pkl"), "wb") as f:
            pickle.dump(self.area_agri, f)

        logging.info('Results saved to {}'.format(results_dir))

    def load_results(self, results_dir):

        with open(os.path.join(results_dir, "shading_percentage.pkl"), "rb") as f:
            self.shading_percentage = pickle.load(f)
        with open(os.path.join(results_dir, "radiation_percentage.pkl"), "rb") as f:
            self.radiation_percentage = pickle.load(f)
        with open(os.path.join(results_dir, "radiation_par.pkl"), "rb") as f:
            self.radiation_par = pickle.load(f)
        with open(os.path.join(results_dir, "area_agri.pkl"), "rb") as f:
            self.area_agri = pickle.load(f)

        logging.info('Results loaded from {}'.format(results_dir))


class AVSystemNSFT(AVSystem):

    def set_meshgrid(self, dA=0.1):
        
        # %% Set starting and ending points of meshgrid

        # [m] starting point of length to calculate average
        self.A_i = self.L_c +self.A + 1/2*self.W_r

        # [m] ending point of length to calculate average
        self.A_f = (self.n_row-1)*(self.A+self.W_r) + self.L_c - 1/2*self.W_r

        # [m] starting point of width to calculate average
        self.W_i = self.L_c + 1/4*self.W_p
        
        # [m] ending point of width to calculate average
        self.W_f = self.L_c + 3/4*self.W_p

        # [m] length of study area
        self.A_study = self.A * self.n_row + (self.n_row-1)* self.W_r + 2*self.L_c
        
        # [m] length of study area
        self.W_study = self.W_p + 2*self.L_c
        
        # %% Create meshgrid for shading and irradiation calculation
        
        # Mesh size
        self.dA = dA

        # Number of meshgrid points in length
        self.n_sA = int(self.A_study/self.dA)

        # Number of meshgrid points in width
        self.n_sW = int(self.W_study/self.dA)

        # Starting point of length to calculate average
        self.n_Ai = int(self.A_i/self.dA+1)

        # Ending point of length to calculate average
        self.n_Af = int(self.A_f/self.dA+1)

        # Starting point of width to calculate average
        self.n_Wi = int(self.W_i/self.dA+1)

        # Ending point of width to calculate average
        self.n_Wf = int(self.W_f/self.dA+1)

    def set_surface_angle(self, tilt_angle):
        
        # Tilt angle of solar panel
        self.beta = radians(tilt_angle)

        # Surface azimuth angle of solar panel
        self.gamma = 0

    def construct_site(self, site_location, L_st):

        # Site location with latitude and longitude
        self.site_location = site_location

        # Standard longitude
        self.L_st = L_st
        
        # Calculate panel coordinates
        self.coordinate_p = sc.site_construction_ns_ft(
            self.W_r, self.A, self.H, 
            self.n_row, self.W_p, self.beta, self.L_c)
        
        # Calculate solar zenith angle and solar azimuth angle
        self.theta_z, self.gamma_s = sa.solar_angle(radians(self.site_location.latitude), 
                                                    self.site_location.longitude, self.L_st, self.num_int)
        
        # Calculate cosine of the angle of incidence
        self.cos_theta = np.cos(self.theta_z)*np.cos(self.beta) + \
            np.sin(self.theta_z)*np.sin(self.beta)*np.cos(self.gamma_s-self.gamma)

        # Calculate shading coordinates
        self.coordinate_s = sh.shading_a(self.coordinate_p, self.gamma_s, self.theta_z, self.n_row, self.num_int)

class AVSystemNSSAT(AVSystem):

    def set_meshgrid(self, dA=0.1):
        
        # %% Set starting and ending points of meshgrid

        # [m] starting point of length to calculate average
        self.A_i = self.L_c +self.A + 1/2*self.W_r

        # [m] ending point of length to calculate average
        self.A_f = (self.n_row-1)*(self.A+self.W_r) + self.L_c - 1/2*self.W_r

        # [m] starting point of width to calculate average
        self.W_i = self.L_c + 1/4*self.W_p
        
        # [m] ending point of width to calculate average
        self.W_f = self.L_c + 3/4*self.W_p

        # [m] length of study area
        self.A_study = self.A * self.n_row + (self.n_row-1)* self.W_r + 2*self.L_c
        
        # [m] length of study area
        self.W_study = self.W_p + 2*self.L_c
        
        # %% Create meshgrid for shading and irradiation calculation
        
        # Mesh size
        self.dA = dA

        # Number of meshgrid points in length
        self.n_sA = int(self.A_study/self.dA)

        # Number of meshgrid points in width
        self.n_sW = int(self.W_study/self.dA)

        # Starting point of length to calculate average
        self.n_Ai = int(self.A_i/self.dA+1)

        # Ending point of length to calculate average
        self.n_Af = int(self.A_f/self.dA+1)

        # Starting point of width to calculate average
        self.n_Wi = int(self.W_i/self.dA+1)

        # Ending point of width to calculate average
        self.n_Wf = int(self.W_f/self.dA+1)

    def set_surface_angle(self, tilt_morning, tilt_noon):

        self.beta_m = radians(tilt_morning)
        self.beta_n = radians(tilt_noon)
        
        # Calculate hourly tilt angle of solar panel
        self.beta = np.zeros((24*self.num_int,1))
        self.gamma =  0
    
        for j in range(24*self.num_int): # 288 = 24*12

            if 6*self.num_int < j <= 12*self.num_int: # old: j < 12, 144 = 12*12
                # Morning
                self.beta[j] = self.beta_m - (self.beta_m - self.beta_n)/(6*self.num_int) * (j-6*self.num_int) # old: 6, 6. 72 = 6*12, 72 = 6*12
            elif 12*self.num_int <j <= 13*self.num_int: # old: j ==12, 144 = 12*12, 156 = 13*12 (12:00-13:00, 1 hour, PV panels stay as tilt_noon)
                # Noon
                self.beta[j] = self.beta_n
            elif 13*self.num_int < j <= 19*self.num_int:
                # Afternoon
                self.beta[j] = self.beta_n + (self.beta_m - self.beta_n)/(6*self.num_int) * (j-13*self.num_int) # old: 6, 12. 72 = 6*12, 156 = 13*12
            else:
                # Before sunrise and after sunset
                self.beta[j] = self.beta_n

    def construct_site(self, site_location, L_st):

        # Site location with latitude and longitude
        self.site_location = site_location

        # Standard longitude
        self.L_st = L_st
        
        # Calculate panel coordinates
        self.coordinate_p = sc.site_construction_ns_sat(
            self.W_r, self.A, self.H, 
            self.n_row, self.W_p, self.beta, self.L_c, self.beta_n, self.num_int)
        
        # Calculate solar zenith angle and solar azimuth angle
        self.theta_z, self.gamma_s = sa.solar_angle(radians(self.site_location.latitude), 
                                                    self.site_location.longitude, self.L_st, self.num_int)
        
        # Calculate cosine of the angle of incidence
        self.cos_theta = np.cos(self.theta_z)*np.cos(self.beta) + \
            np.sin(self.theta_z)*np.sin(self.beta)*np.cos(self.gamma_s-self.gamma)

        # Calculate shading coordinates
        self.coordinate_s = sh.shading_tilt(self.coordinate_p, self.gamma_s, self.theta_z, self.n_row, self.num_int)



class AVSystemEWFT(AVSystem):

    def set_meshgrid(self, dA=0.1):
        
        # %% Set starting and ending points of meshgrid

        # [m] starting point of length to calculate average
        self.A_i = self.L_c + 1/4*self.A

        # [m] ending point of length to calculate average
        self.A_f = self.L_c + 3/4*self.A

        # [m] starting point of width to calculate average
        self.W_i = self.L_c + self.W_p +1/2*self.W_r
        
        # [m] ending point of width to calculate average
        self.W_f = (self.n_row-1)*(self.W_p+self.W_r) + self.L_c - 1/2*self.W_r

        # [m] length of study area
        self.A_study = self.A + 2*self.L_c  
        
        # [m] length of study area
        self.W_study = self.W_p * self.n_row + (self.n_row-1)* self.W_r + 2*self.L_c
        
        # %% Create meshgrid for shading and irradiation calculation
        
        # Mesh size
        self.dA = dA

        # Number of meshgrid points in length
        self.n_sA = int(self.A_study/self.dA)

        # Number of meshgrid points in width
        self.n_sW = int(self.W_study/self.dA)

        # Starting point of length to calculate average
        self.n_Ai = int(self.A_i/self.dA+1)

        # Ending point of length to calculate average
        self.n_Af = int(self.A_f/self.dA+1)

        # Starting point of width to calculate average
        self.n_Wi = int(self.W_i/self.dA+1)

        # Ending point of width to calculate average
        self.n_Wf = int(self.W_f/self.dA+1)

    def set_surface_angle(self, tilt_angle):
        
        # Tilt angle of solar panel
        self.beta = radians(tilt_angle)

        # Surface azimuth angle of solar panel
        if tilt_angle >= 0:
            self.gamma = radians(-90)
        else:
            self.gamma = radians(90)

    def construct_site(self, site_location, L_st):

        # Site location with latitude and longitude
        self.site_location = site_location

        # Standard longitude
        self.L_st = L_st
        
        # Calculate panel coordinates
        self.coordinate_p = sc.site_construction_ew_ft(
            self.W_r, self.A, self.H, 
            self.n_row, self.W_p, self.beta, self.L_c)
        
        # Calculate solar zenith angle and solar azimuth angle
        self.theta_z, self.gamma_s = sa.solar_angle(radians(self.site_location.latitude), 
                                                    self.site_location.longitude, self.L_st, self.num_int)
        
        # Calculate cosine of the angle of incidence
        self.cos_theta = np.cos(self.theta_z)*np.cos(self.beta) + \
            np.sin(self.theta_z)*np.sin(self.beta)*np.cos(self.gamma_s-self.gamma)

        # Calculate shading coordinates
        self.coordinate_s = sh.shading_a(self.coordinate_p, self.gamma_s, self.theta_z, self.n_row, self.num_int)

class AVSystemEWSAT(AVSystem):

    def set_meshgrid(self, dA=0.1):
        
        # %% Set starting and ending points of meshgrid

        # [m] starting point of length to calculate average
        self.A_i = self.L_c + 1/4*self.A

        # [m] ending point of length to calculate average
        self.A_f = self.L_c + 3/4*self.A

        # [m] starting point of width to calculate average
        self.W_i = self.L_c + self.W_p +1/2*self.W_r
        
        # [m] ending point of width to calculate average
        self.W_f = (self.n_row-1)*(self.W_p+self.W_r) + self.L_c - 1/2*self.W_r

        # [m] length of study area
        self.A_study = self.A + 2*self.L_c  
        
        # [m] length of study area
        self.W_study = self.W_p * self.n_row + (self.n_row-1)* self.W_r + 2*self.L_c
        
        # %% Create meshgrid for shading and irradiation calculation
        
        # Mesh size
        self.dA = dA

        # Number of meshgrid points in length
        self.n_sA = int(self.A_study/self.dA)

        # Number of meshgrid points in width
        self.n_sW = int(self.W_study/self.dA)

        # Starting point of length to calculate average
        self.n_Ai = int(self.A_i/self.dA+1)

        # Ending point of length to calculate average
        self.n_Af = int(self.A_f/self.dA+1)

        # Starting point of width to calculate average
        self.n_Wi = int(self.W_i/self.dA+1)

        # Ending point of width to calculate average
        self.n_Wf = int(self.W_f/self.dA+1)

    def set_surface_angle(self, tilt_morning, tilt_noon, tilt_evening):
        
        # Tilt angle of solar panel in the morning in radians
        self.beta_m = radians(tilt_morning)

        # Tilt angle of solar panel at noon
        self.beta_n = radians(tilt_noon)

        # Tilt angle of solar panel in the evening
        self.beta_e = radians(tilt_evening)

        # Calculate hourly tilt angle of solar panel
        self.beta = np.zeros((24*self.num_int, 1))
        self.gamma = np.zeros((24*self.num_int, 1))

        # Loop though each hour of the day
        for j in range(24*self.num_int): # 288 = 24*12
                
            if 7*self.num_int < j <= 13*self.num_int: # old: j < 12, 144 = 12*12
                # Morning
                self.beta[j] = self.beta_n + (self.beta_m - self.beta_n)/(6*self.num_int) * (13*self.num_int-j) # old: 6, 12. 72 = 6*12, 144 = 12*12
                self.gamma[j] = radians(-90)
            elif 13*self.num_int < j <= 19*self.num_int:
                # Afternoon
                self.beta[j] = self.beta_n + (self.beta_e - self.beta_n)/(6*self.num_int) * (j-13*self.num_int) # old: 6, 12. 72 = 6*12, 144 = 12*12
                self.gamma[j] = radians(90)
            else:
                # Before sunrise and after sunset
                self.beta[j] = 0
                self.gamma[j] = 0

        # for j in range(24): # 288 = 24*12
        #     if j < 6 or j > 18 or j == 12: # j < 72 or j > 228 or 144 <= j <= 156: old: j < 6 or j > 18 or j == 12, 72 = 6*12, 228 = 19*12, 144 = 12*12, 156 = 13*12 (12:00-13:00, 1 hour, PV panels stay flat)
        #         self.beta[j] = 0
        #         self.gamma[j] = 0
        #     elif j < 12: # old: j < 12, 144 = 12*12
        #         self.beta[j] = self.beta_n + (self.beta_m - self.beta_n)/6 * (12-j) # old: 6, 12. 72 = 6*12, 144 = 12*12
        #         self.gamma[j] = radians(-90)
        
        #     else:
        #         self.beta[j] = self.beta_n + (self.beta_e - self.beta_n)/6 * (j-12) # old: 6, 12. 72 = 6*12, 144 = 12*12
        #         self.gamma[j] = radians(90)

    def construct_site(self, site_location, L_st):

        # Site location with latitude and longitude
        self.site_location = site_location

        # Standard longitude
        self.L_st = L_st
        
        # Calculate panel coordinates
        self.coordinate_p = sc.site_construction_ew_sat(
            self.W_r, self.A, self.H, 
            self.n_row, self.W_p, self.beta,
            self.L_c, self.beta_n, self.num_int)
        
        # Calculate solar zenith angle and solar azimuth angle
        self.theta_z, self.gamma_s = sa.solar_angle(radians(self.site_location.latitude), 
                                                    self.site_location.longitude, 
                                                    self.L_st, self.num_int)
        
        # Calculate cosine of the angle of incidence
        self.cos_theta = np.cos(self.theta_z)*np.cos(self.beta) + \
            np.sin(self.theta_z)*np.sin(self.beta)*np.cos(self.gamma_s-self.gamma)

        # Calculate shading coordinates
        self.coordinate_s = sh.shading_tilt(self.coordinate_p, self.gamma_s, self.theta_z, self.n_row, self.num_int)
