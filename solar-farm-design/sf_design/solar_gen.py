"""
Calculate solar generation from facilities in NY-Sun or hypothetical scenarios
using WRF meteorology and PVLib-python functions.

TODO: remove the dependency of optwrf kdTree
TODO: PVsystem inverter and module parameters should be passed in as a dictionary
TODO: better documentation


Known Issues/Wishlist:
"""
import pandas as pd
import pvlib
import logging
from pvlib.pvsystem import PVSystem, Array, FixedMount, SingleAxisTrackerMount
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import warnings
from sf_design.avsystem import (AVSystemEWFT, AVSystemEWSAT,
                                AVSystemNSFT, AVSystemNSSAT)

warnings.filterwarnings(action='ignore', module='pvfactors')


def _build_fixed_tilt_system(num_array, surface_tilt_list, surface_azimuth_list,
                             module_parameters_list, inverter_parameters,
                             temperature_model='sapm', array_type='open_rack_glass_glass',
                             module_type='glass_glass', modules_per_string=1,
                             strings_per_inverter=1, albedo=0.2, name=None,
                             num_int=None):
    """
    Internal helper used by both the current and the legacy fixed-tilt APIs.
    """

    # Convert angle to 0-360 degrees
    surface_azimuth_list = [x % 360 for x in surface_azimuth_list]

    # Check if all the arrays have the same surface azimuth
    if not all(x == surface_azimuth_list[0] for x in surface_azimuth_list):
        logging.error('The surface azimuths of the arrays must be the same.')
        raise ValueError('The surface azimuths of the arrays must be the same.')

    # Check if all the arrays have the same surface tilt
    if not all(x == surface_tilt_list[0] for x in surface_tilt_list):
        logging.warning('The surface tilts of the arrays are different.')
        raise RuntimeWarning('The surface tilts of the arrays must be the same.')

    # Get temperature model parameters
    temperature_model_parameters = (
        TEMPERATURE_MODEL_PARAMETERS[temperature_model][array_type])

    # Create a list solar panel arrays
    array_list = list()
    for i in range(num_array):
        mount = FixedMount(surface_tilt=surface_tilt_list[i],
                           surface_azimuth=surface_azimuth_list[i])
        array = Array(mount=mount,
                      albedo=albedo,
                      module_parameters=module_parameters_list[i],
                      temperature_model_parameters=temperature_model_parameters,
                      module_type=module_type,
                      modules_per_string=modules_per_string,
                      name=f'Array {i+1}')
        array_list.append(array)

    # Create an AVSystem object according to the orientation
    if surface_azimuth_list[0] == 180 or surface_azimuth_list[0] == 0:
        system = AVSystemNSFT(arrays=array_list,
                              inverter_parameters=inverter_parameters,
                              strings_per_inverter=strings_per_inverter,
                              name=name)
    elif surface_azimuth_list[0] == 90 or surface_azimuth_list[0] == 270:
        system = AVSystemEWFT(arrays=array_list,
                              inverter_parameters=inverter_parameters,
                              strings_per_inverter=strings_per_inverter,
                              name=name)
    else:
        raise ValueError('Surface azimuth must resolve to 0, 90, 180, or 270 degrees.')

    if num_int is not None:
        system.set_timeinterval(num_int)

    return system


def fixed_tilt_monofacial_system(weather_df, num_array, surface_tilt_list, surface_azimuth_list,
                                module_parameters_list, inverter_parameters,
                                temperature_model='sapm', array_type='open_rack_glass_glass',
                                module_type='glass_glass', modules_per_string=1, 
                                strings_per_inverter=1, albedo=0.2, name=None):
    """
    Create a fixed tilt PV system with multiple arrays.

    Parameters
    ----------
    num_array : int
        Number of arrays in the system.
    surface_tilt_list : list
        List of surface tilt angles for each array.
    surface_azimuth_list : list
        List of surface azimuth angles for each array.
    module_parameters_list : list
        List of module parameters for each array.
    inverter_parameters : dict
        Inverter parameters.
    temperature_model : str
        Temperature model to use.
    array_type : str
        Array type to use.
    module_type : str
        Module type to use.
    modules_per_string : int
        Number of modules per string.
    strings_per_inverter : int
        Number of strings per inverter.
    albedo : float
        Ground surface albedo.

    Returns
    -------
    system : sf_design.avsystem.AVSystem
        Agrivoltaic system object.

    """
    if isinstance(weather_df, pd.DataFrame):
        times = weather_df.index
        dt = times.freq.delta.total_seconds() # time interval in seconds
        num_int = int(3600/dt) # Number of intervals in an hour
    else:
        raise TypeError('weather must be a pandas Series or DataFrame.')

    return _build_fixed_tilt_system(
        num_array=num_array,
        surface_tilt_list=surface_tilt_list,
        surface_azimuth_list=surface_azimuth_list,
        module_parameters_list=module_parameters_list,
        inverter_parameters=inverter_parameters,
        temperature_model=temperature_model,
        array_type=array_type,
        module_type=module_type,
        modules_per_string=modules_per_string,
        strings_per_inverter=strings_per_inverter,
        albedo=albedo,
        name=name,
        num_int=num_int)


def fixed_tilt_system(*args, **kwargs):
    """
    Backward-compatible alias for monofacial fixed-tilt systems.

    Older examples, docs, and tests in this repository still import
    ``fixed_tilt_system``. Keep that entry point available so the project
    works from the current checkout without requiring notebook edits first.
    """

    if args:
        return fixed_tilt_monofacial_system(*args, **kwargs)

    if 'weather_df' in kwargs:
        return fixed_tilt_monofacial_system(**kwargs)

    return _build_fixed_tilt_system(**kwargs)


def single_axis_tracking_monofacial_system(weather_df, num_array, axis_tilt, axis_azimuth,
                                           max_angle, module_parameters_list, inverter_parameters,
                                           backtrack=True, gcr=0.4, albedo=0.2,
                                           temperature_model='sapm',
                                           array_type='open_rack_glass_glass',
                                           module_type='glass_glass',
                                           modules_per_string=1,
                                           strings_per_inverter=1, name=None):
    """
    Create a monofacial single-axis tracking PV system without bifacial irradiance modeling.

    This helper is intended for lightweight application workflows where the
    tracking kinematics matter, but the pvfactors-based bifacial stack is not
    required.
    """

    if isinstance(weather_df, pd.DataFrame):
        times = weather_df.index
        dt = times.freq.delta.total_seconds()
        num_int = int(3600 / dt)
    else:
        raise TypeError('weather must be a pandas Series or DataFrame.')

    temperature_model_parameters = (
        TEMPERATURE_MODEL_PARAMETERS[temperature_model][array_type])

    sat_mount = SingleAxisTrackerMount(axis_tilt=axis_tilt,
                                       axis_azimuth=axis_azimuth,
                                       max_angle=max_angle,
                                       backtrack=backtrack,
                                       gcr=gcr)

    array_list = list()
    for i in range(num_array):
        array = Array(mount=sat_mount,
                      albedo=albedo,
                      module_parameters=module_parameters_list[i],
                      temperature_model_parameters=temperature_model_parameters,
                      module_type=module_type,
                      modules_per_string=modules_per_string,
                      name=f'Array {i+1}')
        array_list.append(array)

    if axis_azimuth == 180 or axis_azimuth == 0:
        system = AVSystemEWSAT(arrays=array_list,
                               inverter_parameters=inverter_parameters,
                               strings_per_inverter=strings_per_inverter,
                               name=name)
    elif axis_azimuth == 90 or axis_azimuth == 270:
        system = AVSystemNSSAT(arrays=array_list,
                               inverter_parameters=inverter_parameters,
                               strings_per_inverter=strings_per_inverter,
                               name=name)
    else:
        raise ValueError('Axis azimuth must resolve to 0, 90, 180, or 270 degrees.')

    system.set_timeinterval(num_int)
    return system


def run_modelchain(system, site_location, weather_df,
                   aoi_model='no_loss', spectral_model='no_loss'):
    """
    Run a modelchain object to calculate solar generation.

    Parameters
    ----------
    system : pvlib.pvsystem.PVSystem
        PV system object.
    site_location : pvlib.location.Location
        Location object.
    weather_df : pandas.DataFrame
        Weather data.
    aoi_model : str
        AOI model to use.
    spectral_model : str
        Spectral model to use.

    Returns
    -------
    solar_gen_ac : pandas.DataFrame
        AC solar generation.
    solar_gen_dc : pandas.DataFrame
        DC solar generation.

    """
    # Define and run a `ModelChain` object to calculate modeling intermediates.
    mc = ModelChain(system, site_location,
                    aoi_model=aoi_model,
                    spectral_model=spectral_model)
    
    # Run the model
    mc.run_model(weather_df)

    # DC power in arrays W
    solar_gen_dc = pd.DataFrame()
    for i in range(len(system.arrays)):
        solar_gen_dc[f'array_{i+1}'] = mc.results.dc[i]

    # Total AC power in W
    solar_gen_ac = mc.results.ac

    return solar_gen_ac, solar_gen_dc


def single_axis_tracking_system(site_location, weather_df,
                                num_array, axis_tilt, axis_azimuth, max_angle,
                                module_parameters_list, inverter_parameters,
                                backtrack=True, gcr=0.4, albedo=0.2,
                                bifacial=True, bifaciality=0.75, pvrow_height=2,
                                pvrow_width=2, n_pvrows=6, index_observed_pvrow=1,
                                temperature_model='sapm', array_type='open_rack_glass_glass',
                                module_type='glass_glass', modules_per_string=1, 
                                strings_per_inverter=1, name=None):
    """
    Create a single axis tracking PV system with multiple arrays.

    Parameters
    ----------
    site_location : pvlib.location.Location
        Location object.
    weather_df : pandas.DataFrame
        Weather data.
    num_array : int
        Number of arrays in the system.
    axis_tilt : float
        Axis tilt angle.
    axis_azimuth : float
        Axis azimuth angle.
    max_angle : float
        Maximum angle of rotation.
    module_parameters_list : list
        List of module parameters for each array.
    inverter_parameters : dict
        Inverter parameters.
    backtrack : bool
        Whether to use backtracking.
    gcr : float
        Ground coverage ratio.
    albedo : float
        Albedo.
    bifacial : bool
        Whether to use bifacial modeling.
    bifaciality : float
        Bifaciality.
    pvrow_height : float
        PV row height.
    pvrow_width : float
        PV row width.
    n_pvrows : int
        Number of PV rows.
    index_observed_pvrow : int
        Index of observed PV row.
    temperature_model : str
        Temperature model to use.
    array_type : str
        Array type to use.
    module_type : str
        Module type to use.
    modules_per_string : int
        Number of modules per string.
    strings_per_inverter : int
        Number of strings per inverter.
    name : str
        Name of the system.

    Returns
    -------
    system : pvlib.pvsystem.PVSystem
        PV system object.
    weather_irrad : pandas.DataFrame
        Weather and irradiance data.
    orientation : pandas.DataFrame
        Orientation of the tracker.

    """

    # Get temperature model parameters
    temperature_model_parameters = (
        TEMPERATURE_MODEL_PARAMETERS[temperature_model][array_type])

    

    if isinstance(weather_df, pd.DataFrame):
        # Get solar position of the site
        times = weather_df.index
        solar_position = site_location.get_solarposition(times)
        dt = times.freq.delta.total_seconds() # time interval in seconds
        num_int = int(3600/dt) # number of intervals in an hour
    else:
        raise TypeError('weather must be a pandas Series or DataFrame.')

    # load solar position and tracker orientation for use in pvsystem object
    sat_mount = SingleAxisTrackerMount(axis_tilt=axis_tilt,
                                       axis_azimuth=axis_azimuth,
                                       max_angle=max_angle,
                                       backtrack=backtrack,
                                       gcr=gcr)

    # created for use in pvfactors timeseries
    orientation = sat_mount.get_orientation(solar_position['apparent_zenith'],
                                            solar_position['azimuth'])

    # Combine weather and irradiance data
    weather_irrad = get_bifacial_irradiance(weather_df, solar_position,
                                            orientation['surface_azimuth'],
                                            orientation['surface_tilt'],
                                            axis_azimuth, gcr,
                                            pvrow_height, pvrow_width, albedo,
                                            n_pvrows, index_observed_pvrow,
                                            bifacial, bifaciality)

    # Create a list of solar arrays
    array_list = list()
    for i in range(num_array):
        array = Array(mount=sat_mount,
                      module_parameters=module_parameters_list[i],
                      temperature_model_parameters=temperature_model_parameters,
                      module_type=module_type,
                      modules_per_string=modules_per_string,
                      name=f'Array {i+1}')
        array_list.append(array)

    # create an AVSystem object according to the orientation
    if axis_azimuth == 180 or axis_azimuth == 0:
        # East-West single axis tracking system
        system = AVSystemEWSAT(arrays=array_list,
                                inverter_parameters=inverter_parameters,
                                strings_per_inverter=strings_per_inverter,
                                name=name)
        system.set_timeinterval(num_int)
    elif axis_azimuth == 90 or axis_azimuth == 270:
        # North-South single axis tracking system
        system = AVSystemNSSAT(arrays=array_list,
                                inverter_parameters=inverter_parameters,
                                strings_per_inverter=strings_per_inverter,
                                name=name)
        system.set_timeinterval(num_int)
    else:
        logging.error('Axis azimuth must be 0, 90, 180, or 270.')
        raise ValueError('Axis azimuth must be 0, 90, 180, or 270.')

    return system, weather_irrad, orientation


def fixed_tilt_bifacial_system(site_location, weather_df,
                                num_array, surface_tilt_list, surface_azimuth_list,
                                module_parameters_list, inverter_parameters,
                                gcr=0.5, albedo=0.2,
                                bifacial=True, bifaciality=0.75, pvrow_height=2,
                                pvrow_width=4, n_pvrows=4, index_observed_pvrow=1,
                                temperature_model='sapm', array_type='open_rack_glass_glass',
                                module_type='glass_glasss', modules_per_string=1, 
                                strings_per_inverter=1, name=None):
    """
    Create a single axis tracking system with multiple arrays.

    Parameters
    ----------
    site_location : pvlib.location.Location
        Location object.
    weather_df : pandas.DataFrame
        Weather data.
    num_array : int
        Number of arrays in the system.
    surface_tilt_list : list
        List of surface tilt angles.
    surface_azimuth_list : list
        List of surface azimuth angles.
    module_parameters_list : list
        List of module parameters for each array.
    inverter_parameters : dict
        Inverter parameters.
    gcr : float
        Ground coverage ratio.
    albedo : float
        Albedo.
    bifacial : bool
        Whether to use bifacial modeling.
    bifaciality : float
        Bifaciality.
    pvrow_height : float
        PV row height.
    pvrow_width : float
        PV row width.
    n_pvrows : int
        Number of PV rows.
    index_observed_pvrow : int
        Index of observed PV row.
    temperature_model : str
        Temperature model to use.
    array_type : str
        Array type to use.
    module_type : str
        Module type to use.
    modules_per_string : int
        Number of modules per string.
    strings_per_inverter : int
        Number of strings per inverter.
    name : str
        Name of the system.

    Returns
    -------
    system : pvlib.pvsystem.PVSystem
        PV system object.
    weather_irrad : pandas.DataFrame
        Weather and irradiance data.
    orientation : pandas.DataFrame
        Orientation of the tracker.

    """
    # Get temperature model parameters
    temperature_model_parameters = (
        TEMPERATURE_MODEL_PARAMETERS[temperature_model][array_type])

    if isinstance(weather_df, pd.DataFrame):
        # Get solar position of the site
        times = weather_df.index
        solar_position = site_location.get_solarposition(times)
        dt = times.freq.delta.total_seconds() # time interval in seconds
        num_int = int(3600/dt) # number of intervals in an hour
    else:
        raise TypeError('weather must be a pandas Series or DataFrame.')

    array_list = list()
    weather_irrad_list = list()

    for i in range(num_array):
        surface_tilt = surface_tilt_list[i]
        surface_azimuth = surface_azimuth_list[i]
        axis_azimuth = surface_azimuth + 90

        # load solar position and tracker orientation for use in pvsystem object
        sat_mount = SingleAxisTrackerMount(axis_azimuth=axis_azimuth, gcr=gcr)

        # Combine weather and irradiance data
        weather_irrad = get_bifacial_irradiance(weather_df, solar_position,
                                                surface_azimuth,
                                                surface_tilt,
                                                axis_azimuth, gcr,
                                                pvrow_height, pvrow_width, albedo,
                                                n_pvrows, index_observed_pvrow,
                                                bifacial, bifaciality)
        weather_irrad_list.append(weather_irrad)

        array = Array(mount=sat_mount,
                      module_parameters=module_parameters_list[i],
                      temperature_model_parameters=temperature_model_parameters,
                      module_type=module_type,
                      modules_per_string=modules_per_string,
                      name=f'Array {i+1}')
        array_list.append(array)

    # Create an AVSystem object according to the orientation
    if axis_azimuth == 180 or axis_azimuth == 0:
        # East-west fixed tilt bifacial system
        system = AVSystemEWFT(arrays=array_list,
                            inverter_parameters=inverter_parameters,
                            strings_per_inverter=strings_per_inverter,
                            name=name)
        system.set_timeinterval(num_int)
    elif axis_azimuth == 90 or axis_azimuth == 270:
        system = AVSystemNSFT(arrays=array_list,
                            inverter_parameters=inverter_parameters,
                            strings_per_inverter=strings_per_inverter,
                            name=name)
        system.set_timeinterval(num_int)
    else:
        logging.error('Axis azimuth must be 0, 90, 180, or 270.')
        raise ValueError('Axis azimuth must be 0, 90, 180, or 270.')

    return system, weather_irrad_list


def run_modelchain_effective_irradiance(system, site_location, weather_df_list,
                                        aoi_model='no_loss', spectral_model='no_loss'):
    """
    Run a modelchain with effective irradiance.
    
    Parameters
    ----------
    system : pvlib.pvsystem.PVSystem
        PV system object.
    site_location : pvlib.location.Location
        Location object.
    weather_df_list : list
        List of weather data in `pandas.DataFrame`.
    aoi_model : str
        AOI model to use.
    spectral_model : str
        Spectral model to use.

    Returns
    -------
    solar_gen_ac : pandas.Series
        AC power in Watts.
    solar_gen_dc : pandas.DataFrame
        DC power in Watts.

    """

    # Define and run a `ModelChain` object to calculate modeling intermediates.
    mc = ModelChain(system, site_location,
                    aoi_model=aoi_model,
                    spectral_model=spectral_model)
    
    # Run the model
    mc.run_model_from_effective_irradiance(weather_df_list)

    # DC power in arrays W
    solar_gen_dc = pd.DataFrame()
    for i in range(len(system.arrays)):
        solar_gen_dc[f'array_{i+1}'] = mc.results.dc[i]

    # Total AC power in W
    solar_gen_ac = mc.results.ac

    return solar_gen_ac, solar_gen_dc


def get_bifacial_irradiance(weather_df, solar_position, surface_azimuth, surface_tilt,
                            axis_azimuth, gcr, pvrow_height, pvrow_width, albedo,
                            n_pvrows=3, index_observed_pvrow=1, bifacial=True, bifaciality=0.75):
    """
    Get bifacial irradiance.

    Parameters
    ----------
    weather_df : pandas.DataFrame
        Weather data.
    solar_position : pandas.DataFrame
        Solar position.
    surface_azimuth : float
        Surface azimuth angle.
    surface_tilt : float
        Surface tilt angle.
    axis_azimuth : float
        Axis azimuth angle.
    gcr : float
        Ground coverage ratio.
    pvrow_height : float
        PV row height.
    pvrow_width : float
        PV row width.
    albedo : float
        Albedo.
    n_pvrows : int
        Number of PV rows.
    index_observed_pvrow : int
        Index of observed PV row.
    bifacial : bool
        Whether to use bifacial modeling.
    bifaciality : float
        Bifaciality.

    Returns
    -------
    irrad : pandas.DataFrame
        Bifacial irradiance.

    """

    # Bi-facial irradiance calculation
    # get rear and front side irradiance from pvfactors transposition engine
    # explicitly simulate on pv array with 3 rows, with sensor placed in middle row
    # users may select different values depending on needs

    times = weather_df.index

    try:
        from pvlib.bifacial.pvfactors import pvfactors_timeseries
    except Exception as exc:
        raise ImportError(
            "Bifacial irradiance modelling requires the optional pvfactors-compatible "
            "stack. The web app's default monofacial workflow does not need it."
        ) from exc

    irrad = pvfactors_timeseries(solar_position['azimuth'],
                                 solar_position['apparent_zenith'],
                                 surface_azimuth,
                                 surface_tilt,
                                 axis_azimuth, times,
                                 weather_df['dni'], weather_df['dhi'],
                                 gcr, pvrow_height,
                                 pvrow_width, albedo,
                                 n_pvrows=n_pvrows,
                                 index_observed_pvrow=index_observed_pvrow)

    # turn into pandas DataFrame
    irrad = pd.concat(irrad, axis=1)

    if bifacial:
        # create bifacial effective irradiance using aoi-corrected timeseries values
        irrad['effective_irradiance'] = (
            irrad['total_abs_front'] + (irrad['total_abs_back'] * bifaciality))
    else:
        # Mono-facial effective irradiance
        irrad['effective_irradiance'] = irrad['total_abs_front']

    # Combine weather and irradiance data
    weather_irrad = pd.merge(
        weather_df, irrad, left_index=True, right_index=True)

    return weather_irrad
