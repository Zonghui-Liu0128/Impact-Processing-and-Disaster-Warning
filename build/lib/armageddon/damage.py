import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from .solver import Planet
from .locator import PostcodeLocator


def Calculate_r(x, *data_pressure):
    """
    The expression of the p(r), and the we will
    find the solver of this equation.

    Parameters
    ----------
    x: float
        the distance along the surface from
        the "surface zero point" which we will compute later

    data_pressure: the pressure from the wave


    Return
    ----------
    the expression/equation
    """
    pressure = data_pressure
    return 3.14e+11 * x ** (-1.3) + 1.8e+7 * x ** (-0.565) - pressure


def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.

    Parameters
    ----------

    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north
        of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels

    Returns
    -------

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying
        the blast radii for the input damage levels

    Examples
    --------

    >>> import armageddon
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,
                   'burst_distance': 90e3, 'burst_peak_dedz': 1e3,
                   'outcome': 'Airburst'}
    >>> armageddon.damage_zones(outcome, 52.79,
    -2.95, 135, pressures=[1e3, 3.5e3, 27e3, 43e3])
    """

    damrad = []
    lat_rad = np.deg2rad(lat)
    bearing_rad = np.deg2rad(bearing)
    R_p = 6371000
    E_k = outcome['burst_energy']
    r = outcome['burst_distance']
    Z_b = outcome['burst_altitude']

    # Computer the distances
    if isinstance(pressures, float):
        data_pressure = pressures
        alpha = fsolve(Calculate_r, [1], args=data_pressure)
        tmp = alpha * E_k ** (2 / 3) - Z_b ** 2
        if tmp < 0:
            damrad.append(0)
        else:
            damrad.append(float(tmp ** (1 / 2)))

    else:
        for i in range(len(pressures)):
            data_pressure = pressures[i]
            alpha = fsolve(Calculate_r, [1], args=data_pressure)
            tmp = alpha * E_k ** (2 / 3) - Z_b ** 2
            if tmp < 0:
                damrad.append(0)
            else:
                r_ele = tmp ** (1 / 2)
                damrad.append(float(r_ele))

    # Computer the latitude of the suface zero point
    c1 = np.sin(lat_rad) * np.cos(r / R_p)
    c2 = np.cos(lat_rad) * np.sin(r / R_p) * np.cos(bearing_rad)
    sin_phi = c1 + c2
    rad = np.arcsin(sin_phi)
    blat = rad * 180 / np.pi

    # Computer the longitude of the surface zero point
    c3 = (np.sin(bearing_rad) * np.sin(r / R_p) * np.cos(lat_rad))
    c4 = (np.cos(r / R_p) - np.sin(lat_rad) * sin_phi)
    tan_diff = c3 / c4
    rad_diff = np.arctan(tan_diff)
    diff = rad_diff * 180 / np.pi
    blon = diff + lon

    return float(blat), float(blon), damrad


fiducial_means = {'radius': 10, 'angle': 20, 'strength': 1e6,
                  'density': 3000, 'velocity': 19e3,
                  'lat': 51.5, 'lon': 1.5, 'bearing': -45.}
fiducial_stdevs = {'radius': 1, 'angle': 1, 'strength': 5e5,
                   'density': 500, 'velocity': 1e3,
                   'lat': 0.025, 'lon': 0.025, 'bearing': 0.5}


def impact_risk(planet, means=fiducial_means,
                stdevs=fiducial_stdevs,
                pressure=1.e3, nsamples=2, sector=True):
    """
    Perform an uncertainty analysis to calculate the risk for each affected
    UK postcode or postcode sector

    Parameters
    ----------
    planet: armageddon.Planet instance
        The Planet instance from which to solve the atmospheric entry

    means: dict
        A dictionary of mean input values for the uncertainty analysis. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``

    stdevs: dict
        A dictionary of standard deviations for each input value. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``

    pressure: float
        The pressure at which to calculate the damage zone for each impact

    nsamples: int
        The number of iterations to perform in the uncertainty analysis

    sector: logical, optional
        If True (default) calculate the risk for postcode sectors, otherwise
        calculate the risk for postcodes

    Returns
    -------
    risk: DataFrame
        A pandas DataFrame with columns for postcode (or postcode sector) and
        the associated risk. These should be called ``postcode`` or ``sector``,
        and ``risk``.
    """
    dam_post_total = []
    for i in range(nsamples):
        # Randomising each values
        radius = np.random.normal(means['radius'], stdevs['radius'])
        velocity = np.random.normal(means["velocity"], stdevs["velocity"])
        density = np.random.normal(means["density"], stdevs["density"])
        strength = np.random.normal(means["strength"], stdevs["strength"])
        angle = np.random.normal(means["angle"], stdevs["angle"])
        lat = np.random.normal(means["lat"], stdevs["lat"])
        lon = np.random.normal(means["lon"], stdevs["lon"])
        bearing = np.random.normal(means["bearing"], stdevs["bearing"])

        # Find the latitude and logitude of the surface zero point
        result = planet.solve_atmospheric_entry(radius, velocity, density,
                                                strength, angle,
                                                init_altitude=100e3,
                                                dt=0.05, radians=False)
        result1 = planet.calculate_energy(result)
        outcome = planet.analyse_outcome(result1)
        lat_0, lon_0, dam_list = damage_zones(outcome,
                                              lat, lon,
                                              bearing, pressures=pressure)

        # Get the postcode in the damage area
        location_0 = []
        location_0.append(lat_0)
        location_0.append(lon_0)
        postcodeLocator = PostcodeLocator()
        list_temp = np.array(dam_list)
        length = len(list_temp)
        for j in range(length):
            r = [list_temp[j]]
            dam_post = postcodeLocator.get_postcodes_by_radius(location_0,
                                                               r, sector)
            dam_post = np.array(dam_post)
            dam_post = dam_post.reshape(dam_post.shape[1])
            dam_post_total.extend(dam_post)

    # Count the probability of
    # each postcode wich will be damaged
    dam_post_total = np.array(dam_post_total)
    dam_post_unique, dam_post_prob = np.unique(
        np.array(dam_post_total), return_counts=True)
    dam_post_unique = dam_post_unique.tolist()
    dam_post_unique = np.array(dam_post_unique)
    dam_post_unique = list(dam_post_unique)
    dam_post_pop = postcodeLocator.get_population_of_postcode(
        [dam_post_unique], sector)

    post_pop_prob = dam_post_prob / nsamples
    risk = dam_post_pop * post_pop_prob
    dam_post_unique = np.array(dam_post_unique)
    dam_post_unique = dam_post_unique.reshape(dam_post_unique.shape[0], 1)
    risk = risk.reshape(risk.shape[1], 1)
    final = np.append(dam_post_unique, np.array(risk), axis=1)
    final_df = pd.DataFrame(final)
    if sector:
        final_df.set_axis(['sector', 'risk'], axis='columns', inplace=True)
    else:
        final_df.set_axis(['postcode', 'risk'], axis='columns', inplace=True)
    return final_df

