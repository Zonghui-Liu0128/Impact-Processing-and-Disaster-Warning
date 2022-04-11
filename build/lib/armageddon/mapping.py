import folium
import pandas as pd
from folium.plugins import HeatMap
import folium.plugins as plugins
import numpy as np

# define a list of colors
colors = ["blue", "green", "yellow", "orange", "red"]

# hospitals information
hospital_info = pd.read_csv('./armageddon/resources/Hospitals.csv',
                            encoding="ISO-8859-1")

# schools information
school_info = pd.read_csv('./armageddon/resources/primary_school.csv',
                          encoding="ISO-8859-1")

# sector information
sector_info = pd.read_csv('./armageddon/resources/sector_data.csv',
                          encoding="ISO-8859-1")


def plot_circle(lat, lon, radius, map=None, show_hospital=True,
                show_school=False, dynamic_graph=True, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float or a list
        radius of circle(s) to plot (m)
    map: folium.Map
        existing map object
    show_hospital: flag
        show hospitals information in the map or not
    show_school": flag
        show primaryschools information in the map or not
    dynamic_graph: flag
        show the dynamic heatmap or still heatmap

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> armageddon.plot_circle(52.79, -2.95, 1e3, map=None)

    >>> plot_circle(51.498356,-0.176894, [1e3, 3e3],
                    map=None, show_hospital = False,
                    show_school = False, dynamic_graph = False)

    >>> plot_circle(51.498356,-0.176894,
                    1e3, map=None, show_hospital = True,
                    show_school = False, dynamic_graph = False)
    """
    # tooltip ='Impact Center'

    if not map:

        # Creat a map
        map = folium.Map(location=[lat, lon],
                         control_scale=True,
                         width='100%',
                         zoom_start=13)

        # Add a Marker to show the Impact Center
        folium.Marker(location=[lat, lon],
                      popup=str(lat) + " " + str(lon),
                      tooltip='Impact Center',
                      control_scale=True,
                      icon=folium.Icon(color='red',
                      prefix='fa', icon='bolt')).add_to(map)

        # Add a Circle to show the Impact Range
        if type(radius) == float:
            folium.Circle([lat, lon],
                          radius,
                          popup='Influence Radiu: ' + str(radius) + 'm',
                          color='#3186cc',
                          fill=True,
                          fillOpacity=0.6,
                          **kwargs).add_to(map)

        elif type(radius) == list:
            for index in range(len(radius)):
                folium.Circle([lat, lon],
                              radius[-index],
                              popup='Influence Radiu: ' +
                              str(radius[-index]) + 'm',
                              color=colors[-index],
                              fill=True,
                              fillOpacity=0.6,
                              **kwargs).add_to(map)

        # Add the Dynamic heatmap to show the population flows
        # Because lack of the in-time data of population now
        if dynamic_graph is True:
            # I use random example number to let the popuation shaky
            # if you want a still graph, set input "dynamic_graph" to False
            Dynamic_DATA_move = pd.DataFrame({
                'lat': list(sector_info['Latitude']) * 24,
                'lon': list(sector_info['Longitude']) * 24,
                'pop': list(sector_info['Population']) * 24,
                'hour': np.repeat(list(range(1, 25)), len(sector_info))})

            Dynamic_DATA_move['pop'] = (
                Dynamic_DATA_move['pop']/100 *
                (np.random.rand(len(Dynamic_DATA_move))))

            data_move = []
            for shaking in range(1, 25):
                data_move.append(
                    Dynamic_DATA_move[Dynamic_DATA_move['hour'] == shaking]
                    [['lat', 'lon', 'pop']].values.tolist())

            hm = plugins.HeatMapWithTime(data_move, radius=36)
            hm.add_to(map)
        else:
            # Convert data format
            heatdata = sector_info[["Latitude", "Longitude",
                                    "Population"]].values.tolist()
            # add incidents to map
            HeatMap(heatdata, radius=24).add_to(map)

        if show_hospital is True:
            # Add hospitals on the map, together with their name and postcode
            for index, row in hospital_info.iterrows():
                folium.Marker([hospital_info.loc[index].Latitude,
                              hospital_info.loc[index].Longitude],
                              tooltip=hospital_info.loc[index][1],
                              popup=hospital_info.loc[index][6],
                              icon=folium.Icon(color='red', prefix='fa',
                                               icon='plus-square')).add_to(map)

        if show_school is True:
            # Add primary schools on the map + their name and postcode
            # Implementing schools will increase runtime dramatically
            for index, row in school_info.iterrows():
                folium.Marker([school_info.loc[index].Latitude,
                              school_info.loc[index].Longitude],
                              tooltip=school_info.loc[index][1],
                              popup="TEL: " + str(school_info.loc[index][4]),
                              icon=folium.Icon(color='blue', prefix='fa',
                                               icon='users')).add_to(map)

    return map
