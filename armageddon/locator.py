"""Module dealing with postcode information."""
import numpy as np
import pandas as pd


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).
    Parameters
    ----------
    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points)
    Returns
    -------
    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)
    Examples
    --------
    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)}
    >>> with numpy.printoptions(formatter={'all', fmt}):
        print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [1.286e+05 6.378e+04]
    """
    if np.ndim(latlon1) == 1:
        latlon1 = [latlon1]

    if np.ndim(latlon2) == 1:
        latlon2 = [latlon2]

    if type(latlon1) != list:
        latlon1 = latlon1.tolist()

    if type(latlon2) != list:
        latlon2 = latlon2.tolist()

    # Radius of earth in km
    R = 6371.0 * 1000
    distance_array = []
    if (type(latlon1[0]) == list and type(latlon2[0]) == int
            or type(latlon2[0]) == float):
        distance_array = np.zeros(shape=(len(latlon1), 1))
        for count, point1 in enumerate(latlon1):
            # Distance in radians between two lat longs
            dlon = np.radians(latlon2[1]) - np.radians(point1[1])
            dlat = np.radians(latlon2[0]) - np.radians(point1[0])

            # Calculations
            a = (np.sin(dlat / 2) ** 2 + np.cos(np.radians(point1[0])) *
                 np.cos(np.radians(latlon2[0])) * np.sin(dlon / 2) ** 2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            distance = R * c
            distance_array[count] = distance

    elif type(latlon1[0]) == list and type(latlon2[0]) == list:
        distance_array = np.zeros(shape=(len(latlon1), len(latlon2)))
        for count1, point1 in enumerate(latlon1):
            for count2, point2 in enumerate(latlon2):
                # Distance in radians between two lat longs
                dlon = np.radians(point2[1]) - np.radians(point1[1])
                dlat = np.radians(point2[0]) - np.radians(point1[0])

                # Calculations
                a = (np.sin(dlat / 2) ** 2 + np.cos(np.radians(point1[0])) *
                     np.cos(np.radians(point2[0])) * np.sin(dlon / 2) ** 2)
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

                distance = R * c
                distance_array[count1][count2] = distance

    elif (type(latlon1[0]) == int or
          type(latlon1[0]) == float and type(latlon2[0])) == list:
        distance_array = np.zeros(shape=(1, (len(latlon2))))
        for count, point2 in enumerate(latlon2):
            # Distance in radians between two lat longs
            dlon = np.radians(point2[1]) - np.radians(latlon1[1])
            dlat = np.radians(point2[0]) - np.radians(latlon1[0])

            # Calculations
            a = (np.sin(dlat / 2) ** 2 + np.cos(np.radians(latlon1[0])) *
                 np.cos(np.radians(point2[0])) * np.sin(dlon / 2) ** 2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            distance = R * c
            distance_array[0][count] = distance

    elif (type(latlon1[0]) == int or type(latlon1[0]) == float and
          type(latlon2[0]) == int or type(latlon2[0]) == float):

        distance_array = np.zeros(shape=(1))
        # Distance in radians between two lat longs
        dlon = np.radians(latlon2[1]) - np.radians(latlon1[1])
        dlat = np.radians(latlon2[0]) - np.radians(latlon1[0])

        # Calculations
        a = (np.sin(dlat / 2) ** 2 + np.cos(np.radians(latlon1[0])) *
             np.cos(np.radians(latlon2[0])) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * c
        distance_array[0] = distance

    return distance_array


class PostcodeLocator(object):
    """Class to interact with a postcode database file."""

    def __init__(self,
                 postcode_file="./armageddon/resources/\
full_postcode_data.csv",
                 census_file="./armageddon/resources/\
population_by_postcode_sector.csv",
                 norm=great_circle_distance):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .csv file containing census data by postcode sector.

        norm : function
            Python function defining the distance between points in
            latitude-longitude space.

        """
        self.norm = norm
        self.postcode_file = postcode_file
        self.census_file = census_file
        self.Rp = 6371000
        self.postcode_pd = pd.read_csv("""./armageddon/resources/pc_pop.csv""")

    def get_postcodes_by_radius(self, X, radii, sector=False):
        """
        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X
        sector : bool, optional
            if true return postcode_pd sectors, otherwise postcode_pd units
        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than
            the elements of radii to the location X.
        """

        # data frame from read.csv
        postcode_pd = self.postcode_pd
        latX_rad = np.radians(X[0])
        lonX_rad = np.radians(X[1])
        result = []

        for r in radii:
            # calculate the outter rec range
            res = r / self.Rp
            max_lat_out, min_lat_out = np.degrees(latX_rad + res), np.degrees(
                latX_rad - res)
            res = r / (self.Rp * np.cos(latX_rad))
            max_lon_out, min_lon_out = np.degrees(lonX_rad + res), np.degrees(
                lonX_rad - res)

            # calculate the inner square
            r1 = r / np.sqrt(2)
            res = r1 / self.Rp
            max_lat_in, min_lat_in = np.degrees(latX_rad + res), np.degrees(
                latX_rad - res)
            res = r1 / (self.Rp * np.cos(latX_rad))
            max_lon_in, min_lon_in = np.degrees(lonX_rad + res), np.degrees(
                lonX_rad - res)

            # get inner square
            postcode_skip_check = postcode_pd[
                postcode_pd.Latitude.ge(min_lat_in)
                & postcode_pd.Latitude.le(max_lat_in)
                & postcode_pd.Longitude.ge(min_lon_in)
                & postcode_pd.Longitude.le(max_lon_in)]

            # get outter square
            postcode_to_check = postcode_pd[
                postcode_pd.Latitude.ge(min_lat_out)
                & postcode_pd.Latitude.le(max_lat_out)
                & postcode_pd.Longitude.ge(min_lon_out)
                & postcode_pd.Longitude.le(max_lon_out)]
            # Manipulation
            postcode_skip_check = \
                postcode_skip_check.append(postcode_skip_check)
            postcode_skip_check = \
                postcode_skip_check.drop_duplicates(keep=False)

            if sector:
                tem = postcode_skip_check['Postcode'].str[:-2].tolist()
            else:
                tem = postcode_skip_check['Postcode'].tolist()

            for index, row in postcode_to_check.iterrows():
                if self.norm([X[0], X[1]], [row['Latitude'],
                             row['Longitude']]) <= r:
                    if not sector:
                        tem.append(row['Postcode'])
                    else:
                        tem.append(row['Postcode'][:-2])
            result.append(list(set(tem)))
        return result

    def get_population_of_postcode(self, postcodes, sector=False):
        """
        Return populations of a list of postcode units or sectors.

        Parameters
        ----------
        postcodes : list of lists
            list of postcode units or postcode sectors
        sector : bool, optional
            if true return populations for postcode sectors,
            otherwise postcode units

        Returns
        -------
        list of lists
            Contains the populations of input postcode units or sectors


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_population_of_postcode
            ([['SW7 2AZ','SW7 2BT','SW7 2BU','SW7 2DD']])
        >>> locator.get_population_of_postcode([['SW7  2']], True)
        """

        # Create empty dataframe
        Get_population = pd.DataFrame()
        Population_List = []
        for count, postcode in enumerate(postcodes):
            if postcode == [] or postcode == [[]]:
                Population_List.append([])
                continue
            # Fill empty dataframe
            if count > 0:
                temp_dataframe = pd.DataFrame()
                temp_dataframe["List_{}".format(count)] = postcode
                Get_population = pd.concat(
                                    [Get_population,
                                     temp_dataframe["List_{}".format(count)]],
                                    axis=1)
            else:
                Get_population["List_{}".format(count)] = postcode

            try:
                Get_population["List_{}".format(count)].str.len()
            except AttributeError:
                continue
            if sector:
                # Read sector csv file
                Population_dataframe = pd.read_csv(self.census_file)

                Mod_PC = []

                for pc in Get_population["List_{}".format(count)].dropna():
                    if pc.isspace() is False:
                        pc = pc[:4] + ' ' + pc[-1]

                    elif (len(pc) == 5 and pc.find(' ') is True and
                            pc.find('  ') is False):
                        pc = pc.replace(" ", "  ")

                    elif (len(pc) == 5 and pc.find('  ') is True):
                        pc = pc.replace("  ", "   ")
                    Mod_PC.append(pc)
                temp_dataframe = pd.DataFrame()
                temp_dataframe["district_{}".format(count)] = Mod_PC
                Get_population = pd.concat(
                    [Get_population,
                        temp_dataframe["district_{}".format(count)]],
                    axis=1)

                Get_population["district_{}".format(count)] = (
                    Get_population["district_{}".format(count)]
                    .drop_duplicates()
                )
                new_frame = pd.DataFrame()
                new_frame_2 = pd.DataFrame()
                Population_dataframe["isin"] = (
                    Population_dataframe["geography"].isin(
                        Get_population["district_{}".format(count)].tolist()))
                new_frame["Population"] = (
                    Population_dataframe["Variable: All usual\
 residents; measures: Value"]
                    [Population_dataframe["isin"]])
                new_frame["Codes"] = (
                    Population_dataframe["geography"]
                    [Population_dataframe["isin"]])
                new_frame_2["Original"] = Mod_PC
                new_frame_2["difference"] = (
                    Get_population["district_{}".
                                   format(count)].isin(new_frame["Codes"]))
                new_frame_2["difference"] = (
                    new_frame_2["difference"].astype(int))

                i = 0
                list_1 = new_frame["Population"].tolist()
                list_2 = []
                for x in new_frame_2["difference"]:
                    if x == 1:
                        x = list_1[i]
                        i += 1
                    list_2.append(x)

                new_frame = new_frame.reset_index()
                new_frame = new_frame.drop("index", axis=1)
                new_frame = new_frame.fillna(0)
                Population_List.append(list_2)

            if not sector:
                # No need to format strings

                # Read full population file
                Population_dataframe = pd.read_csv(self.postcode_file)

                # Formatting the postcode to fit the new csv file:
                Get_population["List_{}".format(count)] = (
                    Get_population["List_{}".format(count)]
                    .str.replace(" ", ""))
                Population_dataframe["Postcode"] = (
                    Population_dataframe["Postcode"].str.replace(" ", ""))

                # Create and new dataframe with simimlarities between dataframe
                new_frame = pd.DataFrame()
                Population_dataframe["isin"] = (
                    Population_dataframe["Postcode"]
                    .isin(Get_population["List_{}".format(count)].tolist()))
                new_frame["Population"] = (
                    Population_dataframe["Population"]
                    [Population_dataframe["isin"]])
                new_frame["Codes"] = (
                    Population_dataframe["Postcode"]
                    [Population_dataframe["isin"]])
                new_frame = new_frame.reset_index()
                new_frame = new_frame.drop("index", axis=1)
                new_frame = new_frame.fillna(0)

                # Make list of lists
                Population_List.append(new_frame["Population"].tolist())

        return Population_List
