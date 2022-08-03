####################################################
## Authors: Franklin Ye, Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Date: Spring 2021
####################################################

import re
import time

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter

import pandas as pd
import numpy as np

"""
General purpose Locality Density code.
Curvature-aware distance measure details: https://geopy.readthedocs.io/en/stable/#module-geopy.distance
"""


# Words that preceed a unit number in an address. Need to be filtered out for geopy.
FILTER_WORDS = ["ste", "suite", "apt", "unit"]
regex_string = '|'.join([f"({s})" for s in FILTER_WORDS])
regex_string = f"( ({regex_string}).+)"


def make_progress_counter(total_len, print_percents=0.10):
    """
    returns a function that manages progress printing overhead when called
    params
      total_len: total calls to be made
      print_percents: percents to print progress at
    """
    counter = {"i": 0}
    interval = int(total_len * print_percents)
    def count():
        i = counter["i"]
        if i % interval == 0:
          print(f"{100 * i // total_len}%...", end='')
        counter.update({"i": i+1})
    return count


def add_missing_coords(df, missing_coords):
    """
    Takes in missing_coords, a dict of locator_address to coordinates, and fills them into df
    """
    for address, coord in missing_coords.items():
      df.loc[df["locator_address"] == address, "coord"] = coord
    return df


def create_address(addresses):
    """
    Requires: "Street Address", "City", "State", & "Zip Code" as columns to define unique addresses
    """
    addresses['Street Address'] = addresses['Street Address'].str.lower()
    addresses['City'] = addresses['City'].str.capitalize()
    addresses['Zip Code'] = addresses['Zip Code'].replace(to_replace='nan', value=pd.NA).fillna('')
    addresses['Zip Code'] = addresses['Zip Code'].str[:5]
    locator_address = addresses['Street Address'] + ' ' + addresses['City'] + ' ' + addresses['State'] + ' ' + addresses['Zip Code']
    addresses['locator_address'] = locator_address.str.split().str.join(' ')
    return addresses



def get_coord(df, address_info, fallback=True, print_percents=1/10, verbose=True, **kwargs):
    """
    Uses geopy to fetch the coordinates for each address in df.
    Uses a common Rate limited geocoder to protect against errors.
    Uses address_info to extract additional address info to test out various other combos of address information in case the geopy API doesn't like the original attempt
    Combos:
      Clean the address field of any potential unit info
      Try street + state + zip
      Try street + state
      Try street + zip
      If fallback=True, Try zip + state (no street info)
    fallback indicates whether to estimate the coordinates of unparsable adderesses using only the zipcode and state
    """
    geocoder = RateLimiter(Nominatim().geocode, **kwargs)
    print_prog = make_progress_counter(df.shape[0], print_percents)

    def get_coord_func(row):
        """
        retrive the coordinates (lat, long) for an address in a Series
        Somtimes geopy demands a very specific format for the address so we try various combos
        If all address combos fail, return None
        """
        if verbose:
            print_prog()

        address = row.loc["locator_address"]
        geo_address = geocoder(address)

        if not geo_address:
            row = address_info[address_info["locator_address"] == address].iloc[0]
            streetName = re.sub(regex_string, '', row["Street Address"])
            zipNum = row["Zip Code"]
            stateName = row["State"]
            geo_address = geo_address or geocoder(streetName + ' ' + stateName + ' ' + zipNum)
            geo_address = geo_address or geocoder(streetName + ' ' + stateName)
            geo_address = geo_address or geocoder(streetName + ' ' + zipNum)
            geo_address = geo_address or (fallback and geocoder(zipNum + ' ' + stateName))

        if not geo_address:
            return pd.NA
        return (geo_address.latitude, geo_address.longitude)

    return df.apply(get_coord_func, axis=1)



def get_LD(df_focus_group, df_competitors, id_col, time_period_col=False, max_dist=np.inf, print_percents=0.1, verbose=True):
    """
    Returns localized density (LD) scores for focus group & count of competition used.
    Each focus group's LD is an inverse sum of it's distances to the competitors group.
    Params
      df_focus_group: Calculates LDs for each row in this DataFrame
      df_competitors: DataFrame used as competitors for each df_focus_group's LD score
      id_col: column name uniquely identifying a business within focus and competitor groups.
        Used to potentially avoid including a competitor in it's own LD score.
      time_period_col (optional): column name used for specifying scope of competition
      max_dist (optional): upperbound on how far away competitor may be
      print_percents: print progress at specifed intervals
      verbose: boolean to print periodic updates
    Creates a table (tperiod_groups) that organizes all dispenseries by time period
    Then calculates the LD for each dispensery based on its respective time period
    Returns the series of LDs calculated for each dispensery
    Each LD ignores it's own location.
    All locations without a coordinate will be ignored.
    """
    assert 'coord' in df_focus_group.columns and 'coord' in df_competitors.columns, \
        "column 'coord' with coordinates must be present in both DataFrames"

    # if no specified time_period_col, set all rows to share common time period
    if not time_period_col:
        time_period_col = 'TIME_PERIOD'
        df_focus_group[time_period_col], df_competitors[time_period_col] = 1, 1

    df_competitors = df_competitors.set_index([id_col, time_period_col])
    print_prog = make_progress_counter(df_focus_group.shape[0], print_percents)

    # remove nulls
    df_competitors_clean = df_competitors[~pd.isnull(df_competitors["coord"])]
    period_groups_competition = df_competitors_clean.groupby(time_period_col)
    avalible_periods = period_groups_competition.groups.keys()


    def get_LD_func(row):
        if verbose:
          print_prog()

        if pd.isnull(row["coord"]):
          return np.nan

        # calculate distances for a dispensery time_period_col's competition
        tperiod = row[time_period_col]
        if tperiod not in avalible_periods:
            return [pd.NA, 0] # no competition avalible
        tperiod_group = period_groups_competition.get_group(tperiod)

        # ignore self, if revalant
        row_id = row[id_col], row[time_period_col]
        if row_id in tperiod_group:
            tperiod_group.drop(row_id, inplace=True)

        dist_to_curr = lambda coord: geodesic(eval(row["coord"]), eval(coord)).miles
        distances = tperiod_group["coord"].apply(dist_to_curr)
        distances = np.minimum(distances, max_dist) # Upperbound how large a distances
        return [np.sum(1 / (1 + distances)), distances.shape[0]]

    start_time = time.time()
    out = df_focus_group.apply(get_LD_func, axis=1, result_type='expand')
    print("\nExecuted in", (time.time() - start_time)//60, "minutes")
    return out.iloc[:, 0], out.iloc[:, 1]
