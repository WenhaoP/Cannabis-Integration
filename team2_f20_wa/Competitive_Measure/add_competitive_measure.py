####################################################
## Authors: Franklin Ye, Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Date: Spring 2021
####################################################

import re
import time
import os

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter

import pandas as pd
import numpy as np

import utils as cmu
# from .missing_coords import MISSING_DATA # not importing properly, so manually storing here

# from missing_coords.py
MISSING_DATA = {
  '19127 smokey point blvd bldg 1 Arlington WA 98223':        '(48.17053334205619, -122.18784541724916)',
  '193161 hwy 101 Forks WA 98331':                            '(47.96854221356822, -124.4039346117107)',
  '203 kinwood st se Lacey WA 98503':                         '(47.05087645062443, -122.79155758290705)',
  '2733 4th ave s 1st floor Seattle WA 98134':                '(47.57895282648133, -122.32950108658585)',
  '28120 hwy 410 e unit a8 Buckley WA 98321':                 '(47.15898317012183, -122.05274098660038)',
  '282023 hwy 101 Port townsend WA 98368':                    '(47.994375237662666, -122.88623971041517)',
  '36711 u.s. highway 12 Dayton WA 98328':                    '(46.31138618254054, -117.99416306943886)',
  '3772 sr 4 ste a Grays river WA 98621':                     '(46.35641147379252, -123.60949010196991)',
  '3831 highway 3 Shelton WA 98584':                          '(47.24868540843781, -123.0448829252292)',
  '410 ronlee ln nw a1 Olympia WA 98502':                     '(47.04974824554182, -122.99490957495917)',
  '41711 state route 2, #1 Gold bar WA 98251':                '(47.8442757914642, -121.67350827308313)',
  '5675 state route 12 ste 1 Elma WA 98541':                  '(46.93816331300561, -123.31047294427897)',
  '5809 112th st e bldg b Puyallup WA 98373':                 '(47.155067495065936, -122.35158967310694)',
  '6063b highway 291 Nine mile falls WA 99026':               '(47.820927972520124, -117.58866755774196)',
  '6230 197th sw ave Rochester WA 98579':                     '(46.80308179485759, -123.01460352895478)',
  '6323 ne bothel way Kenmore WA 98028':                      '(47.75866326699089, -122.2563946037701)',
  '6818 ne 4th plain blvd ste c Vancouver WA 98661':          '(45.64498628863629, -122.60227456151344)',
  '8040 ne day rd w bldg 3 ste 1 Bainbridge island WA 98110': '(47.68107796102876, -122.54209330192435)',
  '8142 highway 14 Lyle WA 98635':                            '(45.65812315279535, -121.19318613507984)',
  "8202 ne state hwy 104 ste 101 Kingston WA 98346":          '(47.811091452673786, -122.54012060177433)',
  "retail title certificate Lakewood WA 98499":               '(47.174863066069165, -122.50980616464203)',
  "1017 e methow valley hwy Twisp WA 98856":                  '(48.35860449775222, -120.1106618130911)',
  "120 ne canning dr Colville WA 99114":                      '(48.55524491150703, -117.91976644562857)',
  "3411 n capitol ave ste a Pasco WA 99301":                  '(46.26151271874826, -119.07961998802215)',
  "2733 4th ave s 1st floor Seattle WA 98134":                '(47.578909402126214, -122.32950108613103)',
  "708 rainier ave s #108/109 Seattle WA 98144":              '(47.59651119388283, -122.31129620332891)',
  "29 horizon flats rd ste 8 Winthrop WA 98862":              '(48.46577065826923, -120.18386199774325)',
  "1131 e state route 532 Camano island WA 98282":            '(48.23998993369884, -122.42129907076695)',
  "29297 hwy 410 e ste d Buckley WA 98321":                   '(47.15840804646947, -122.03743687316857)',
  "230 ne juniper st #101a Issaquah WA 98027":                '(47.53962345010246, -122.03180677078728)',
  "18420 68th ave s #102 Kent WA 98032":                      '(47.43749529004417, -122.24519600147993)',
  "21127 state rt 9 se Woodinville WA 98072":                 '(47.80563743548152, -122.14245723082237)',
  "410 ronlee ln nw a1 Olympia WA 98502":                     '(47.0497116943656, -122.99481301498244)',
  "12925 martin luther kingjr way Seattle WA 98178":          '(47.48665017945762, -122.25791471682344)',
  "8040 ne day rd w bldg 3 ste 1 Bainbridge island WA 98110": '(47.680998505096625, -122.54203965729192)',
  "1912 201st place se #202 Bothell WA 98012":                '(47.815340086453254, -122.20700200146905)',
  "5826 s kramer rd ste a Langley WA 98260":                  '(47.99910248695318, -122.45655245913618)',
  "6039 sw 197th ave Rochester WA 98579":                     '(46.80375815209636, -123.01147263218779)',
  "939 n callow ave # 100-b Bremerton WA 98312":              '(47.57089310355567, -122.653537657295)',
  "5200 172nd st ne f-101 Arlington WA 98223":                '(48.15186349396825, -122.16055830939008)',
  "retail title certificate Monroe WA 98272":                 '(47.85756230012783, -121.98650267661789)',
  "6733 state hwy 303 ne Bremerton WA 98311":                 '(47.62447012798018, -122.6295008303108)',
  "19315 bothell everett hwy #1 Bothell WA 98012":            '(47.82263058284023, -122.2064229726325)',
  "36711 u.s. highway 12 Dayton WA 98328":                    '(46.31114903571003, -117.99432400151206)',
  "9107 n country homes blvd #13 Spokane WA 99218":           '(47.741091679193815, -117.41370053030755)',
  "261 state hwy 28 west Soap lake WA 98851":                 '(47.370961746986204, -119.49560980148188)',
  "21502 e gilbert rd Otis orchards WA 99027":                '(47.70692049887425, -117.11701531681709)',
  "2008 n durry rd unit 2 Sprague WA 99032":                  '(47.208914267550064, -118.22607291683134)',
  "5809 112th st e bldg b Puyallup WA 98373":                 '(47.15491428335135, -122.35161113032436)',
  "3062 sw hwy 16 ste a Bremerton WA 98312":                  '(47.52523252399203, -122.69378725914993)',
  "9507 state route 302 nw # b Gig harbor WA 98329":          '(47.38926044274271, -122.66743835915379)',
  "19127 smokey point blvd bldg 1 Arlington WA 98223":        '(48.1704474792261, -122.18780250145859)',
  "6063b highway 291 Nine mile falls WA 99026":               '(47.820755079199564, -117.58872120146901)',
  "8125 birch bay square st #222 Blaine WA 98230":            '(48.93677442697616, -122.66846690143632)',
  "5978 highway 291 unit 6 Nine mile falls WA 99026":         '(47.81526957453812, -117.57604037263285)',
}

"""
Curvature-aware distance measure details: https://geopy.readthedocs.io/en/stable/#module-geopy.distance
"""

# used to upperbound the miles between any two locations, in case of incorrect locations
WA_DIAMETER = 433

# switch to use cached address-coordinate map, if available
USE_CACHED_COORDS = True

def preprocess_routine(buisness_data, fallback=False, cached_address_coords=None):
  """
  API being used doesn't like being called in large batches. 24K addresses.
  Solution is to only get Coordinates of the unique addresses.
  Use `get_coord_generator` to gracefully  handle errors in the API call.
    fallback indicates if zipcode-state should be used to approximate poorly-formated street addresses
    cached_address_coords is an optional dataframe with all data that would've been retrived from geopy. Will still intergrate MISSING_COORDS into this dataframe
  For proper coordinate extraction, df should have columns
    'Street Address', 'City', 'State', and 'Zip Code' (all strings)
  """

  # remove any duplicate ("Year Month", "buisness tag") pairs. cuts WA State from 24715 -> 19240 rows
  buisness_data = buisness_data.groupby(["business tag", "Year Month"], as_index=False).first()

  buisness_data["Zip Code"] = buisness_data["ZipCode"].str.split('-').str[0]
  buisness_data = cmu.create_address(buisness_data)

  # Get all unique addresses
  address_uniques = pd.unique(buisness_data.loc[:,"locator_address"])
  address_coords = pd.DataFrame(address_uniques, columns=["locator_address"])

  # choose to restore cached coordinates or recreate data
  if isinstance(cached_address_coords, pd.DataFrame):
    address_coords = cached_address_coords
    # address_coords = address_coords.merge(cached_address_coords, on='locator_address', how='left')
    # print('Successful Merge with lookup table')
  else:
    # call geopy api to get coordinates
    address_info = buisness_data.loc[:,["locator_address", "Street Address", "Zip Code", "State"]]
    print(f"Fetching {address_coords.shape[0]} unique addresses...")
    start_time = time.time()
    address_coords["coord"] = cmu.get_coord(address_coords, address_info, fallback=fallback, max_retries=1, error_wait_seconds=3.0, verbose=True)
    print("done\nExecuted in", (time.time() - start_time)//60, "minutes")

  # add manually inputed coordinates
  address_coords = cmu.add_missing_coords(address_coords, MISSING_DATA)

  # determine which addresses the api failed to find coordinates for
  failed_addresses = pd.isnull(address_coords["coord"])
  print("addresses without coordinates:", sum(failed_addresses), "/", address_coords.shape[0])

  # left join dispernsery data with a table matching address to coordinate
  matched_coords = pd.merge(buisness_data, address_coords, how="left", on="locator_address")

  return matched_coords


def main():
  """
  3 competition measures:
    - Compare competition of WA state data to WA state data
    - Compare competition of WA state data to weedmaps illegal
  """

  # get WA state data
  racial_wa_join = pd.read_csv('Processed_Data/washington_with_tagged.csv')
  racial_wa_join["business tag"] = racial_wa_join["License Number"]
  time_racial_wa_join = pd.to_datetime(racial_wa_join["Reporting Period"], infer_datetime_format=True)
  racial_wa_join["Year Month"] = time_racial_wa_join.dt.strftime('%Y-%m')
  racial_wa_join["Year"] = time_racial_wa_join.dt.year
  racial_wa_join["Month"] = time_racial_wa_join.dt.month
  racial_wa_join_address_coords = None
  if USE_CACHED_COORDS and os.path.isfile('Lookup_Tables/wa_state_address_coord_map.csv'):
    # attempt to use cached coordinate data
    racial_wa_join_address_coords = pd.read_csv('Lookup_Tables/wa_state_address_coord_map.csv')
  racial_wa_join = preprocess_routine(racial_wa_join, fallback=False, cached_address_coords=racial_wa_join_address_coords)

  illegal_wa_column_mapper = {
    "address": "Street Address",
    "city": "City",
    "zipcode": "ZipCode",
  }
  # get medical dispensary data
  wa_illegal = pd.read_csv("Raw_Data/112115_62319_WAall_illegal_fixed.csv")
  wa_illegal= wa_illegal.rename(columns=illegal_wa_column_mapper)
  wa_illegal["State"] = "WA"
  wa_illegal["business tag"] = wa_illegal["wmsite"].str.split('/').str[-1]
  wa_illegal["Year Month"] = pd.to_datetime(wa_illegal["dateaccess"], infer_datetime_format=True).dt.strftime('%Y-%m')
  wa_illegal_address_coords = None
  if USE_CACHED_COORDS and os.path.isfile('Lookup_Tables/wa_illegal_coord_map.csv'):
    # attempt to use cached coodinate data
    wa_illegal_address_coords = pd.read_csv('Lookup_Tables/wa_illegal_coord_map.csv')
  wa_illegal = preprocess_routine(wa_illegal, fallback=True, cached_address_coords=wa_illegal_address_coords)


  # LD: WA state data to WA state data
  LD_state_to_state = cmu.get_LD(racial_wa_join, racial_wa_join, "business tag", time_period_col="Year Month", max_dist=WA_DIAMETER, print_percents=10e-2)
  racial_wa_join["LD_state_to_state"], racial_wa_join["min_LD_state_to_state"], racial_wa_join["min_3_LD_state_to_state"], racial_wa_join["LD_state_to_state_count"] = LD_state_to_state[0], LD_state_to_state[1], LD_state_to_state[2], LD_state_to_state[3]
  # new_LD_state_to_state = cmu.get_new_LD(racial_wa_join, racial_wa_join, "business tag", time_period_col="Year Month", max_dist=WA_DIAMETER, print_percents=10e-2)
  # racial_wa_join["new_LD_state_to_state"], racial_wa_join["new_LD_state_to_state_count"] = new_LD_state_to_state[0], new_LD_state_to_state[1]
  # LD: WA state data to weedmaps illegal
  LD_state_to_illegal = cmu.get_LD(racial_wa_join, wa_illegal, "business tag", time_period_col="Year Month", max_dist=WA_DIAMETER, print_percents=10e-2)
  racial_wa_join["LD_state_to_illegal"], racial_wa_join["min_LD_state_to_illegal"], racial_wa_join["min_3_LD_state_to_illegal"], racial_wa_join["LD_state_to_illegal_count"] = LD_state_to_illegal[0], LD_state_to_illegal[1], LD_state_to_illegal[2], LD_state_to_illegal[3]
  # new_LD_state_to_illegal = cmu.get_new_LD(racial_wa_join, wa_illegal, "business tag", time_period_col="Year Month", max_dist=WA_DIAMETER, print_percents=10e-2)
  # racial_wa_join["new_LD_state_to_illegal"], racial_wa_join["new_LD_state_to_illegal_count"] = new_LD_state_to_illegal[0], new_LD_state_to_illegal[1]
  racial_wa_join.to_csv('Processed_Data/pipeline_final_output.csv', index=False)

if __name__ == "__main__":
    main()
