####################################################
## Authors: Xuting Liu, Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Date: Fall 2020, Spring 2021
####################################################

# -*- coding: utf-8 -*-
"""Linking_WMaps_pipeline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11E7FnxNYCQb2ClyldDsfK7pXVzea4dyA

# Linking WMaps menus to Washington data
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import os
import re


def initial_clean(wmap_rec_raw, washington_raw, applicants_raw):
    """
    Initial cleaning: clean the name, street address, phone, email, and date fields of wmap_rec and washington
    - calls initial_clean_xxx() functions below

    - Only clean addresses for applicants table because this is reference panel, and not used in the final dataset
    """
    wmap_rec = wmap_rec_raw.copy()
    washington = washington_raw.copy()
    applicants = applicants_raw.copy()

    # Clean names
    wmap_rec['dispensaryname'] = initial_clean_name(wmap_rec["dispensaryname"])
    washington["Tradename"] = initial_clean_name(washington["Tradename"])

    # Clean Addresses
    # wmap_rec['street'] = initial_clean_street(wmap_rec['street'])
    # washington["Street Address"] = initial_clean_street(washington["Street Address"])
    # applicants['Street Address'] = initial_clean_street(applicants["Street Address"])

    # Clean phones
    washington["DayPhone"] = initial_clean_phone(washington["DayPhone"], "WA")

    # Clean emails
    wmap_rec['email'] = initial_clean_email(wmap_rec["email"])
    washington["Email"] = initial_clean_email(washington["Email"])

    # Clean date (YYYY-MM) and create slugs
    wmap_rec['slug'] = wmap_rec['wmsite'].str.split('/').str[-1]
    wmaps_dates = pd.to_datetime(wmap_rec['dateaccess'], infer_datetime_format=True)
    wmap_rec['dateaccess'] = wmaps_dates.dt.strftime('%Y-%m')
    wmap_rec = wmap_rec.rename(columns={'dateaccess': 'monthly date'})

    # change the format of Reporting Period in washington: from Date to string
    wa_dates = pd.to_datetime(washington['Reporting Period'], infer_datetime_format=True)
    washington['Reporting Period'] = wa_dates.dt.strftime('%Y-%m')

    return wmap_rec, washington, applicants


def initial_clean_name(names):
    """
    Input: names - series with raw Dispensary Names
    Output: names - series with cleaned Dispensary Names
    """
    names = names.apply(lambda x: x.upper())
    clean_names = []
    for name in names:
        name = name.replace("-", " ").replace("(", "").replace(")", "").replace(":", "").replace(".", "").replace("'",
                                                                                                                  "").replace(
            ",", "")
        name = name.replace("RECREATIONAL", "").replace("COMPANY", "CO").replace("&", "AND").replace("   ",
                                                                                                     " ").replace("  ",
                                                                                                                  " ")
        # name = name.replace("STREET", "ST").replace("HIGHWAY", "HWY")
        if "***" in name:
            to_remove = name.index("*")
            name = name[0:to_remove]
        if "GRAND OPENING" in name:
            to_remove = name.index("GRAND")
            name = name[0:to_remove]
        if '21+' in name:
            to_remove = name.index("21+")
            name = name[0:to_remove]
        clean_names.append(name)
    clean_names = [_.strip() for _ in clean_names]
    return clean_names


def initial_clean_street(streets):
    """
    Input: streets - series with raw Dispensary Addresses
    Output: streets - series with cleaned Dispensary Addresses
    """
    clean_streets = streets.copy()
    clean_streets = clean_streets.str.upper()
    clean_streets = clean_streets.str.replace(' +', ' ', regex=True)
    clean_streets = clean_streets.str.strip()

    raw_cleaning_list = ['.', ',', '#'] # no regex
    for word in raw_cleaning_list:
        clean_streets = clean_streets.str.replace(word, ' ')
        clean_streets = clean_streets.str.replace(' +', ' ', regex=True)
        clean_streets = clean_streets.str.strip()

    # get rid of apostrophes
    clean_streets = clean_streets.str.replace("\'", '')

    # regex replacement
    address_replacement_dict = {"STREET": "ST", "AVENUE": "AVE", "SUITE": "STE", "HIGHWAY": "HWY", "ROAD": "RD",
                                "ROUTE": "RT", "PLACE": "PL", "NORTH": "N", "SOUTH": 'S', 'EAST': 'E', 'WEST': 'W',
                                'NORTHEAST': 'NE', 'NORTHWEST': 'NW', 'SOUTHEAST': 'SE', 'SOUTHWEST': 'SW',
                                'WASHINGTON': 'WA'}

    for word, replacement in address_replacement_dict.items():
        pattern = '(^|\s)' + word + '($|\s)'
        string_replacement = ' ' + replacement + ' ' # maintain spaces
        clean_streets = clean_streets.str.replace(pattern, string_replacement, regex=True)
        clean_streets = clean_streets.str.replace(' +', ' ', regex=True)
        clean_streets = clean_streets.str.strip()

    return clean_streets


def create_pseudo_address(streets):
    """
    Creates more consistent addresses for proper joining between wmaps and WA data

    Input: streets - series with raw Dispensary Addresses
    Output: streets - series with cleaned Dispensary Addresses
        - reduced to [Street Number, Street Name, Street Designation] for consistent formatting
    """
    # clean_streets = streets.copy()
    clean_streets = initial_clean_street(streets)
    clean_streets = clean_streets.str.replace(' +', ' ', regex=True)
    clean_streets = clean_streets.str.strip()

    # Remove directional designation due to mismatches between datasets
    direction_list = ['N', 'S', 'W', 'E', 'NE', 'NW', 'SE', 'SW'] # double space after replacement

    for word in direction_list:
        pattern = '(^|\s)' + word + '($|\s)'
        clean_streets = clean_streets.str.replace(pattern, ' ')
        clean_streets = clean_streets.str.replace(' +', ' ', regex=True)
        clean_streets = clean_streets.str.strip()

    # Just get [Street Number, Street Name, Street Designation] - Ignore STE, etc
    split = clean_streets.str.split(' ')
    clean_streets = split.map(lambda x: x[:3])
    clean_streets = clean_streets.str.join(" ")
    clean_streets = clean_streets.astype(str)
    clean_streets.str.strip()
    return clean_streets


def initial_clean_phone(phones, code):
    """
    Input:
    - phones - series with raw Dispensary Addresses
    - code: ['WA', 'WM'] corresponding to the dataset that phones come from
    Output: phones - series with cleaned Dispensary Addresses
    """
    clean_phones = []
    if code == "WM":
        for phone in phones:
            phone = phone.replace("-", "").replace("(", "").replace(")", "").replace(" ", "").replace(".", "").replace(
                "+1", "").replace("WEED", "")
            clean_phones.append(phone)
        clean_phones = [_.strip() for _ in clean_phones]
    elif code == "WA":
        for phone in phones:
            phone = str(phone)
            if pd.notna(phone):
                phone = phone[:10]
            clean_phones.append(phone)
    return clean_phones


def initial_clean_email(emails):
    """
    Input: emails - series with raw Dispensary Emails
    Output: emails - series with cleaned Dispensary Emails
    """
    clean_emails = emails.copy()
    clean_emails = clean_emails.astype(str)

    clean_emails = clean_emails.str.strip()
    clean_emails = clean_emails.str.upper()
    clean_emails = ["TBD" if "TBD" in e else e for e in clean_emails]
    return clean_emails


def coverage(lst_a, lst_b, dataset_name):
    """
    Return the percentage of the number of elements in lst_b
    that are also in lst_a.
    """
    same = 0
    for ele in lst_b:
        if ele in lst_a:
            same += 1
    return same / len(lst_b)


def coverage_unique(lst_a, lst_b, dataset_name):
    """
    Return the percentage of the number of UNIQUE elements in lst_b
    that are also in lst_a.
    """
    uniuqe_a = list(set(lst_a))
    uniuqe_b = list(set(lst_b))
    same = 0
    print(f'\nUnique Weedmaps Addresses not in {dataset_name}, and that do not match on MM/YYYY)')
    for ele in uniuqe_b:
        if ele in uniuqe_a:
            same += 1
        else:
            print(ele)
    return same / len(uniuqe_b)


def replace_similar(lst_a, lst_b, r):
    """
    Return a list with all elements from lst_b.
    If an element is similar to one from lst_a
    (similarity is larger than r) and the street number
    is the same, or it is very similar (r > 90),
    replace it with the one from lst_a.

    RECOMMENDED: r = 50
    """
    result = []
    replace = {}
    for ele in lst_b:
        ele = str(ele)
        # If we have alreadt replaced it, use it again.
        # if ele in replace:
        # result.append(replace[ele])
        # continue

        # compute Levenshtein Distance using fuzzywuzzy for every pair.
        matched = False
        for comp in lst_a:
            comp = str(comp)
            ratio = fuzz.ratio(ele, comp)
            if (ratio > r and same_street_num(ele, comp)) or ratio > 90:
                replace[ele] = comp
                result.append(comp)
                matched = True
                break

        # No match. Add the original ele to result
        if not matched:
            result.append(ele)
    replace_dict = pd.DataFrame(list(zip(replace.keys(), replace.values())),
                                columns=['original', 'replaced by'])
    return result, replace_dict


def replace_with_best(lst_a, lst_b, output_dict_dir):
    """
    Return a list. For all elements in lst_b, find the most similar element in lst_a
    and replace it with the element in lst_a.

    THIS IS THE METHOD BEING USED.

    Outputs a look-up table for replacing weedmaps addresses
    - output_dict_dir: directory for table
    """
    result = []
    replace = {}
    for ele in lst_b:
        ele = str(ele)
        # If we have already replaced it, use it again.
        # if ele in replace:
        # result.append(replace[ele])
        # continue

        best_r = 0
        best_match = ele
        for comp in lst_a:
            comp = str(comp)
            ratio = fuzz.ratio(ele, comp)
            if (ratio > best_r and same_street_num(ele, comp)):
                best_r = ratio
                best_match = comp

        result.append(best_match)
        replace[ele] = best_match

    replace_dict = pd.DataFrame(list(zip(replace.keys(), replace.values())),
                                columns=['original', 'replaced by'])
    replace_dict.to_csv(output_dict_dir, index=False)
    return result, replace_dict


def update_wmaps_addresses(wmap_series, join_series, table):
    """
    Function for Address Matching
    1) If the wmaps address is already in the WA dataset, do nothing
    2) If wmaps is not in the WA dataset, check if it's in the look-up table
        - look-up table generated by replace_with_best()
    3) Otherwise, do nothing
        - may get corrected for with pseudo address later
    """
    updated_addresses = []
    for ele in wmap_series:
        if ele in join_series:
            continue
        elif ele in list(table['original']):
            row = table.loc[table['original'] == ele]
            updated_addresses.append(row.iloc[0, -1])
        else:
            updated_addresses.append(ele)
    updated_addresses = pd.Series(updated_addresses)
    return updated_addresses


def do_address_update(wmap_rec, washington, applicants):
    """
    Does all address updates
    - calls update_wmaps_addresses() and replace_with_best()
    """
    wmap_updated = wmap_rec.copy()

    # Use Look-Up table if it exists
    wa_sales_lookup_filepath = 'Lookup_Tables/replace_dict_sales.csv'
    if os.path.isfile(wa_sales_lookup_filepath):
        wmap_replace_sales_dict = pd.read_csv('Lookup_Tables/replace_dict_sales.csv')
        wmap_updated['street_update_sales'] = update_wmaps_addresses(wmap_updated["street"], washington["Street Address"],
                                                                   wmap_replace_sales_dict)
    else:
        wmap_updated['street_update_sales'], wmap_replaced_with_final_join_dict = replace_with_best(wmap_updated["street"],
                                                                                                  washington[
                                                                                                      "Street Address"],
                                                                                                  wa_sales_lookup_filepath)

    wa_applicants_lookup_filepath = 'Lookup_Tables/replace_dict_applicants.csv'
    if os.path.isfile(wa_applicants_lookup_filepath):
        wmap_replace_applicants_dict = pd.read_csv('Lookup_Tables/replace_dict_applicants.csv')
        wmap_updated['street_update_applicants'] = update_wmaps_addresses(wmap_updated["street"],
                                                                        applicants["Street Address"],
                                                                        wmap_replace_applicants_dict)
    else:
        wmap_updated['street_update_applicants'], wmap_replaced_with_final_join_dict = replace_with_best(
                                                                                    wmap_updated["street"],
                                                                                    applicants["Street Address"],
                                                                                    wa_applicants_lookup_filepath)

    return wmap_updated


def same_street_num(address_a, address_b):
    """
    Return True iff address_a and address_b are
    the same until first space.
    """
    num_a = address_a.split(' ')[0]
    num_b = address_b.split(' ')[0]
    return num_a == num_b


def replace_similar_dispensary(lst_a, lst_b, r):
    """
    Return a list with all elements from lst_b.
    If an element is similar to one from lst_a
    (similarity is larger than r),
    replace it with the one from lst_a.

    Use this for dispensary names.

    RECOMMENDED: r = 80
    """
    result = []
    replace = {}
    for ele in lst_b:
        ele = str(ele)
        # If we have already replaced it, use it again.
        if ele in replace:
            result.append(replace[ele])
            continue

        # compute Levenshtein Distance using fuzzywuzzy for every pair.
        matched = False
        for comp in lst_a:
            comp = str(comp)
            ratio = fuzz.ratio(ele, comp)
            if ratio > r or ele in comp or (comp in ele and (comp != "POT SHOP" and comp != "HWY 420")):
                replace[ele] = comp
                result.append(comp)
                matched = True
                break

        # No match. Add the original ele to result
        if not matched:
            result.append(ele)
    replace_dict = pd.DataFrame(list(zip(replace.keys(), replace.values())),
                                columns=['original', 'replaced by'])
    return result, replace_dict


def no_match(lst_a, lst_b):
    """
    Return the list of elements present in lst_b that
    are NOT present in lst_a
    """
    dont_match = []
    for ele in lst_b:
        if ele not in lst_a and ele not in dont_match:
            dont_match.append(ele)
    return dont_match


def get_wa_address(washington, dispensary_name):
    """
    Return adresses of a given dispensary from washington data set
    """
    return washington[washington['Tradename'] == dispensary_name]["Street Address"].unique()


def get_rec_address(wmap_rec, dispensary_name):
    """
    Return adresses of a given dispensary from wmap_rec data set
    """
    return wmap_rec[wmap_rec['dispensaryname'] == dispensary_name]['street'].unique()


def get_wa_name(washington, address):
    """
    Return dispensary name of a given address from washington data set
    """
    return washington[washington['Street Address'] == address]["Tradename"].unique()


def get_rec_name(wmap_rec, address):
    """
    Return dispensary name of a given address from wmap_rec data set
    """
    return wmap_rec[wmap_rec['street'] == address]["dispensaryname"].unique()


def merge_with_washington_and_applicants(wmap_rec, washington, applicants):
    # Create Pseudo Addresses for Join
    wmap_rec['street_applicants_join'] = create_pseudo_address(wmap_rec['street_update_applicants'])
    wmap_rec['street_sales_join'] = create_pseudo_address(wmap_rec['street_update_sales'])
    washington["Street Address Join"] = create_pseudo_address(washington["Street Address"])
    applicants['Street Address Join'] = create_pseudo_address(applicants["Street Address"])

    # Rename duplicate columns
    wmap_rec = wmap_rec.rename(columns = {'ManuallyCleaned': 'Wmaps_ManuallyCleaned',
                                          'Source': 'Wmaps_Source'})

    # merge with applicants and final_join
    # WA (Sales) on left, wmaps on right
    wmap_applicants = applicants.merge(wmap_rec, how='left', right_on='street_applicants_join',
                                               left_on='Street Address Join')
    wmap_final_join = washington.merge(wmap_rec, how='left', right_on=['street_sales_join', 'monthly date'],
                                       left_on=['Street Address Join', 'Reporting Period'])

    # the unmatched part of applicants merge
    ## Updated to transactions on left, wmaps on right
    wmap_applicants_unmatched = applicants.merge(wmap_rec, how='right', right_on="street_applicants_join",
                                                         left_on="Street Address Join", indicator=True)
    wmap_applicants_unmatched = wmap_applicants_unmatched.query('_merge == "right_only"')

    # the unmatched part of final_join merge
    wmap_final_join_unmatched = washington.merge(wmap_rec, how='right',
                                                 right_on=["street_sales_join", "monthly date"],
                                                 left_on=["Street Address Join", "Reporting Period"], indicator=True)
    wmap_final_join_unmatched = wmap_final_join_unmatched.query('_merge == "right_only"')

    return wmap_applicants, wmap_applicants_unmatched, wmap_final_join, wmap_final_join_unmatched, wmap_rec


def main():
    # Loading in Data
    wmap_rec_raw = pd.read_csv('Raw_Data/210405_wmaps_linked.csv')
    washington = pd.read_csv('Processed_Data/washington_dataset_with_area.csv')
    applicants = pd.read_csv('Processed_Data/applicants_cleaned.csv')

    # Clean Datasets
    wmap_rec, washington, applicants = initial_clean(wmap_rec_raw, washington, applicants)

    # Perform weedmaps address update based on look-up
    wmap_updated = do_address_update(wmap_rec, washington, applicants)

    # Reduce Weedmaps to only one entry per [MM/YYYY, Slug]
    # wmap_updated = wmap_updated.sort_values('monthly date', ascending=False)
    wmap_updated = wmap_updated.groupby(['monthly date', 'slug']).first().reset_index()

    # Merge Datasets
    wmap_applicants, wmap_applicants_unmatched, wmap_final_join, wmap_final_join_unmatched, wmap_updated = \
        merge_with_washington_and_applicants(wmap_updated, washington, applicants)

    # Coverage Results - Unique
    unique_coverage_applicants = coverage_unique(wmap_applicants['street_applicants_join'].tolist(),
                                          wmap_updated['street_applicants_join'].tolist(), "Applicants")
    unique_coverage_final_join = coverage_unique(wmap_final_join['street_sales_join'].tolist(),
                                          wmap_updated['street_sales_join'].tolist(), "Washington")
    print("\nMerge completed.\nUnique Coverage when merged with applicants: {}\nUnique Coverage when merged with "
          "washngton_dataset: {}".format(unique_coverage_applicants, unique_coverage_final_join))

    # Coverage Results - Overall
    coverage_applicants = coverage(wmap_applicants['street_applicants_join'].tolist(),
                                                 wmap_updated['street_applicants_join'].tolist(), "Applicants")
    coverage_final_join = coverage(wmap_final_join['street_sales_join'].tolist(),
                                                 wmap_updated['street_sales_join'].tolist(), "Washington")
    print("\nCoverage when merged with applicants: {}\nCoverage when merged with "
          "washngton_dataset: {}".format(coverage_applicants, coverage_final_join))

    # Download merged - remove pseudo columns
    pseudo_columns = ['Street Address Join', 'monthly date', 'street_update_sales', 'street_update_applicants',
                      'street_applicants_join', 'street_sales_join']
    wmap_applicants = wmap_applicants.drop(pseudo_columns, axis=1)
    wmap_final_join = wmap_final_join.drop(pseudo_columns, axis=1)

    wmap_applicants.to_csv('Processed_Data/wmap_applicants.csv', index=False)
    wmap_final_join.to_csv('Processed_Data/wmap_final_join.csv', index=False)
    wmap_applicants_unmatched.to_csv('Processed_Data/wmap_applicants_unmatched.csv', index=False)
    wmap_final_join_unmatched.to_csv('Processed_Data/wmap_final_join_unmatched.csv', index=False)


if __name__ == "__main__":
    main()