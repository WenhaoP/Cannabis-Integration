####################################################
## Authors: Xuting Liu, Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Date: Spring 2021
####################################################

"""
Original file is located at
    https://colab.research.google.com/drive/11kOBy-axbJ7gz1EzYb7quYufvgiQNJ5N

Step 1:
Format geographic information to match “County” in main data set

Take a look at both fields and you’ll see what you need to do (change case, remove word county, etc). 

Step 2:
Take key and merge in each field to main data set based on County and Year.

Note that there is a one year lag and that the year for each demographics data set is in. See key for a guide. 

Don’t merge in the Old Field name, use the New Field Name that I created instead. 

Step 3:
 Update Readme with Process and also Input Files and New Field Name and Descriptor info from Key.
"""

import pandas as pd
from os import path

file_loc = 'Raw_Data/Area Variables/'

key = pd.read_excel(file_loc + 'Area Variables Key v6.xlsx', engine='openpyxl')

# Extract all the county names from this file. The counties in other files are the same as the counties in this file.
WA_counties = pd.read_csv(file_loc + 'ACSDP5Y2013.DP05_data_with_overlays_2021-03-24T005758.csv')

def clean_df(df):
    """
    Clean the NAME field of the given dataframe, which should have the format '<county>, Washington'
    """
    cleaned = df.copy()

    # Drop the first and last row because the first row are column names and the last row is "total".
    cleaned = df.drop(cleaned.index[0])
    cleaned = df.drop(cleaned.index[-1])

    # Format the county column: get rid of "County , Washington" and to upper case.
    cleaned['County'] = cleaned['NAME'].str.split(' ').str[0].str.upper()

    return cleaned


def process_year(year):
    """
    The key file(`key`) tells us what we should look for and where we can find it.
    For the given year, find the corresponding links to files as listed in the key file.
    After we know the file, find the value in the column indicated by the key file.
    """
    links = key[str(year) + ' link']
    old_field = key['Old Field']
    new_field = key['New Field Name']
    cleaned = clean_df(WA_counties)
    counties = cleaned['County']
    result = pd.DataFrame({'County': counties})
    for i, link in enumerate(links):
        new_name = new_field[i]
        field = old_field[i]

        if not pd.isna(link):
            link = link.strip()
            year_path = file_loc + link
            df = pd.read_csv(year_path)
            df = clean_df(df)
            field_df = df.loc[:, ['County', field]]
            result = result.merge(field_df, left_on='County', right_on='County', how='outer')
            result[field] = pd.to_numeric(result[field], errors='coerce')
            result = result.rename(columns={field: new_name})

    result['Year'] = int(year)
    result = result.drop(result.index[0])  # drop row of descriptions
    print(f'Area variables created for {year}')
    return result


def land_area():
    """
    Clean and return Land Area.csv.
    """
    area = pd.read_csv(file_loc + 'Land Area.csv')
    area['County'] = area['County'].str.strip(',').str.upper()
    return area


def main():
    years_to_process = [2014, 2015, 2016, 2017, 2018, 2019]
    processed_years = [process_year(year) for year in years_to_process]
    area = land_area()

    """
    1. Combine all the dataframes above to area_variables.csv, save in Processed_Data
    2. Read in washington_dataset_no_duplicates.csv, add a year field based on reporting period.
    3. merge in all the dataframes above, update washington_dataset_no_duplicates.csv.
    """

    area_variables = pd.concat(processed_years, ignore_index=True)
    area_variables_with_area = area_variables.merge(area, on='County')
    print('Land Area variables merged')
    area_variables_with_area.to_csv('Processed_Data/area_variables.csv', index=False)
    print('Downloaded concatenated area_variables.csv as a reference')

    washington = pd.read_csv('Processed_Data/washington_dataset_no_duplicates.csv')

    # Add year filed
    years = washington['Reporting Period'].str.split('-').str[0]
    washington['Year'] = years.astype(int)

    # Merge
    washington = washington.merge(area_variables_with_area, on=['County', 'Year'], how='left')
    washington.to_csv('Processed_Data/washington_dataset_with_area.csv', index=False)
    print('Downloaded washington_dataset_with_area.csv')


if __name__ == '__main__':
    main()

