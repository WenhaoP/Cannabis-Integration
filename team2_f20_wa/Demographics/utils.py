import pandas as pd
import numpy as np


# categories for mapping "Gender"
MAP_GENDERS = {
  'F': 'Female',
  'M': 'Male',
  'Female': 'Female',
  'Male': 'Male',
}

# categories for mapping "Census Category"
MAP_ETHNICITIES = {
  'American Indian or Alaska Native': 'American_Indian',
  'Black or African American': 'Black',
  'Hispanic/Latina/Latino': 'Hispanic',
  'Native Hawaiian or Pacific Islander': 'Hawaiian_Pac',
  'Asian': 'Asian',
  'White': 'White',
  'Multiracial': 'Multiracial',
}

def normalize_ownerships(df_ownership, normalize_zero_ownership=False):
  """
  Normalizes individuals' ownerships by ensuring all dispensaries have ownerships that add up to 100%
    normalize_zero_ownership: whether to force 0% total ownership -> even ownership among all
  """
  ownership_percents = df_ownership.loc[:,"% Ownership"]
  unnormalized_ownership_totals = df_ownership[["License #", "% Ownership"]].groupby("License #").sum()

  # if any dispensary ownership totals to 0, evenly distribute ownership among all owners
  zero_ownership_licenses = unnormalized_ownership_totals.loc[unnormalized_ownership_totals["% Ownership"] == 0, :].index
  if normalize_zero_ownership and len(zero_ownership_licenses):
    zero_ownership_individuals = df_ownership["License #"].isin(zero_ownership_licenses).astype(int)
    ownership_percents += zero_ownership_individuals
    unnormalized_ownership_totals["% Ownership"] += pd.Series(zero_ownership_individuals.values, index=df_ownership["License #"]).groupby("License #").sum()
  
  df_ownership_totals = df_ownership[["License #"]].merge(unnormalized_ownership_totals, how="left", left_on="License #", right_index=True)["% Ownership"]
  return ownership_percents / (df_ownership_totals / 100)


def get_ethnicity_and_gender(df, normalize_ownership=True, verbose=False, get_bad_ownership=False):
  """
  Takes data on dispensary leadership, and calculates ownership statistics for ethnicity and gender on a per-dispensary basis

  Dispensaries must be uniquely identified by column "License #"
  Individuals must each have: 
    "% Ownership": string, will handle "%" suffixes
    "Census Category": string, options and respective output listed in MAP_ETHNICITIES
    "Gender": string, options and respective output listed in MAP_GENDERS

  Output columns will take on MAP_GENDERS and MAP_ETHNICITIES's categories in the format <category>_Ownership_Perc.
  Categories not included are listed as "Unknown_<col_name>"

  params:
    df: the dispensary leadership data
    verbose: whether to print status on total dispensary ownership integrity (count dispenaries with total ownership != 100%)
    get_bad_ownership: return the unbalanced ownership dispensaries as well
    normalize_ownership: divide all ownership percents by the total ownership (so totals sum to 100%)

  """
  people_data = df.loc[:,["License #"]]
  people_data['% Ownership'] = df['% Ownership'].str.extract(r'(\d+.?\d*)%?').astype(float)

  # inspect dispensaries that don't have a total ownership of 100%, and potentially normalize
  if normalize_ownership:
    people_data['% Ownership'] = normalize_ownerships(people_data)

  ownership_totals = people_data.groupby("License #").sum().rename(columns={"% Ownership": "Total Ownership"})
  bad_ownership_licenses = ownership_totals.loc[~np.isclose(ownership_totals["Total Ownership"], 100.0, atol=0.1),:]
  if verbose:
    print(f"dispensaries with ownerships that don't sum to 100%: {bad_ownership_licenses.shape[0]} / {ownership_totals.shape[0]}")

  # factorize gender & ethnicities according to US Census standards
  people_data['Gender_Cleaned'] = df['Gender'].str.strip().map(MAP_GENDERS).fillna('Unknown_Gender')
  people_data['Race_Cleaned'] = df['Census Category'].str.strip().map(MAP_ETHNICITIES).fillna('Unknown_Race')

  # sum up each dispensary's ("License #") ownership by both metrics
  license_gender_map = pd.pivot_table(
    people_data, index="License #", values="% Ownership", columns="Gender_Cleaned", aggfunc=np.sum, fill_value=0.0
    ).rename("{}_Ownership_Perc".format, axis='columns')
  license_race_map = pd.pivot_table(
    people_data, index="License #", values="% Ownership", columns="Race_Cleaned", aggfunc=np.sum, fill_value=0.0
    ).rename("{}_Ownership_Perc".format, axis='columns')
  
  # create output dataframe
  # person_specific_columns = ['Gender', '% Ownership', 'Self-Identified Race', 'Census Category', 'Percent of Cumulative Majority']
  dispensary_data = df[["License #", "Final", "Ownership"]].rename(
    columns={"Final": "Ownership_Specific", "Ownership": "Ownership_General"}
    ).groupby('License #').first()
  # add license data to output
  dispensary_data = dispensary_data.merge(license_gender_map, how="left", left_index=True, right_index=True)
  dispensary_data = dispensary_data.merge(license_race_map, how="left", left_index=True, right_index=True)

  if get_bad_ownership:
    return dispensary_data, bad_ownership_licenses
  return dispensary_data


  