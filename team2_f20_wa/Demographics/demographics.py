import pandas as pd
import numpy as np

import utils as du


def preprocess_license_aliases(df):
  """
  Remove null values, convert licenses to ints, and remove duplicate licenses aliases
  Set the true "License #" as the index
  """
  df = df.loc[~pd.isnull(df["License #"]), :]
  df = df.astype(int).groupby("License #").first()
  return df


def add_demographic_data(wmap_final_join, ownership_data, ethnicity_licenses_aliases=None, verbose=True):
  """
  Takes in `wmap_final_join`, and adds demographic ownership stats based off `ownership_data`.

  `ethnicity_licenses_aliases` is optional dataset that specifies additional license numbers for `ownership_data` licenses to alias as
  """

  # get a map of License Number to demographic data stats
  if verbose: print("merging demographic data...")
  ethnicity_data = du.get_ethnicity_and_gender(ownership_data, verbose=verbose, get_bad_ownership=False, normalize_ownership=True)

  if isinstance(ethnicity_licenses_aliases, pd.DataFrame):
    ethnicity_licenses_aliases = preprocess_license_aliases(ethnicity_licenses_aliases)
    ethnicity_data_aliases = ethnicity_licenses_aliases.merge(ethnicity_data, how="inner", left_on="Key_For_Missing_Ethnicity", right_index=True).drop(columns="Key_For_Missing_Ethnicity")
    ethnicity_data = pd.concat([ethnicity_data, ethnicity_data_aliases])
    # Conflicting Ownership occurs if a License has multiple aliases with conflicting information
    if verbose: print(f"Licenses with conflicting owners: {ethnicity_data.shape[0] - len(set(ethnicity_data.index))}")

  # merge in the demographic data stats
  wmap_final_join_ethnicity = wmap_final_join.merge(ethnicity_data, how="left", left_on="License Number", right_index=True)
  if verbose:
    i_matched = wmap_final_join_ethnicity["License Number"].isin(ethnicity_data.index)
    unmatched_license = wmap_final_join_ethnicity.loc[~i_matched, "License Number"].unique()
    n_matched = i_matched.sum()
    print(f"matched dispensaries: {n_matched} / {wmap_final_join_ethnicity.shape[0]}")

  return wmap_final_join_ethnicity


def main():
  # Join ethnicity and gender data: left join, based on license #

  # get data
  wmap_final_join = pd.read_csv('Processed_Data/wmap_final_join.csv')
  ownership_data = pd.read_csv('Raw_Data/Individuals_Ownership_Demographics.csv')

  # perform logic
  ethnicity_licenses_aliases = pd.read_csv('Lookup_Tables/Fixing_Missing_Ethnicity_Key.csv')
  wmap_final_join = add_demographic_data(wmap_final_join, ownership_data, ethnicity_licenses_aliases=ethnicity_licenses_aliases, verbose=True)

  # save data
  wmap_final_join.to_csv('Processed_Data/washington_with_demographics.csv', index=False)


if __name__ == "__main__":
    main()


