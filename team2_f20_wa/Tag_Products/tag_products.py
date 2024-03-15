import pandas as pd
import numpy as np

import utils as tu

"""
specify subsets of data you want to work:
key: column name format
  column: columns name to compare with
  targets: set of values to match <column name> to
"""
TARGETED_CALCULATIONS = {
    "{}_flwrs": {
        "column": "producttype",
        "targets": ["Indica", "Sativa", "Hybrid"]
    },
    "{}_dabs": {
        "column": "producttype",
        "targets": ["Concentrate", "Wax"]
    },
}


def get_all_tagged_products(df):
    product_infos = tu.get_tagged_products(df, add_temporal=True)
    for s_format, specifics in TARGETED_CALCULATIONS.items():
        column, targets = specifics["column"], specifics["targets"]
        df_reduced = df[df[column].isin(targets)]
        targeted_product_info = tu.get_tagged_products(df_reduced, format_column_s=s_format, add_temporal=True)
        product_infos = pd.concat([product_infos, targeted_product_info], axis=1)
    return product_infos


def preprocess(df):
    """
    drop grams over 150 and eighths over 400 before running any of the avg_eighths, avg_grams
    """
    df = df.rename(columns= lambda s: s.lower())
    n_old = df.shape[0]
    zero_to_nan_cols = ["eighth", "gram", "half", "quarter", "oz"]

    # remove extreme data points
    eighth_tresh = 150
    gram_tresh = 400
    df = df[df["eighth"].replace({np.nan:0}) <= eighth_tresh]
    df = df[df["gram"].replace({np.nan:0}) <= gram_tresh]

    # replace $0 prices to Na
    for col in zero_to_nan_cols:
        df[col] = df[col].replace({0:np.nan})

    # remove "_labeled" suffix from column names
    df = df.rename(columns=lambda s: s.split("_labeled")[0])

    column_name_mapper = {
        'commoditization': 'commod',
        'intoxication': 'intox',
        'medical': 'med',
        'medical_undersampled': 'med_under',
        'medical_wellness': 'med_well_new',
        'post_medical_wellness': 'post_med_well_new'
    }
    df = df.rename(columns=column_name_mapper)

    # drop scrapes 255 & 257
    bad_scrapes = ["255", "257"]
    df = df[~df["scrape"].astype("str").isin(bad_scrapes)]

    n_new = df.shape[0]
    print(f"preprocessing tagged products reduced data from {n_old} to {n_new}")
    return df


def main():
    # Join WA Label data: left join, based on wmsite and monthly date

    # get data
    wmap_final_join = pd.read_csv('Processed_Data/washington_with_demographics.csv')
    wmap_final_join['slug'] = wmap_final_join['wmsite'].str.split('/').str[-1]
    product_data = pd.read_csv('Raw_Data/full_dataset_with_labels.csv')
    product_data = preprocess(product_data)
    # perform logic
    label_1 = get_all_tagged_products(product_data)
    wmap_final_join = wmap_final_join.merge(label_1, how='left', left_on=['slug', 'Reporting Period'],
                                            right_index=True)

    # save data
    wmap_final_join.to_csv('Processed_Data/washington_with_tagged.csv', index=False)

if __name__ == "__main__":
    main()