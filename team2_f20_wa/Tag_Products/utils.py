import pandas as pd
import numpy as np

import utils_temporal

"""
column_names_qualitative: qualitative product data aggrgated via random selection  
column_names_quantitative: quantitative product data aggrgated via averaging
"""
column_names_qualitative = [
  'delivery',
  # 'strain',
  # 'strain_description',
]

column_names_quantitative = [
  'gram',
  'eighth',
  'cannabinoid',
  'genetics',
  'intox',
  'wellness',
  'commod',
  'smellflavor',
  'med_under',
  'med_well',
  'med_well_new',
  'post_med_well_new',
  'pre_hybrid',
  'post_hybrid',
  'word_count',
  'is_dab',
]

def get_med_well(df):
    """
    gets med_well, indicating a medical/wellness product
    """
    undersampled = df['med_under'].astype(int).fillna(0)
    wellness = df['wellness'].astype(int).fillna(0)
    return undersampled | wellness


def get_word_count(df):
    """
    gets word_count, total word count of "strain" and "description" fields
    """
    word_count_strain = df['strain'].fillna('').str.split().str.len().astype(int)
    word_count_desc = df['description'].fillna('').str.split().str.len().astype(int)
    return word_count_strain + word_count_desc


def get_dab(df):
    """
    gets is_dab, indicating a product type of "Concentrate", or "Wax"
    """
    product_type_lower = df['producttype'].str.lower()
    return ((product_type_lower == "concentrate") | (product_type_lower == "wax")).astype(int)


def get_tagged_products(df, format_column_s="{}", add_temporal=False):
    """
    Aggregates Qualitative (selects random) & Quantitative (takes average) data for each dispensary's product offering.
    Uses `column_names_quantitative` and `column_names_qualitative` to select columns for aggregation.
      Each quantitative column will be renamed as `avg_<name>`.
      Qualitative column names are kept as is.
    format_column_s: optional string in format "<text>{<column_name>}<text>" to rename all columns according to format
    """
    # create slugs &`monthly date` (YYYY-MM)
    df = df.copy().rename(columns=lambda s: ''.join(s.lower().split()))
    df['slug'] = df['wmsite'].str.split('/').str[-1]
    dates = pd.to_datetime(df['dateaccess'], infer_datetime_format=True)
    df['monthly date'] = dates.dt.strftime('%Y-%m')
    df = df.set_index(['slug', 'monthly date'])

    # add extra quantitative features
    df['med_well'] = get_med_well(df)
    df['word_count'] = get_word_count(df)
    df['is_dab'] = get_dab(df)

    # aggregate each (dispenary, monthly date)'s quantitative & qualitative data
    df_qualitative = df[column_names_qualitative].groupby(['slug', 'monthly date']).first()
    df_quantitative = df[column_names_quantitative].groupby(['slug', 'monthly date']).mean()
    df_counts = df[column_names_quantitative[0]].groupby(['slug', 'monthly date']).agg(["size"]).rename(
      columns=lambda _: "productCNT")

    tagged = pd.concat([df_qualitative, df_quantitative, df_counts], axis=1)
    tagged = tagged.rename(columns=format_column_s.format)

    if add_temporal:
        # add period col
        tagged_with_period = tagged.reset_index().copy()
        unique_dates = tagged_with_period["monthly date"].unique()
        sorted_scrape_dates = np.sort(unique_dates)
        date_scrape_map = {date: i for i, date in enumerate(sorted_scrape_dates)}
        date_scrape_map = pd.DataFrame.from_dict(date_scrape_map, orient='index').rename(columns=lambda _: "period")
        tagged_with_period = tagged_with_period.merge(date_scrape_map, how="left", left_on="monthly date", right_index=True)

        # add offset data
        offsets = [-1, -2]
        data_suffices = ["_6M", "_1Y", "_beginning"]
        lagged_data = utils_temporal.getLaggedCols(tagged_with_period, tagged.columns, offsets, data_suffices,
                                                   id_col="slug", date_col="period", create_beginning=True)

        # perform offset data calculations
        out_suffices = ["_6Mo6M", "_YoY", "_deltaBeginning"]
        tagged = utils_temporal.addLaggedDelta(lagged_data, tagged.columns, data_suffices, out_suffices, id_col="slug",
                                               use_percents=False)
        tagged = tagged.set_index(['slug', 'monthly date'])

    return tagged


