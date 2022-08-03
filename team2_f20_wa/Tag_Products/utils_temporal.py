import pandas as pd
import numpy as np

def getLaggedCols(data, target_cols, offsets, offset_suffices, id_col="License Number", date_col="Reporting Period",
                  create_beginning=True):
    """
    Very generalized version of getting lagged data, offsets can be either generic integers or pd.dateoffset,
      depending on how you format the date_col data.
    Offsets are the time offsets you want from the current date, formatted to match date_col's data type
    Create variables <target_col><offset_suffix>, i.e. <avg_gram><_1y>
    if create_beginning, <target_col>_beginning for all target_cols
    """
    id_dates = data.copy()
    # offset sales dates to be rejoined with it's future date
    offset_data = data[[id_col, *target_cols]].copy()
    offset_data["original_date"] = id_dates[date_col]

    # merge offset sales onto it's own true sales
    make_target_col_map = lambda suffix: {target_col: f"{target_col}{suffix}" for target_col in target_cols}

    for offset, offset_suffix in zip(offsets, offset_suffices):
        offset_date_col = f"date{offset_suffix}"
        offset_data[offset_date_col] = id_dates[date_col] - offset
        id_dates = id_dates.merge(
            offset_data[[id_col, offset_date_col, *target_cols]].rename(columns=make_target_col_map(offset_suffix)),
            how="left", left_on=[id_col, date_col], right_on=[id_col, offset_date_col]).drop(columns=offset_date_col)

    if create_beginning:
        # create a map from id_col -> earliest data
        beginning_data = id_dates[[id_col, date_col]].groupby(id_col, as_index=False).min()
        beginning_data = beginning_data.merge(offset_data[[id_col, "original_date", *target_cols]], how="left",
                                              left_on=[id_col, date_col], right_on=[id_col, "original_date"])
        # merge earliest data in
        id_dates = id_dates.merge(
            beginning_data[[id_col, *target_cols]].rename(columns=make_target_col_map("_beginning")),
            how="left", on=id_col)

    return id_dates.drop(columns=date_col)


def addLaggedDelta(lagged_data, target_cols, data_suffices, out_suffices, id_col="License Number", use_percents=True):
    """
    Given paired new & old data, creates percent changes for data vars
    """

    data = lagged_data.copy()
    for data_suffix, out_suffix in zip(data_suffices, out_suffices):
        for target_col in target_cols:
            new_values, old_values = data[target_col], data[f"{target_col}{data_suffix}"]
            data[f"{target_col}{out_suffix}"] = (new_values - old_values)
            if use_percents:
                data[f"{target_col}{out_suffix}"] /= old_values.replace(0, pd.NA) / 100

    return data
