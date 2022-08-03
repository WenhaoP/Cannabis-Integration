####################################################
## Authors: Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Date: Spring 2021
####################################################

"""
Utilities for creating penalty features
"""

import pandas as pd
import numpy as np
import re
from datetime import timedelta

# Years to include for penalty data (each sheet corresponds to different year in the inout file)
penalty_years = ['2014', '2015', '2016', '2017', '2018', '2019', '2020']

def clean_penalty(penalty_df, year):
    """
    Cleans penalty_df for a specific year

    Inputs:
    - penalty_df: penalty data for specific year
    - year: string corresponding to year that data comes from, useful for handling 2020 formatting edge case

    Outputs:
    - penalty_df_cleaned: cleaned version of penalty_df
    """
    penalty_df_cleaned = penalty_df.copy()

    # predefine columns for formatting consistency across years
    column_names = ['Date', 'Trade Name', 'Violation', 'City', 'UBI', 'License Number', 'Region', 'Case Number', 'Type',
                    'Code', 'Amount Paid', 'Date Paid']

    if year == '2020': # edge case for most recent year
        penalty_df_cleaned["Code"] = np.nan
        penalty_df_cleaned = penalty_df_cleaned.iloc[:, :12]

    penalty_df_cleaned.columns = column_names
    penalty_df_cleaned = penalty_df_cleaned[column_names]
    penalty_df_cleaned['Type'] = penalty_df_cleaned['Type'].str.capitalize()
    penalty_df_cleaned['Amount Paid'] = penalty_df_cleaned['Amount Paid'].astype(str)
    penalty_df_cleaned['Amount Paid'] = penalty_df_cleaned['Amount Paid'].str.replace(r'^discont(.*)', 'discontinued')
    penalty_df_cleaned['Amount Paid'] = penalty_df_cleaned['Amount Paid'].str.replace(r'^Cancel(.*)', 'Cancellation of License')

    # Some data are in format 21xx-mm-dd, instead of 201x-mm-dd
    # Other data are in format 200x-mm-dd, instead of 201x-mm-dd
    penalty_df_cleaned['Date Paid'] = [date - pd.DateOffset(years=100) if date > pd.to_datetime('2100-01-01')
                    else date for date in penalty_df_cleaned['Date Paid']]
    penalty_df_cleaned['Date Paid'] = [date + pd.DateOffset(years=10) if date < pd.to_datetime('2010-01-01')
                    else date for date in penalty_df_cleaned['Date Paid']]

    return penalty_df_cleaned


def concatenate_penalty_data():
    """
    Concatenated all sheets of penalty data

    Output:
    - penalty: concatenated penalty data that will be used for variable creation
    """
    penalty_df_list = []

    for year in penalty_years:
        penalty_df = pd.read_excel('Raw_Data/PRR_21-01-085_MJ.xlsx', sheet_name=year, engine='openpyxl')
        penalty_df_cleaned = clean_penalty(penalty_df, year)
        penalty_df_list.append(penalty_df_cleaned)

    # Create concatenated penalty dataframe
    penalty = pd.concat(penalty_df_list, ignore_index=True)
    return penalty


def penalty_pay_lag(penalty):
    """
    Creates "Pay Lag" variable: difference between date of offense and date paid

    Input:
    - penalty: concatenated penalty data created by concatenate_penalty_data()
    """
    penalty_pay_lag_df = penalty.copy()

    # Create Pay Offense Lag Variable
    penalty_pay_lag_df['Date'] = pd.to_datetime(penalty_pay_lag_df['Date'])
    penalty_pay_lag_df['Date Paid'] = pd.to_datetime(penalty_pay_lag_df['Date Paid'])

    pay_lags = []
    for i in range(len(penalty_pay_lag_df)):
        date = penalty_pay_lag_df['Date'][i]
        pay_date = penalty_pay_lag_df['Date Paid'][i]
        if (not pd.isnull(date)) & (not pd.isnull(pay_date)):
            lag = pay_date - date
        else:
            lag = np.nan
        pay_lags.append(lag)

    penalty_pay_lag_df['Pay_Lag'] = pd.Series(pay_lags)
    return penalty_pay_lag_df


def create_penalty_non_agg_features(penalty):
    """
    Creates the following penalty_level features from the penalty dataset:

    - FinesCnt: Number of times fined that month year based on Type and Date Paid variables
        - Does not include if Amount Paid is “discontinued” or warning. It only counts if there is a dollar amount
          associated with it
    - FinesAmnt: Total amount of fines that month year based on Amount Paid
    - SuspensionDays: Number of Days suspended that month year based on Type and Date Paid variables
        - Base month year based on Date Paid (end date of suspension).
        - So if it’s 30 days on Feb 15, that would be 15 days in Feb and 15 days in Jan
        - any amounts that contains 'days' are suspensions
    - Cancellation: Coded 1 if Date Paid is in that Month Year. Code 0 otherwise
        - Use 'Cancellation of License' in the "Date Paid" column to indicate cancellation
    - Future Cancelled: is cancelled, or is going to be cancelled in the future
    - DiscontinuedCnt: Number of times discontinued that Month Year based on Date of Offense and Amount Paid
        - Use 'discontinued' in the "Date Paid" column to indicate discontinued
    - WarningsCnt: Number of times warned that Month Year based on Date Of Offense and Amount Paid
        - Use 'warnings' in the "Date Paid" column to indicate warning
    - AvgPayLag: Average difference between Date and Date Paid
        - filled NA values with median to be robust to outliers

    Input:
    - penalty: concatenated penalty data across all years

    Output:
    - grouped: penalty data grouped by License Number, Reporting Period (Date Paid), and Trade Name, with added features
               outlined above
    """
    # Dataset that will get joined to throughout variable creation process
    penalty_grouped = pd.DataFrame(penalty.groupby(['License Number', 'Reporting Period', 'Trade Name']).size()).reset_index()
    # penalty_grouped = pd.DataFrame(penalty.groupby(['License Number', 'Reporting Period']).size()).reset_index()
    penalty_grouped = penalty_grouped.drop(columns={0}, axis=1)

    # initalization for count variables
    penalty = penalty.rename(columns={'is_Fine': 'FinesCnt'})
    penalty = penalty.rename(columns={'Amount_Paid_Int': 'FinesAmnt'})
    penalty['Cancellation'] = (penalty['Amount Paid'] == "Cancellation of License").astype(int)
    penalty['DiscontinuedCnt'] = (penalty['Amount Paid'] == "discontinued").astype(int)
    penalty['WarningsCnt'] = (penalty['Amount Paid'] == "warnings").astype(int)

    # format for joining
    penalty['License Number'] = pd.to_numeric(penalty['License Number'], downcast="integer", errors="coerce")
    penalty_grouped['License Number'] = pd.to_numeric(penalty_grouped['License Number'], downcast="integer",
                                                      errors="coerce")

    # compute count variables and join to grouped dataset
    cnt_vars = ["FinesCnt", "FinesAmnt", "Cancellation", "DiscontinuedCnt", "WarningsCnt"]
    for var in cnt_vars:
        # var_df = pd.DataFrame(penalty.groupby(['License Number', 'Reporting Period'])[var].sum()).reset_index()
        var_df = pd.DataFrame(
                    penalty.groupby(['License Number', 'Reporting Period', 'Trade Name'])[var].sum()).reset_index()
        if var == 'Cancellation': # binarize Cancellation
            var_df[var] = (var_df[var] > 0).astype(int)
        penalty_grouped = penalty_grouped.merge(var_df, on=['License Number', 'Reporting Period', 'Trade Name'])
        # penalty_grouped = penalty_grouped.merge(var_df, on=['License Number', 'Reporting Period'])

    # Retailers vs Non-retailers
    check_cancellation(penalty_grouped)

    # AvgPayLag
    penalty_grouped = add_AvgPayLag(penalty, penalty_grouped)

    # SuspensionDays
    penalty_grouped = add_SuspensionDays(penalty, penalty_grouped)

    # Future Cancellation
    # penalty_grouped = add_Future_Cancelled(penalty_grouped)
    return penalty_grouped


def add_AvgPayLag(penalty, penalty_grouped):
    """
    Helper Function to add AvgPayLag feature
    """
    conversion_factor = 1 / (24 * 60 * 60 * 1000000000)  # convert datetime into to days
    median_pay_lag = float(np.nanmedian(penalty["Pay_Lag"].values)) * conversion_factor # for NA replacement

    def avg_pay_lag(series, median_pay_lag):
        """
        Aggregate with median if pay lag at all, otherwise take the mean
        """
        if series.isnull().values.all():
            return median_pay_lag
        else:
            return np.mean(series)

    AvgPayLag_df = pd.DataFrame(penalty.groupby(['License Number', 'Reporting Period', 'Trade Name'],
                                                as_index=False)['Pay_Lag'].agg(avg_pay_lag, median_pay_lag))
    AvgPayLag_df = AvgPayLag_df.reset_index().rename(columns={0: 'AvgPayLag'})
    # AvgPayLag_df = AvgPayLag_df.drop('index', axis=1)

    penalty_grouped = penalty_grouped.merge(AvgPayLag_df, on=['License Number', 'Reporting Period', 'Trade Name'])
    # terminal vs bash yields different column names, not sure why
    penalty_grouped = penalty_grouped.rename(columns={'Pay_Lag': "AvgPayLag"})

    penalty_grouped["AvgPayLag"] = penalty_grouped["AvgPayLag"].fillna(timedelta(days=median_pay_lag))
    return penalty_grouped


# def add_Future_Cancelled(penalty_grouped):
#     """
#     Helper Function to add Future_Cancelled feature
#     """
#     future_cancelled = []
#     for i in range(len(penalty_grouped)):
#         license_number = penalty_grouped['License Number'][i]
#         license_df = penalty_grouped[penalty_grouped['License Number'] == license_number]
#         if np.sum(license_df['Cancellation']) > 0:
#             future_cancelled.append(1)
#         else:
#             future_cancelled.append(0)
#     penalty_grouped['Future_Cancelled'] = future_cancelled
#     return penalty_grouped


def add_SuspensionDays(penalty, penalty_grouped):
    """
    Helper Function to add SuspensionDays feature
    """
    suspension_days = []
    for amount in penalty['Amount Paid']:
        if 'days' in amount:
            suspension_days.append(int(amount.split(" days")[0]))
        else:
            suspension_days.append(0)
    penalty['SuspensionDays'] = suspension_days

    def suspension_backtrack(days_in_given_month, suspension_days):
        """
        Used to create SuspensionDays variable
        - If there are days in the suspension that were observed in previous month
          function returns list where first entry corresponds to days in previous month
          and second entry corresponds to days in the given month
        """
        if days_in_given_month < suspension_days:
            days_in_prev_month = suspension_days - days_in_given_month
            return [days_in_prev_month, days_in_given_month]
        else:
            return suspension_days

    penalty['Day'] = pd.to_datetime(penalty['Date Paid']).dt.day # day of month
    penalty['SuspensionDays_Overlap'] = pd.Series(
        penalty.apply(lambda row: suspension_backtrack(row['Day'], row['SuspensionDays']), axis=1))

    penalty_copy = penalty.copy() # gets modified if we add additional rows corresponding to backtracked suspension days

    for i in range(len(penalty)):
        row = penalty.iloc[i, :]
        overlap = row['SuspensionDays_Overlap']
        if type(overlap) == list:
            # create new row for previous month, update date to be previous month
            ## ultimately other info doesn't matter when it gets grouped
            row_prev = row.copy()
            row_prev['SuspensionDays_Overlap'] = overlap[0]
            row_prev['Date Paid'] = row_prev['Date Paid'] - pd.DateOffset(months=1)
            row_prev['Reporting Period'] = str(row_prev['Date Paid'])

            row['SuspensionDays_Overlap'] = overlap[1]

            penalty_copy.iloc[i, :] = row
            penalty_copy = penalty_copy.append(row_prev, ignore_index=True)
        else:
            continue

    penalty_copy = penalty_copy.sort_values('Date Paid')
    penalty_copy = penalty_copy.drop('SuspensionDays', axis=1)
    penalty_copy = penalty_copy.rename(columns={'SuspensionDays_Overlap': 'SuspensionDays'})

    SuspensionDays_df = pd.DataFrame(penalty_copy.groupby(['License Number', 'Reporting Period', 'Trade Name'])[
                                         'SuspensionDays'].first()).reset_index()
    # left join to get extra month_years that were added
    penalty_grouped = SuspensionDays_df.merge(penalty_grouped, on=['License Number', 'Reporting Period', 'Trade Name'],
                                              how='left')

    return penalty_grouped


def penalty_helper_features(penalty_df):
    """
    Creates features within the penalty dataset that are not explicitly included in the final washington dataset,
    but that are useful to create the features created by create_penalty_non_agg_features()

    Input:
    - penalty_df: concatenated penalty dataframe

    Output:
    - penalty_helper_featurized: penalty dataframe with additional columns containing "helper" features
    """

    penalty_helper_featurized = penalty_df.copy()

    # One-hot for penalties that are fines
    penalty_helper_featurized['Fine'] = (penalty_helper_featurized['Type'] == 'Fine').astype(int)

    # Integer value from Amount Paid - 0 if some other type of penalty
    penalty_helper_featurized['Amount_Paid_Int'] = [int(amount) if re.match("^[0-9]+$", amount) else 0
                                  for amount in penalty_helper_featurized['Amount Paid']]

    # One-hot for penalties that are fines and have amount greater than 0
    penalty_helper_featurized['is_Fine'] = ((penalty_helper_featurized['Fine'] == 1) &
                                           (penalty_helper_featurized['Amount_Paid_Int'] > 0)).astype(int)

    # Reporting Period Variable for grouping with Sales data
    penalty_helper_featurized['Reporting Period'] = penalty_helper_featurized['Date Paid'].astype(str)

    return penalty_helper_featurized


def create_penalty_agg_features(penalty_grouped, sales):
    """
    Creates the following aggregate features from the penalty dataset:

    - FinesCntCum: Total number of fines up to and including that month year
        - only paid fines
    - FinesAmntCum: Total amount of fines up to and including that month year
        - only paid fines
    - SuspensionDaysCum: Total number of days suspended including that month year
    - DiscontinuedCntCum: Same logic as above and other cumulative variables
    - WarningsCntCum: same logic as above and other cumulative variables

    Inputs:
    - penalty_grouped: concatenated penalty data across all years, grouped by License Number, Data, and Name
    - sales: working dataframe that includes features/information from all other data sources

    Output:
    - grouped: penalty data grouped by License Number, Reporting Period (Date Paid), and Trade Name, with added features
               outlined above
    - WarningsCntCum_df: dataframe with all cumulative variables calculated (extra features)
    """
    grouped = penalty_grouped.copy()

    ## Pre-featurize step
    # Reduce to just month, year for date identifier
    grouped['Reporting Period'] = pd.to_datetime(grouped['Reporting Period'])
    sales['Reporting Period'] = pd.to_datetime(sales['Reporting Period'])

    grouped['Reporting Period'] = grouped['Reporting Period'].dt.to_period('M')
    sales['Reporting Period'] = sales['Reporting Period'].dt.to_period('M')

    grouped['License Number'] = grouped['License Number'].astype(str)
    sales['License Number'] = sales['License Number'].astype(str)

    # format for joining
    grouped['License Number'] = pd.to_numeric(grouped['License Number'], downcast="integer", errors="coerce")
    sales['License Number'] = pd.to_numeric(sales['License Number'], downcast="integer", errors="coerce")

    grouped = grouped.rename(columns={"Trade Name": "Tradename"})
    # Outer join to mantain all penalty data
    ## Remove penalties with no associated sales data at the end
    # outer = sales.merge(grouped, on=['Reporting Period', 'License Number'], how='left')
    outer = sales.merge(grouped, on=['Reporting Period', 'License Number', 'Tradename'], how='outer')
    outer = outer.sort_values('Reporting Period')
    outer = outer.reset_index().drop('index', axis=1)

    # filling NA count variables with 0
    for column in grouped.columns.values:
        if column not in ['Reporting Period', 'License Number', 'Tradename', 'index']:
            outer[column] = outer[column].fillna(0)

    # Variable Creation
    cum_vars = ["FinesCnt", "FinesAmnt", "SuspensionDays", "DiscontinuedCnt", "WarningsCnt"]
    cumulative_df = outer.copy()

    for var in cum_vars:
        outer[var] = outer[var].fillna(0)
        cumulative_df[var] = cumulative_df[var].fillna(0)
        cum_var_df = pd.DataFrame(outer.groupby(['License Number'])[var].cumsum())
        cum_var_df = cum_var_df.rename(columns={var: var + "Cum"})
        cumulative_df = cumulative_df.join(cum_var_df)

    # Future Cancellation
    cumulative_df = add_Future_Cancelled(cumulative_df)

    return cumulative_df


def add_Future_Cancelled(cumulative_df):
    """
    Helper Function to add Future_Cancelled feature
    """
    future_cancelled = []
    for i in range(len(cumulative_df)):
        license_number = cumulative_df['License Number'][i]
        license_df = cumulative_df[cumulative_df['License Number'] == license_number]
        if np.sum(license_df['Cancellation']) > 0:
            future_cancelled.append(1)
        else:
            future_cancelled.append(0)
    cumulative_df['Future_Cancelled'] = future_cancelled
    return cumulative_df


def clean_all_wash_data(cumulative_df):
    """
    Takes in cumulative_df with all penalty variables - removes Null sales and duplicate entries
    """
    washington_final = cumulative_df[~cumulative_df['Total Sales'].isna()]

    # Use ["License Number", "Reporting Period", "Total Sales", "Excise Tax Due"] as unique ID
    # to account for any last duplicate sales entries
    washington_final = washington_final.sort_values('DateIssued').drop_duplicates(
        ["License Number", "Reporting Period", "Total Sales", "Excise Tax Due"],
        keep='last').reset_index().drop('index', axis=1)

    # Fix NA fields
    na_cols = ['Total Sales', "Excise Tax Due", "Source", "Med Privilege Code", "Email"]
    for col in na_cols:
        if col in ['Total Sales', "Excise Tax Due", "Med Privilege Code"]: # numeric vs string types
            washington_final[col] = washington_final[col].replace({0: np.nan})
        else:
            washington_final[col] = washington_final[col].replace({'0': np.nan})
    return washington_final


def check_cancellation(penalty_grouped):
    """
    Compute number of cancellations that were Retailers vs Non-retailers
    """
    penalty_grouped_copy = penalty_grouped.copy() # create copy to maintain formatting

    print('\nChecking Cancellations from the penalty data...')
    applicants = pd.read_csv('Processed_Data/applicants_cleaned.csv')

    # # String type for join
    penalty_grouped_copy['License Number'] = pd.to_numeric(penalty_grouped_copy['License Number'], downcast="integer",
                                                           errors="coerce")
    applicants['License Number'] = pd.to_numeric(applicants['License Number'], downcast="integer", errors="coerce")

    joined = applicants.merge(penalty_grouped_copy, on='License Number', how='right') # maintain all records from raw penalty data
    cancelled_sub_df = joined[joined['Cancellation'] > 0]
    all_cancelled_count = len(cancelled_sub_df)
    print(f'All Cancelled Count = {all_cancelled_count}')
    retailer_count = sum(cancelled_sub_df['PrivDesc'] == 'MARIJUANA RETAILER')
    print(f'{retailer_count} Retail Cancellations')
    non_retailer_count = sum(cancelled_sub_df['PrivDesc'] != 'MARIJUANA RETAILER')
    print(f'{non_retailer_count} Non-Retail Cancellations\n')
