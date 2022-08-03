####################################################
## Author: Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Date: Fall 2020, Spring 2021
####################################################

"""
Main method imports all WA raw data, cleans datasets, and makes appropriate joins
- granularity is sales per month for a given license number
- contains applicant info (name, addressm zip, etc), medical endorsement info, enforcement info, violation info, and
  penalty info
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
# Supress seaborn warnings
import warnings
warnings.filterwarnings("ignore")

import Penalty.penalty_variables as pv

# Set Boolean Variables for Downloads

## set True to download the cleaned applicants reference (granularity is applicants, NOT sales)
download_applicant_reference = True
## set True to download sales with no applicant info for manual inspection
download_sales_with_no_applicant_info = True
## set True to plot and download png files of endorsements
plot_medical_endorsements_over_time = True


def clean_sales(sales):
    """
    Takes in sales dataframe and cleans it
    - "Old" (pre 2017) Source: 'By-License-Number-MJ-Tax-Obligation-by-Licensee-thru-10_31_17.xlsx'
    - "New" (post 2017) Source: '2020-06-09-MJ-Sales-Activity-by-License-Number-Traceability-Contingency-Reporting-Retail.xlsx'

    Inputs
    - sales: dataframes containing sales information

    Output
    - sales_cleaned: cleaned dataframe
    """
    sales = strip_columns(sales)

    # drop column with NA sales info
    sales_cleaned = sales.dropna(subset=['Total Sales'])
    sales_cleaned = sales_cleaned.reset_index().drop(['index'], axis=1)

    sales_cleaned = sales_cleaned.iloc[:, 0:4]
    sales_cleaned = clean_license_number(sales_cleaned)
    # only extract start of the period
    split_dates = sales_cleaned['Reporting Period'].str.split(' -')
    sales_cleaned['Reporting Period'] = pd.Series([x[0] for x in split_dates])
    # Remove summary and missing info, Convert to datetime period for later comparison
    sales_cleaned = sales_cleaned[sales_cleaned['Reporting Period'] != 'Reporting Period']
    sales_cleaned = sales_cleaned[~sales_cleaned['Reporting Period'].isnull()]
    sales_cleaned['Reporting Period'] = pd.to_datetime(sales_cleaned['Reporting Period'])

    # Cleaning financial data to float values
    for col in ['Total Sales', 'Excise Tax Due']:
        sales_cleaned[col] = sales_cleaned[col].str.strip()
        sales_cleaned[col] = sales_cleaned[col].str.replace('-', '')
        sales_cleaned[col] = sales_cleaned[col].str.replace(',', '')
        sales_cleaned[col] = sales_cleaned[col].str.replace('$', '')
        sales_cleaned[col] = sales_cleaned[col].str.replace('(', '')
        sales_cleaned[col] = sales_cleaned[col].str.replace(')', '')
        sales_cleaned[col] = sales_cleaned[col].str.replace(')', '')
        sales_cleaned[col] = sales_cleaned[col].replace(r'^\s*$', '0', regex=True)
        sales_cleaned[col] = sales_cleaned[col].astype(float)
    return sales_cleaned


def clean_license_number(df):
    """
    Cleans license number column in a particular dataframe, and returns the dataframe with the cleaned column
    - Requires that a column contains a column called 'License Number'
    - any dataframe in WA database (all should have a License Number column, but may be named differently)
    """

    df = df[(df['License Number'] != 'nan') & (~df['License Number'].isnull())]
    df['License Number'] = df['License Number'].astype(str)
    # Clean Applicant type - these are from sales data
    df = df[df["License Number"] != "PRODUCERS PERIOD TOTAL"]
    df = df[df["License Number"] != "PROCESSORS PERIOD TOTAL"]
    df = df[df["License Number"] != "RETAILERS PERIOD TOTAL"]
    df = df.reset_index().drop(['index'], axis=1)

    # Clean formatting: leading and trailing characters, maintain 6 digit format
    df['License Number'] = df['License Number'].str.strip()
    df['License Number'] = df['License Number'].astype(str)
    # df['License Number'] = ['0' + x if len(x) == 5 else x for x in df['License Number']]
    df['License Number'] = [x.split('-')[0] if '-0' in x else x for x in df['License Number']]
    df['License Number'] = [x.split('-')[0] if '-1' in x else x for x in df['License Number']]
    return df


def join_sales_tables(sales_df_list):
    """
    Merge all sales dataframes together (pre 2017 and post 2017 in WA database)
    - return concatenated dataframe to get full sales data
    """
    full_sales = pd.concat(sales_df_list)
    full_sales = full_sales.sort_values('Reporting Period')
    full_sales = full_sales.reset_index().drop(['index'], axis=1)
    full_sales = clean_license_number(full_sales)
    return full_sales


def join_applicant_info(applicant_df_list):
    """
    Take in list of all applicant dataframes, concatenate them, and clean them
    - MarijuanaApplicants.xlsx: post 2017 - all retailers
    - MarijuanaApplicants_2017.xlsx: 2017 and before, 3 dataframes in 3 different sheets
    """
    # Fix column naming for appropriate concat
    for i in range(len(applicant_df_list)):
        if "DateCreated" in applicant_df_list[i].columns:
            cleaned_df = applicant_df_list[i].rename(columns={"DateCreated": "DateIssued"})
            applicant_df_list[i] = cleaned_df
    # Concatenate all dataframes in the list
    applicants = pd.concat(applicant_df_list)
    applicants.reset_index(inplace=True)

    applicants['PrivDesc'] = applicants['PrivDesc'].str.strip()

    # Fix formatting
    applicants = strip_columns(applicants)
    applicants = applicants.drop(['index'], axis=1)
    applicants = applicants.rename(columns={'License #': 'License Number'})
    return applicants


def strip_columns(df):
    """
    Take in dataframe, strip column names, return cleaned dataframe
    - Many column names have "invisible" leading and trailing spaces
    """
    for column in df.columns:
        df = df.rename(columns={column : column.strip()})
    return df


def clean_zipcode(zipcodes):
    """
    Takes in a series of Zipcodes, cleans them, and returns cleaned series
    - put Zipcodes into xxxxx-xxxx format
    """
    zipcodes = zipcodes.astype(str)
    # 9 digit form
    zipcodes = [x[:5] + '-' + x[5:] for x in zipcodes]
    return zipcodes


def clean_applicants(applicants, download):
    """"
    Takes in dataframe with all applicant information (after being concatenated) and cleans it

    Inputs
    - applicants: concatenated applicant information
    - download: Boolean indicating if you want to download the cleaned applicants dataframe as a reference
        - granulairty of applicants is a particular applicant, but granularity of final dataframe is monthly sales
          for a given license number

    Output
    - applicants_cleaned: cleaned applicants dataframe
    """
    applicants_cleaned = applicants.copy()
    applicants_cleaned = applicants_cleaned.dropna(how='all', axis='columns')

    # Fix formatting
    applicants_cleaned = clean_license_number(applicants_cleaned)
    applicants_cleaned = applicants_cleaned.replace({'00000000': np.NaN})
    applicants_cleaned = applicants_cleaned.replace(r'^\s*$', np.NaN, regex=True)
    applicants_cleaned['ZipCode'] = clean_zipcode(applicants_cleaned['ZipCode'])
    applicants_cleaned['DateIssued'] = applicants_cleaned['DateIssued'].replace({'0': np.nan, 0: np.nan})
    applicants_cleaned['DateIssued'] = pd.to_datetime(applicants_cleaned['DateIssued'], format='%Y%m%d')

    # Account for any last duplicates, take most recent
    applicants_cleaned['Tradename'] = applicants_cleaned['Tradename'].str.strip()
    applicants_cleaned = applicants_cleaned.sort_values(['DateIssued'], ascending=False)
    applicants_cleaned = applicants_cleaned.groupby(['Tradename',  'License Number', "Street Address"]).first().reset_index()

    # Cleaning + Formatting
    applicants_cleaned = strip_columns(applicants_cleaned)
    applicants_cleaned = clean_license_number(applicants_cleaned)
    applicants_cleaned['Street Address'] = applicants_cleaned['Street Address'].str.strip()

    if download:
        applicants_cleaned.to_csv('Processed_Data/applicants_cleaned.csv', index=False)
        print('Downloaded applicants_cleaned.csv, which include retailers, processors, and producers')

    # Clean PrivDesc -> should only include "MARIJUANA RETAILER", but include all in applicants reference above
    applicants_cleaned = applicants_cleaned[applicants_cleaned['PrivDesc'] == "MARIJUANA RETAILER"]
    applicants_cleaned = applicants_cleaned.reset_index().drop('index', axis=1)
    print(f"Total Number of Retailers in WA = {len(applicants_cleaned)}")
    return applicants_cleaned


def applicant_sales_join(applicants, sales, download):
    """
    Inputs
    - applicants: concatenated applicant information
    - download: Boolean indicating if you want to download the cleaned applicants dataframe as a reference
        - granularity of applicants is a particular applicant, but granularity of final dataframe is monthly sales
          for a given license number

    Output
    - applicant_sales_joined_cleaned: all sales joined with applicant information
        - drop sales with no applicant information
    """
    # join retailers and sales on "License Number", ensure proper formatting
    applicants['License Number'] = applicants['License Number'].astype(float).astype(int)
    sales['License Number'] = sales['License Number'].astype(float).astype(int)
    applicant_sales_joined = sales.merge(applicants, how='inner', on='License Number')
    applicant_sales_joined['DateIssued'] = pd.to_datetime(applicant_sales_joined['DateIssued'])

    if download:
        no_tradename = applicant_sales_joined[applicant_sales_joined['Tradename'].isna()]
        no_applictant_sales_df = no_tradename[['License Number', 'Reporting Period', 'Total Sales', 'Excise Tax Due']]
        no_applictant_sales_df.to_csv('Processed_Data/sales_with_no_applicant_info.csv', index=False)
        print('downloaded sales_with_no_applicant_info.csv')

    applicant_sales_joined_cleaned = applicant_sales_joined[~applicant_sales_joined['Tradename'].isna()]
    return applicant_sales_joined_cleaned


def clean_endorsements(endorsements):
    """
    Clean endorsements dataframe, return cleaned dataframe
    """
    endorsements_cleaned = endorsements.copy()
    # Only keep fields with relevant/non-redundant info
    endorsements_cleaned = endorsements_cleaned.drop(['UBI', 'City', 'State', 'County', 'Zip Code', 'Day Phone',
                                                      'Termination Code', "Suite Rm", "Status", "Privilege"], axis=1)
    # Clean license number, date created for later comparison
    endorsements_cleaned["Date Created"] = endorsements_cleaned["Date Created"].astype(str)
    endorsements_cleaned["Date Created"] = pd.to_datetime(endorsements_cleaned["Date Created"], format='%Y%m%d')
    endorsements_cleaned = clean_license_number(endorsements_cleaned)
    return endorsements_cleaned


def join_endorsements(applicant_sales_joined, endorsements):
    """
    Merges sales data (with applicant info) with endorsement info for a given applicant in the given month

    Inputs:
    - applicant_sales_joined: current joined dataframe (sales granularity) with applicant info
    - endorsements: cleaned endorsements dataframe

    Output
    - returns the cleaned, joined dataframe
    """
    # Clean and transform for proper join
    applicant_sales_joined["License Number"] = applicant_sales_joined["License Number"].astype(str)
    endorsements['Tradename'] = endorsements['Tradename'].str.strip()
    applicant_sales_joined['Tradename'] = applicant_sales_joined['Tradename'].str.strip()
    endorsements['Street Address'] = endorsements['Street Address'].str.strip()
    applicant_sales_joined['Street Address'] = applicant_sales_joined['Street Address'].str.strip()

    # License Number, Tradename, and Address constitute unique identifier
    endorsements_joined = applicant_sales_joined.merge(endorsements, how = "left",
                                              on=["License Number", "Tradename", "Street Address"])
    endorsements_joined = endorsements_joined.reset_index().drop(['index'], axis=1)

    return endorsements_joined


def add_medical_endorsement(endorsements_joined, plot):
    """
    Takes running joined dataframe and adds endorsement information for applicant, month pair

    Inputs
    - endorsements_joined: running join (see flowchart)
    - plot: Boolean to plot medical endorsements over time
        - code saves plots as .png files
    """

    # Endorsed if date created (endorsement occurence) occurs before reporting period (sales occurence)
    endorsements_joined["Reporting Period"] = pd.to_datetime(endorsements_joined["Reporting Period"])
    endorsements_joined["Date Created"] = pd.to_datetime(endorsements_joined["Date Created"])

    medical_endorsement = []
    for ind in endorsements_joined.index:
      curr_rp = endorsements_joined["Reporting Period"][ind]
      curr_dc = endorsements_joined["Date Created"][ind]
      if pd.notna(curr_dc):
        if curr_rp > curr_dc:
          medical_endorsement.append(1)
        else:
          medical_endorsement.append(0)
      else:
        medical_endorsement.append(0)
    endorsements_joined["Medical Endorsement"] = medical_endorsement
    endorsements_joined['Date Created'] = endorsements_joined['Date Created']

    # outputs two plots
    ## one with proportion of retailers with an endorsement over time
    ## second of count of retailer with an endorsement over time
    if plot:
        # Cleaning for plot formatting
        retailers = endorsements_joined[endorsements_joined["PrivDesc"] == 'MARIJUANA RETAILER']
        retailers['No Endorsement'] = [1 if x == 0 else 0 for x in retailers['Medical Endorsement']]
        retailer_rpg = retailers.groupby('Reporting Period').sum().reset_index()
        denom = retailer_rpg['Medical Endorsement'] + retailer_rpg['No Endorsement']
        retailer_rpg['Medical Endorsement'] = retailer_rpg['Medical Endorsement'] / denom
        retailer_rpg['Reporting Period Just Date'] = retailer_rpg[
            'Reporting Period'].dt.date  # Create New column for plotting

        # Plot running count of retailers with endorsements
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x="Reporting Period Just Date", y="Medical Endorsement", data=retailer_rpg)
        ax = sns.barplot(x="Reporting Period Just Date", y="Medical Endorsement", data=retailer_rpg)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_xticklabels([t.get_text().split(' ')[0] for t in ax.get_xticklabels()])
        plt.title('Proportion of Retailers with a Medical Endorsement over time')
        plt.tight_layout()
        fig.savefig('Plots/Proportion_Medical_Endorsements_Over_Time.png')
        print('Downloaded Proportion_Medical_Endorsements_Over_Time.png')

        # Plot running count of retailers with endorsements
        retailer_rpg['Medical Endorsement'] = retailer_rpg['Medical Endorsement'] * denom
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.barplot(x="Reporting Period Just Date", y="No Endorsement", data=retailer_rpg)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_xticklabels([t.get_text().split(' ')[0] for t in ax.get_xticklabels()])
        plt.title('Count of Retailers without a Medical Endorsement over time')
        plt.tight_layout()
        fig.savefig('Plots/Count_Without_Medical_Endorsements_Over_Time.png')
        print('Downloaded Plots/Count_Without_Medical_Endorsements_Over_Time.png')
    return endorsements_joined


def clean_and_create_enforcements(enforcements):
    """
    Cleaned enforcements table and assigns enforcement to appropriate reporting period
    - returns cleaned enforcements dataframe
    """
    # Define mapping for enforcements - cleaning the date
    activity_dates = enforcements['Date']
    activity_reporting_period = []
    for date in activity_dates:
        year = int(date[6:10])
        month = int(date[0:2])
        # add 1 month
        if month == 12:
            month, year = 1, year + 1
        else:
            month += 1
        # add a 0 in front if needed
        month = str(month)
        if len(month) == 1:
            month = str(0) + month
        day = '01'
        activity_reporting_period.append(str(year) + '-' + month + '-' + day)

    # Format + Clean
    enforcements_cleaned = enforcements.copy()
    enforcements_cleaned['Reporting Period'] = activity_reporting_period
    enforcements_cleaned['Reporting Period'] = pd.to_datetime(enforcements_cleaned['Reporting Period'])
    enforcements_cleaned = enforcements_cleaned.dropna(subset=['License Number']).reset_index().drop('index', axis=1)
    return enforcements_cleaned


def enforcement_abbreviations(enforcements_cleaned):
    """
    Abbreviate Enforcement data for easier interpretation
    - full descriptions stored in 'WA_metadata.xlsx'
    """
    activity_abbr = []
    for activity in enforcements_cleaned['Activity" ']:
        if activity == 'Marijuana Premises Check':
          abbr = 'PC'
        elif activity == 'Marijuana Compliance Check-no Sale':
          abbr = 'CCN'
        elif activity == 'Marijuana Compliance Check-Sale':
          abbr = 'CCS'
        elif activity == 'Marijuana Applicant Site Verification':
          abbr = 'SV'
        elif activity == 'MARIJUANA INVENTORY AUDIT':
          abbr = 'IA'
        elif activity == 'Marijuana Administrative Hold Initiated':
          abbr = 'AH'
        else:
          abbr = 'UNKNOWN'
        activity_abbr.append(abbr)
    enforcements_cleaned['Activity'] = activity_abbr
    enforcements_cleaned = enforcements_cleaned.drop(columns=['Date', 'Activity" ', 'City Name', 'County Name'])
    enforcements_cleaned = enforcements_cleaned.rename(columns={'Activity': 'Enforcement Activity'})

    # One-hot encode Activity
    activity_dummies = pd.get_dummies(enforcements_cleaned['Enforcement Activity'])
    enforcements_cleaned = enforcements_cleaned.join(activity_dummies)
    enforcements_cleaned = enforcements_cleaned.drop(['Enforcement Activity'], axis=1)
    return enforcements_cleaned


def join_enforcements(medical_endorsement_join, enforcements):
    """
    Joins enforcements with existing "master" df (sales granularity)
    - returns joined dataframe
    """
    # Group by License Number only to get the accumulative (across all reporting periods) enforcement counts
    enforcements_acc = enforcements.groupby(['License Number']).sum().reset_index()
    enforcements_acc = enforcements_acc.rename(columns = {'AH': 'acc_AH', 'CCN': 'acc_CCN', 'CCS': 'acc_CCS', 'IA': 'acc_IA', 'PC': 'acc_PC', 'SV': 'acc_SV'})
    enforcements_acc['Total Accumulative Enforcements'] = enforcements_acc['acc_AH'] + enforcements_acc['acc_CCN'] \
                                                           + enforcements_acc['acc_CCS'] + enforcements_acc['acc_IA'] \
                                                           + enforcements_acc['acc_PC'] + enforcements_acc['acc_SV']

    # Group by License Number and Reporting period to get Counts
    enforcements_reporting_grouped = enforcements.groupby(['License Number', 'Reporting Period']).sum().reset_index()
    enforcements_reporting_grouped['Total Enforcements'] = enforcements_reporting_grouped['AH'] + enforcements_reporting_grouped['CCN'] \
                                                        + enforcements_reporting_grouped['CCS'] + enforcements_reporting_grouped['IA'] \
                                                        + enforcements_reporting_grouped['PC'] + enforcements_reporting_grouped['SV']

    # Verify the computation results of two aggregation above
    method_1 = enforcements_acc.sort_values('License Number')['Total Accumulative Enforcements']
    method_2 = enforcements_reporting_grouped.groupby('License Number').sum().sort_index()['Total Enforcements']
    assert method_1.reset_index()['Total Accumulative Enforcements'].equals(method_2.reset_index()['Total Enforcements'])

    # Merge
    enforcements_reporting_grouped = enforcements_reporting_grouped.merge(enforcements_acc)
    enforcements_join = medical_endorsement_join.merge(enforcements_reporting_grouped, how="left",
                                                       on=['License Number', 'Reporting Period'])
    # clean NA in newly joined columns
    fill_na_cols = ['AH', 'CCN', 'CCS', 'IA', 'PC', 'SV', 'Total Enforcements',
                    'acc_AH', 'acc_CCN', 'acc_CCS', 'acc_IA', 'acc_PC', 'acc_SV', 'Total Accumulative Enforcements']
    for col in fill_na_cols:
        enforcements_join[col] = enforcements_join[col].fillna(0)
    return enforcements_join


def clean_violations(violations):
    """
    Cleans violations dataframe and returns cleaned dataframe
    """
    violations_cleaned = violations.copy()
    # Clean City Name for consistent formatting
    violations_cleaned["City Name"] = violations_cleaned["City Name"].str.replace('UNINCORP. AREAS', 'UNINCORPORATED AREAS')
    violations_cleaned["City Name"] = violations_cleaned["City Name"].str.replace(' (CITY)', '')

    # Add one 01 for day to merge with reporting period
    violations_cleaned['Visit Date'] = pd.to_datetime(violations_cleaned['Visit Date'],
                                                      infer_datetime_format=True).dt.to_period('M')
    violations_cleaned['Visit Date'] = violations_cleaned['Visit Date'].astype(str)
    violations_cleaned['Reporting Period'] = [x + '-01' for x in violations_cleaned['Visit Date']]
    violations_cleaned['Visit Date'] = pd.to_datetime(violations_cleaned['Visit Date'])

    # Offset with 1 month forward (reporting period)
    violations_cleaned['Reporting Period'] = pd.to_datetime(violations_cleaned['Reporting Period'])
    violations_cleaned['Reporting Period'] = violations_cleaned['Reporting Period'].dt.date + pd.DateOffset(months=1)

    violations_cleaned = violations_cleaned.drop(['City Name', 'County Name'], axis=1)

    return violations_cleaned


# Merge with final join
def join_violations(enforcements_joined, violations):
    """
    Join running master join (sales granularity) with violations info for each applicant, month pair
    """
    violations['License Number'] = violations['License Number'].astype(str)
    violations_joined = enforcements_joined.merge(violations, how='left', on=['License Number', 'Reporting Period'])

    # fill remaining columns with 0 (sales with no violations during that period
    remaining_columns = violations.columns[~violations.columns.isin(['License Number', 'Reporting Period'])]

    for col in remaining_columns:
      violations_joined[col] = violations_joined[col].fillna(0)

    return violations_joined


def main():
    """
    Imports all WA raw data, cleans datasets, and makes appropriate joins
    - granularity is sales per month for a given license number
    - contains applicant info (name, addressm zip, etc), medical endorsement info, enforcement info, and violation info
    """
    # Import Datasets
    sales_old = pd.read_csv('Raw_Data/By-License-Number-MJ-Tax-Obligation-by-Licensee-thru-10_31_17.csv', header=1)
    ## New Sales dataset
    sales_new = pd.read_csv(
        'Raw_Data/2021-04-06-MJ-Sales-Activity-by-License-Number-Traceability-Contingency-Reporting-Retail.csv',
        header=3)

    applicants_current = pd.read_csv('Raw_Data/MarijuanaApplicants.csv') # Applicants (2017 onward)
    retailers = pd.read_csv('Raw_Data/MarijuanaApplicants_2017.csv') # Retailers (from pre-2017 file)
    producers = pd.read_csv('Raw_Data/MarijuanaApplicants_2017_producers.csv')  # Producers (from pre-2017 file)
    processors = pd.read_csv('Raw_Data/MarijuanaApplicants_2017_processors.csv')  # Processors (from pre-2017 file)
    endorsements = pd.read_csv('Raw_Data/MedicalMarijuanaEndorsements.csv', header=2) # Medical Endorsements
    enforcements = pd.read_csv('Raw_Data/Enforcement_Visits_Dataset.csv') # Enforcements
    violations = pd.read_csv('Raw_Data/Violations_Dataset.csv') # Violations
    penalty = pv.concatenate_penalty_data()

    # Sales Cleaning
    sales_old_cleaned = clean_sales(sales_old)
    sales_new_cleaned = clean_sales(sales_new)

    # Concatenate the sales tables
    sales_df_list = [sales_old_cleaned, sales_new_cleaned]
    full_sales = join_sales_tables(sales_df_list)

    # Concatenate the Applicant Info
    applicant_df_list = [applicants_current, retailers, producers, processors]
    applicants = join_applicant_info(applicant_df_list)

    # Clean the applicant dataframe
    applicants_cleaned = clean_applicants(applicants, download_applicant_reference)

    # Join the Applicants with the Sales
    applicant_sales_joined = applicant_sales_join(applicants_cleaned, full_sales,
                                                  download_sales_with_no_applicant_info)

    # Clean the Endorsement Dataset
    endorsements_cleaned = clean_endorsements(endorsements)

    # Join endorsements onto the applicants and sales dataframe
    endorsements_joined = join_endorsements(applicant_sales_joined, endorsements_cleaned)

    # Add Medical Endorsements
    medical_endorsement_joined = add_medical_endorsement(endorsements_joined, plot_medical_endorsements_over_time)

    # Create Enforcements Field
    enforcements_cleaned = clean_and_create_enforcements(enforcements)
    enforcements_abbreviated = enforcement_abbreviations(enforcements_cleaned)

    # Join Enforcements
    enforcements_joined = join_enforcements(medical_endorsement_joined, enforcements_abbreviated)

    # Clean and Encode Enforcements
    violations_cleaned = clean_violations(violations)
    #violations_encoded, violations_metadata = violations_mapping(violations_cleaned)

    # Join Violations
    violations_joined = join_violations(enforcements_joined, violations_cleaned)

    # Penalty Variable Creation
    penalty_pay_lag_df = pv.penalty_pay_lag(penalty)
    penalty_helper = pv.penalty_helper_features(penalty_pay_lag_df)
    penalty_non_agg = pv.create_penalty_non_agg_features(penalty_helper)
    penalty_agg_features = pv.create_penalty_agg_features(penalty_non_agg, violations_joined)
    washington_dataset = pv.clean_all_wash_data(penalty_agg_features)
    print('Penalty features added to sales data')
    washington_dataset = washington_dataset.sort_values(by=['License Number', 'Reporting Period'])

    # Download final
    washington_dataset.to_csv('Processed_Data/washington_dataset.csv', index=False)
    print('downloaded washington_dataset')

if __name__ == "__main__":
    main()
