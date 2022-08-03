####################################################
## Author: Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Date: Spring 2021
####################################################

"""
Main method concatenates sales dataframes, removes duplicates from the sales tables and creates an 'alt_sales' field
- exports cleaned Sales table to be used in Cleaning_Pipeline.py
"""

from Cleaning_Pipeline import *
import utils_temporal
import os

DUPLICATES_PATH = 'Processed_Data/duplicate_sales.csv'

def export_duplicate_sales(wa):
    """
    Finds duplicate sales based on (License #, MM/YYYY) primary key
    - export dataframe to csv for reference (not used directly to compute alternate sales)
    """
    sales = wa.iloc[:, :4]
    grouped = sales.groupby(['License Number', 'Reporting Period']).size().reset_index()
    dulplicates_df = grouped[grouped[0] > 1]
    dulplicates_df.to_csv(DUPLICATES_PATH, index=False)


def create_alt_sales(sales):
    """
    Implements logic to create 'alt_sales' field
    - if sales has duplicates:
        keep the larger 'Total Sales'
        store other value in 'alt_sales'
    - then drop duplicates
    """
    sales_clean = sales.copy()
    sales_clean['Total Sales'] = sales_clean['Total Sales'].fillna(0)

    sales_max = sales_clean.groupby(['License Number', 'Reporting Period'])['Total Sales'].max().reset_index()
    sales_min = sales_clean.groupby(['License Number', 'Reporting Period'])['Total Sales'].min().reset_index()
    alt_sales = sales_min['Total Sales'] * (sales_min['Total Sales'] != sales_max['Total Sales']).astype(int)

    sales_clean = sales_clean.sort_values('Total Sales', ascending=False)
    sales_clean = sales_clean.drop_duplicates(subset=['License Number', 'Reporting Period'], keep='first')
    sales_clean['alt_sales'] = alt_sales

    sales_clean['Total Sales'] = sales_clean['Total Sales'].replace({0: np.nan})

    # fix column formatting
    cols = sales_clean.columns.to_list()
    cols.insert(4, cols.pop(cols.index('alt_sales')))
    sales_clean = sales_clean.reindex(columns=cols)

    return sales_clean


def add_temporal_variables(sales):
    """
    wrapper function that adds sales data using utils_temporal.py
    """
    sales["age"] = utils_temporal.getAge(sales)
    sales["failure"] = utils_temporal.getFailure(sales)
    sales = utils_temporal.getLaggedSales(sales)
    sales["sales_MoM"], sales["sales_YoY"] = utils_temporal.getSalesGrowth(sales, sales_offset_1m=sales["lagged_sales_1m"], sales_offset_1y=sales["lagged_sales_1y"])
    return sales


def main():
    # Import Datasets
    wa = pd.read_csv('Processed_Data/washington_dataset.csv')

    if not os.path.isfile(DUPLICATES_PATH):
        export_duplicate_sales(wa)

    sales_no_duplicates = create_alt_sales(wa)

    num_duplicates = len(wa) - len(sales_no_duplicates)
    print(f'Computed Alternate Sales, {num_duplicates} Duplicate sales removed from WA dataset')
    sales_no_duplicates = add_temporal_variables(sales_no_duplicates)
    sales_no_duplicates.to_csv("Processed_Data/washington_dataset_no_duplicates.csv", index=False)
    print('Downloaded washington_dataset_no_duplicates.csv')


if __name__ == "__main__":
    main()