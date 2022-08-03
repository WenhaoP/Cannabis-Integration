import pandas as pd
import numpy as np



def getAge(sales_data, id_col="License Number", date_col="Reporting Period"):
  """
  Create age variable. # of months since first month company had sales. Age 0 in first month.
  """
  license_dates = sales_data[[id_col]].copy()
  license_dates["date"] = pd.to_datetime(sales_data[date_col])
  inceptions = license_dates.groupby(id_col).min().rename(columns=lambda _: "inception_date")

  license_dates = license_dates.merge(inceptions, how="left", left_on=id_col, right_index=True)
  license_dates["age"] = 12*(license_dates["date"].dt.year - license_dates["inception_date"].dt.year)
  license_dates["age"] += license_dates["date"].dt.month - license_dates["inception_date"].dt.month
  return license_dates["age"]



def getFailure(sales_data, id_col="License Number", date_col="Reporting Period"):
  """
  Create failure variable, coded 0 in a focal month if a dispensary exists in the following month and 
    coded 1 if it does not exist after that month. 
    Exception is for the last month of the data set
  """
  license_dates = sales_data[[id_col]].copy()
  license_dates["date"] = pd.to_datetime(sales_data[date_col])
  final_dates = license_dates.groupby(id_col).max().rename(columns=lambda _: "final_date")
  global_data_cutoff = final_dates["final_date"].max()

  license_dates = license_dates.merge(final_dates, how="left", left_on=id_col, right_index=True)
  license_dates["failure"] = (license_dates["date"] == license_dates["final_date"]) & (license_dates["date"] != global_data_cutoff)
  license_dates["failure"] = license_dates["failure"].astype(int)
  return license_dates["failure"]



def getLaggedSales(sales_data, id_col="License Number", date_col="Reporting Period", sales_col="Total Sales"):
  """
  Create lagged_sales_1m and lagged_sales_1y variable, which is the preceding months sales.
  """
  license_dates = sales_data#[[id_col]].copy()
  license_dates["date"] = pd.to_datetime(sales_data[date_col])

  # offset sales dates to be rejoined with it's future date
  license_offsets = sales_data[[id_col, sales_col]].copy()
  license_offsets["date_offset_1m"] = license_dates["date"] + pd.DateOffset(months=1)
  license_offsets["date_offset_1y"] = license_dates["date"] + pd.DateOffset(years=1)

  # merge offset sales onto it's own true sales
  license_dates = license_dates.merge(
      license_offsets[[id_col, "date_offset_1m", sales_col]].rename(columns={sales_col: "lagged_sales_1m"}),
      how="left", left_on=[id_col, "date"], right_on=[id_col, "date_offset_1m"])
  license_dates = license_dates.merge(
      license_offsets[[id_col, "date_offset_1y", sales_col]].rename(columns={sales_col: "lagged_sales_1y"}),
      how="left", left_on=[id_col, "date"], right_on=[id_col, "date_offset_1y"])

  license_dates = license_dates.drop("date", axis=1)
  return license_dates



def getSalesGrowth(sales_data, id_col="License Number", date_col="Reporting Period", sales_col="Total Sales", 
                   sales_offset_1m=None, sales_offset_1y=None):
  """
  Create sales month_over_month_growth variable: 
    [(total sales - total sales from 1 months ago)/total sales from 1 months ago]*100.
  Leave blank if there weren't total sales 1 month ago.
  """
  license_sales = sales_data[[id_col, sales_col]].copy()
  license_sales["sales_MoM"] = 100 * (license_sales[sales_col] - sales_offset_1m)/sales_offset_1m
  license_sales["sales_YoY"] = 100 * (license_sales[sales_col] - sales_offset_1y)/sales_offset_1y
  return license_sales["sales_MoM"], license_sales["sales_YoY"]

