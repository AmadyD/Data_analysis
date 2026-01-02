import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

column_to_replace_by_mean = ["normalized-losses", "stroke", "bore","horsepower", "peak-rpm"]
column_to_replace_by_common = ["num-of-doors"]
float_data = ["bore", "stroke","price","peak-rpm"]
int_data = ["normalized-losses"]

file_name = 'auto.csv'

# Load the dataset with specified headers
df = pd.read_csv(file_name, names=headers)

# Replace "?" with NaN
df.replace("?",np.nan , inplace=True)

# Check for missing data
missing_data = df.isnull()


# Count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


def get_mean(column_name):
    mean_value = df[column_name].astype('float').mean(skipna=True, axis=0)
    print(f"mean value of {column_name} : {mean_value}")
    return mean_value

def get_most_common_value(column_name):
    value = df[column_name].value_counts().idxmax()
    print(f"The most common value of {column_name} is: {value}")
    return value

""" function to replace nan by the mean on the column
or you can use directly fillna() function 
"""
def replace_nan_by_mean(column_name):
    df1 = df[column_name].replace(np.nan,get_mean(column_name))
    return df1

# Display the first 5 rows of the dataframe and missing data

# print(df.head(5))
# print(missing_data.head(5))

# print(df)
# print(df.dtypes)

#df['normalized-losses'] = replace_nan_by_mean('normalized-losses')

""" d:\python\data_science\lab_2.py:51: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object."""

#df['normalized-losses'].fillna(get_mean('normalized-losses'), inplace=True)

# Replace missing data from colums by their means
def fillna_by_mean(column_names):
    for column in column_names:
        mean_value = get_mean(column)
        df[column] = df[column].fillna(mean_value)

# Replace missing data from colums by their most common value
def fillna_by_most_common(column_names):
    for column in column_names:
        common_value = get_most_common_value(column)
        df[column] = df[column].fillna(common_value)

def float_converter(colum_names):
    df[colum_names] = df[colum_names].astype("float")

def int_converter(colum_names):
        df[colum_names] = df[colum_names].astype("int")

fillna_by_mean(column_to_replace_by_mean)
fillna_by_most_common(column_to_replace_by_common)

# delete all the rows taht doesn't have a price data
df.dropna(subset=["price"], axis=0, inplace=True)

# reste the indexes
df.reset_index(drop=True, inplace=True)

# convert data types
"""
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
"""
float_converter(float_data)
int_converter(int_data)

# Data standardization

print(df.head(0))

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

print(df.head(0))