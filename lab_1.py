import pandas as pd
import numpy as np

file_name = 'auto.csv'

df = pd.read_csv(file_name)

# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
#print("headers\n", headers)
df.columns = headers
#replace "?" to NaN
df.replace("?", np.nan, inplace=True)

df.dropna(subset=["price"], axis=0, inplace=True)

# print dataframe

# print(df.describe(include='all'))

print(df['price'].min())

#df.to_csv("auto_cleaned.csv", index=False)