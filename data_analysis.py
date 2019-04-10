# Importing the  Dataset

import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(url, header = None)

headers = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors",
           "body-style","drive-wheels","engine-location","wheel-base","length","width",
           "height","curb-weight","engine-type","num-of-cylinders","engine-size",
           "fuel-system","bore","stroke","compression-ratio","horse-power","peak-rpm",
           "city=mpg","highway-mpg","price"]

df.columns = headers

# To Know the datatype of each columns
df.dtypes()

# To take the first 10 rows of dataset
df.head(10)

# To take the last 10 rows of dataset
df.tail(10)

df.describe(include = "all")

df.info()

# To store the dataset in system 
path = "/home/aditya1310/Documents/Data Analysis/automobile.csv"
df.to_csv(path)

# Data Wrangling

# ---Missing Values
df["price"].dropna(inplace = True)

mean = df["price"].mean()
df["price"].replace(np.nan, mean)

# ---Data Formating
df["city-mpg"] = 235/df["city-mpg"]
df.rename(columns = {"city-mpg": "city-L/100km"}, inplace = True)

# ---Convert the datatype as requirement
df["price"] = df["price"].astype('int64')

# ---Data Normailization

# ------Feature Scaling
df["length"] = df["length"]/df["length"].max()
# ------Min Max method
df["width"] = (df["width"] - df["width"].min())/(df["width"].max() - df["length"].min())
# ------Z-Score method
df["height"] = (df["height"]-df["height"].mean())/df["height"].std()

# ---Bining--it is a method to provide a particular value to a range for better result
import numpy as np
bins = np.linspace(min(df["price"]), max(df["price"]),4)
group_names = ["Low","Medium","High"] 
df["price_binned"] = pd.cut(df["price"], bins, labels = group_names, include_lowest = True)






