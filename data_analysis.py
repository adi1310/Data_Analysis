# Importing the  Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(url, header = None)

headers = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors",
           "body-style","drive-wheels","engine-location","wheel-base","length","width",
           "height","curb-weight","engine-type","num-of-cylinders","engine-size",
           "fuel-system","bore","stroke","compression-ratio","horse-power","peak-rpm",
           "city=mpg","highway-mpg","price"]

df.columns = headers

# ---To Know the datatype of each columns
df.dtypes

# ---To take the first 10 rows of dataset
df.head(10)

# ---To take the last 10 rows of dataset
df.tail(10)

df.describe(include = "all")

df.info()

# ---To store the dataset in system 
path = "/home/aditya1310/Documents/Data Analysis/automobile.csv"
df.to_csv(path)

# Data Wrangling

# ---Missing Values

df["price"].replace("?", np.NaN, inplace = True)
df.dropna(subset = ["price"], inplace = True)
mean = df["price"].mean()
df["price"].replace(np.NaN, mean, inplace = True)

df["horse-power"].replace("?", np.NaN, inplace = True)
df.dropna(subset = ["horse-power"], inplace = True)
mean = df["horse-power"].mean()
df["horse-power"].replace(np.NaN, mean, inplace = True)

# ---Data Formating
df["city-mpg"] = 235/df["city-mpg"]
df.rename(columns = {"city-mpg": "city-L/100km"}, inplace = True)

# ---Convert the datatype as requirement
df["price"] = df["price"].astype('int64')
df["horse-power"] = df["horse-power"].astype('int64')

# ---Data Normailization

# ------Feature Scaling
df["length"] = df["length"]/df["length"].max()
# ------Min Max method
df["width"] = (df["width"] - df["width"].min())/(df["width"].max() - df["length"].min())
# ------Z-Score method
df["height"] = (df["height"]-df["height"].mean())/df["height"].std()

# ---Bining=it is a method to provide a particular value to a range for better result
bins = np.linspace(min(df["price"]), max(df["price"]),4)
group_names = ["Low","Medium","High"] 
df["price_binned"] = pd.cut(df["price"], bins, labels = group_names, include_lowest = True)

# ---Catogorlcal Variable
pd.get_dummies(df["fuel-type"])

# Exploratory Data Analysis

# ---Descriptive Statistics
# ------for the numeric type data
describe = df.describe()
# ------for the categorical variable
drive_wheel_counts = df["drive-wheels"].value_counts()
drive_wheel_counts.rename(columns={"drive-wheels": "value_counts"}, inplace =True)
drive_wheel_counts.index.name = 'drive-wheels'
# ------boxplot for visualising the contribution to price by each type
sns.boxplot(x ='drive-wheels', y ='price', data =df) 
# ------scatter plot b/w the independent variable and target values
x = df["engine-size"]
y = df["price"]
plt.scatter(x, y)
plt.title("engine-size vs price")
plt.xlabel("engine-size")
plt.ylabel("price")
# ------Groupby
df_test = df[["drive-wheels","body-style","price"]]
df_grp = df_test.groupby(["drive-wheels","body-style"], as_index = False).mean()
df_pivot = df_grp.pivot(index = "drive-wheels", columns = "body-style")
# ------Heatmap
plt.pcolor(df_pivot, cmap = "RdBu")
plt.colorbar()
plt.show()

# ---Corelation
sns.regplot(x = "engine-size", y ="price", data = df)
plt.ylim(0, )
sns.regplot(x = "highway-mpg", y ="price", data = df)
plt.ylim(0, )

# ---Pearson Corelation
from scipy.stats import pearsonr
pearson_coef, p_value = pearsonr(df["horse-power"], df["price"])
    # pearson coeff should be close +1 or -1 and p_value lass than 0.001 for better result 
    # P_value tells us significant figure b/w the independent and target variable  
# ---Anova (Analysis of Variance)
from scipy.stats import f_oneway
df_anova = df[["make","price"]]    
grouped_anova = df_anova.groupby(["make"])
anova_results = f_oneway(grouped_anova.get_group("honda")["price"],
                         grouped_anova.get_group("subaru")["price"])
    # for better result f_value(1) should be higher and p_value(0) 
    # should be less than 0.001 
anova_results_1 = f_oneway(grouped_anova.get_group("honda")["price"],
                         grouped_anova.get_group("jaguar")["price"])