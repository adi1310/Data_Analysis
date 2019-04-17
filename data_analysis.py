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
           "city-mpg","highway-mpg","price"]

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

# Model Evalution 

# ---Simple Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
x = df[['highway-mpg']]
y = df['price']
lm.fit(x, y)
y_pred = lm.predict(x)
# ------Visualise the model
# ---------Visualise using matplotlib
plt.scatter(x, y,color = 'red')
plt.plot(x, y_pred, color = 'blue')
plt.title('highway-mpg vs price')
plt.xlabel('highway-mpg')
plt.ylabel('price')
plt.show()
# ---------Visualise using seaborn
sns.regplot(x = 'highway-mpg', y = 'price', data = df)
# ---------Visualising the residual value(difference b/w actual and fitted value)
sns.residplot(df['highway-mpg'],df['price'])
# ---------Visualising the distribution plot
ax1 = sns.distplot(df['price'], hist = False, color = "r", label = "Actual Value")
sns.distplot(y_pred, hist = False, color = "b" , label = "Fitted Value", ax= ax1)

# ---Multiple Linear Regression
z = df[['horse-power','curb-weight','engine-size','highway-mpg']]
mlm = LinearRegression()
mlm.fit(z, y)
y0_pred = mlm.predict(z)

# ---Polynomial Regression
# ------Normalization
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(df[['horse-power','highway-mpg']])
x_scale = scale.transform(df[['horse-power','highway-mpg']])
# ------Polynomial Featuring
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 2, include_bias = False)
x_poly = pr.fit_transform(x_scale)
# ------Linear Regression Model
lm.fit(x_poly, y)


# ---Pipeline(All the steps in single process)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
Input = [('scale', StandardScaler()),
         ('pr', PolynomialFeatures(degree =2)),
         ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(df[['horse-power','engine-size','highway-mpg','curb-weight']], y)
y_poly = pipe.predict(df[['horse-power','engine-size','highway-mpg','curb-weight']])

# ---Prediction and visualisation uisng single variable as input in polybnomial regression

# ---Insample Evaluation
from sklearn.metrics import mean_squared_error
mse_linear = mean_squared_error(df['price'], y_pred)
mse_multiple = mean_squared_error(df['price'], y0_pred)
mse_poly = mean_squared_error(df['price'], y_poly)

# Model Evaluation
# ---Spliting Dataset in Training set or Test set
from sklearn.model_selection import train_test_split
x_data = df[["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors",
           "body-style","drive-wheels","engine-location","wheel-base","length","width",
           "height","curb-weight","engine-type","num-of-cylinders","engine-size",
           "fuel-system","bore","stroke","compression-ratio","horse-power","peak-rpm",
           "city-L/100km","highway-mpg",]]

x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size = 0.3, random_state = 0)

# ---Cross Validation score(R Squared)
from sklearn.model_selection import cross_val_score
x_data_cross = df[['horse-power','engine-size','highway-mpg','curb-weight']]
# ---It divides the dataset in three equal partition and train by two partition 
# ---and test the result by other partition these step is repeated 3 times for 
# ---different training set
scores = cross_val_score(pipe, x_data_cross, y, cv = 3)
np.mean(scores)

# ---Cross Validation Prediction
from sklearn.model_selection import cross_val_predict
y_predicted = cross_val_predict(pipe, x_data, y, cv = 3)

# ---R squared value for polynomial regression(horse-power) with order 1,2,3,4
Rsqu_test = []
order = [1,2,3,4,5,6,7,8,9,10]
for n in order:
    pr = PolynomialFeatures(degree = n)
    x_train_pr = pr.fit_transform(x_train[["horse-power"]])
    x_test_pr = pr.fit_transform(x_test[["horse-power"]])
    lm.fit(x_train_pr, y_train)
    Rsqu_test.append(lm.score(x_test_pr, y_test))
    
# ---Grid Search
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
parameters1 = [{'alpha': [0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000]}]
RR = Ridge()
Grid1 = GridSearchCV(RR, parameters1, cv = 4)
Grid1.fit(x_data_cross, y)
Grid1.best_estimator_
scores = Grid1.cv_results_
scores['mean_test_score'] 