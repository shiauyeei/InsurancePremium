# Import Library
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Import Pre-processing Package
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Import Regression Package
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble

# Read data
df = pd.read_csv(r'C:\Users\Jimosy\Desktop\Practicum NTU\insurance.csv')
cha = df.iloc[:,6]
x_df = df.iloc[:,0:6]

# Check for NULL & Data Type
df.info()
df.head()

# Transform Categorical Variable
labenc = LabelEncoder()
df.region = labenc.fit_transform(df["region"])
df.smoker = labenc.fit_transform(df["smoker"])
df.sex = labenc.fit_transform(df["sex"])

# Data Exploration
# Charges Distribution
plt.figure(1, figsize = (12,6))
sbn.set_style("whitegrid", {'grid.linestyle': '--'})
sbn.distplot(df["charges"])

# LogCharges Distribution
logcha = np.log(df["charges"])
plt.figure(2, figsize = (12,6))
sbn.set_style("whitegrid", {'grid.linestyle': '--'})
sbn.distplot(logcha)

# Age Distribution
plt.figure(3, figsize = (12,6))
sbn.set_style("whitegrid", {'grid.linestyle': '--'})
sbn.distplot(df["age"])

# BMI Distribution
plt.figure(4, figsize = (12,6))
sbn.set_style("whitegrid", {'grid.linestyle': '--'})
sbn.distplot(df["bmi"])

# Children Distribution
plt.figure(5, figsize = (12,6))
sbn.set_style("whitegrid", {'grid.linestyle': '--'})
sbn.catplot(x="children", kind="count", palette="ch:.25", data=df, size = 6)

# Correlation check
f,ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sbn.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sbn.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)

plt.figure(7)
sbn.pairplot(df, hue="smoker")
plt.title("Pairplot for All attribute with Smokers Filter")

# Number of Gender Count seperated by smoker
plt.figure(8, figsize = (12,6))
sbn.set_style("whitegrid", {'grid.linestyle': '--'})
sbn.catplot(x= "sex", kind="count",hue = 'smoker', data=df)

# Charges Distribution Split by Smoker
f= plt.figure(figsize=(12,8))
 
ax=f.add_subplot(111)
sbn.distplot(df[(df.smoker == 1)]["charges"], bins = 20, color='r',ax=ax)

ax=f.add_subplot(111)
sbn.distplot(df[(df.smoker == 0)]["charges"],bins = 20, color='b',ax=ax)
ax.set_title('Distribution of charges for smokers(Red) and non-smokers(blue)')

A = (df[(df.smoker == 0)])


# Age vs Charges split by sex
sbn.set_style("whitegrid", {'grid.linestyle': '--'})
plt.figure(5, figsize = (12,6))
sbn.scatterplot(df["age"], cha, df["sex"])
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Distribution of charges by age and sex")

# Age vs Charges split by smoker
sbn.set_style("whitegrid", {'grid.linestyle': '--'})
plt.figure(10, figsize = (12,6))
sbn.scatterplot(df["age"], cha, df["smoker"])
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Distribution of charges by age and smokers")

plt.figure(11, figsize = (12,6))
sbn.lmplot(x="age", y="charges", hue="smoker", data=df, size = 9)
plt.title("Distribution of charges by age and smokers - LM plot")

# BMI vs Charges split by Smoker
sbn.set_style("whitegrid", {'grid.linestyle': '--'})
plt.figure(12, figsize = (12,6))
sbn.scatterplot(df["bmi"], cha, df["smoker"])
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("Distribution of charges by smokers and BMI")

plt.figure(13, figsize = (12,6))
sbn.lmplot(x="bmi", y="charges", hue="smoker", data=df, size = 9)
plt.title("Distribution of charges by BMI and smokers - LM plot")

# Children vs Charges split by Smoker
plt.figure(14, figsize = (12,6))
sbn.boxplot(df["children"], cha, df["smoker"])
plt.xlabel("Children")
plt.ylabel("Charges")
plt.title("Boxplot of charges by number of children and smokers")

# Region vs Charges split by Smoker
plt.figure(15, figsize = (10,5))
sbn.boxplot(x_df["region"], cha, df["smoker"])
plt.xlabel("Region")
plt.ylabel("Charges")
plt.title("Boxplot of charges by Region and smokers")

# Data Preprocessing
lbenc = LabelEncoder()
x1 = lbenc.fit_transform(df["region"])
enc = OneHotEncoder(categorical_features = "all")
reg_enc = enc.fit_transform(x1.reshape(-1,1)).toarray()
x2 = pd.DataFrame(reg_enc, columns=['NorthEast', 'NorthWest', 'SouthEast', 'SouthWest'])
df_new = pd.merge(df, x2, right_index = True, left_index = True)

# Feature engineering
df_new["BMI_cat"] = np.nan
lst = [df_new]

for col in lst:
    col.loc[col["bmi"] < 18.5, "BMI_cat"] = "Underweight"
    col.loc[(col["bmi"] >= 18.5) & (col["bmi"] < 24.986), "BMI_cat"] = "Normal Weight"
    col.loc[(col["bmi"] >= 25) & (col["bmi"] < 29.926), "BMI_cat"] = "Overweight"
    col.loc[col["bmi"] >= 30, "BMI_cat"] = "Obese"

x3 = lbenc.fit_transform(df_new["BMI_cat"])
BMI_enc = enc.fit_transform(x3.reshape(-1,1)).toarray()
x4 = pd.DataFrame(BMI_enc, columns=['NormalWeight', 'Obese', 'Overweight', 'Underweight'])
df_new2 = pd.merge(df_new, x4, right_index = True, left_index = True)

# Building "CLEAN" Dependent and independent Variable Dataframe
# "age", "sex", "bmi", "children", "smoker", "region", "charges", "BMI_cat",
# "NorthEast", "NorthWest", "SouthEast", "SouthWest", 
# "NormalWeight", "Obese", "Overweight", "Underweight"

x_data = df_new2.drop(["charges", "region", "BMI_cat", "NormalWeight", "Obese", "Overweight", "Underweight", "sex"], axis = 1)

# Regression Model
# Basic Linear Model

x_train,x_test,y_train,y_test = train_test_split(x_data, logcha, test_size=0.2, random_state = 0)
lr = LinearRegression().fit(x_train,y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print("\n" + "Linear Regression Output:")
print(lr.score(x_test,y_test))
print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,y_train_pred),
mean_squared_error(y_test,y_test_pred)))
print('MAE train data: %.3f, MAE test data: %.3f' % (
mean_absolute_error(y_train,y_train_pred),
mean_absolute_error(y_test,y_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,y_train_pred),
r2_score(y_test,y_test_pred)))

MSEscores = cross_val_score(lr, x_data, logcha, cv=5, scoring = "neg_mean_squared_error")
print("Cross Validation neg MSE: %.3f" %(MSEscores.mean()))
MAEscores = cross_val_score(lr, x_data, logcha, cv=5, scoring = "neg_mean_absolute_error")
print("Cross Validation neg MAE: %.3f" %(MAEscores.mean()))
R2scores = cross_val_score(lr, x_data, logcha, cv=5, scoring = "r2")
print("Cross Validation R^2: %.3f" %(R2scores.mean()))

# OLS
results = sm.OLS(cha, x_data).fit()
print(results.summary())


# Random Forest
forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(x_train,y_train)
forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

print("\n" + "Random Forest Output:")
print(forest.score(x_test,y_test))
print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,forest_train_pred),
mean_squared_error(y_test,forest_test_pred)))
print('MAE train data: %.3f, MAE test data: %.3f' % (
mean_absolute_error(y_train,forest_train_pred),
mean_absolute_error(y_test,forest_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))

MSEscores = cross_val_score(forest, x_data, logcha, cv=5, scoring = "neg_mean_squared_error")
print("Cross Validation neg MSE: %.3f" %(MSEscores.mean()))
MAEscores = cross_val_score(forest, x_data, logcha, cv=5, scoring = "neg_mean_absolute_error")
print("Cross Validation neg MAE: %.3f" %(MAEscores.mean()))
R2scores = cross_val_score(forest, x_data, logcha, cv=5, scoring = "r2")
print("Cross Validation R^2: %.3f" %(R2scores.mean()))


# Ensemble learning
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gboost = ensemble.GradientBoostingRegressor(**params)
gboost.fit(x_train, y_train)
gboost_train_pred = gboost.predict(x_train)
gboost_test_pred = gboost.predict(x_test)

print("\n" + "Gradient Boosting Regression Output:")
print(gboost.score(x_test,y_test))
print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,gboost_train_pred),
mean_squared_error(y_test,gboost_test_pred)))
print('MAE train data: %.3f, MAE test data: %.3f' % (
mean_absolute_error(y_train,gboost_train_pred),
mean_absolute_error(y_test,gboost_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,gboost_train_pred),
r2_score(y_test,gboost_test_pred)))

MSEscores = cross_val_score(gboost, x_data, logcha, cv=5, scoring = "neg_mean_squared_error")
print("Cross Validation neg MSE: %.3f" %(MSEscores.mean()))
MAEscores = cross_val_score(gboost, x_data, logcha, cv=5, scoring = "neg_mean_absolute_error")
print("Cross Validation neg MAE: %.3f" %(MAEscores.mean()))
R2scores = cross_val_score(gboost, x_data, logcha, cv=5, scoring = "r2")
print("Cross Validation R^2: %.3f" %(R2scores.mean()))


