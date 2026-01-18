import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

file_name = 'laptop_pricing_dataset_mod2.csv'
df = pd.read_csv(file_name)

df.fillna(np.nan, inplace=True)

print(df)

#SLR

print('\n ///////////////////   Single Linear Regression ////////////// \n')
lm = LinearRegression()

x = df[["CPU_frequency"]]
y = df[["Price"]]

lm.fit(x,y)

Yhat = lm.predict(x)
mse = mean_squared_error(y,Yhat)
print(Yhat[0:5])
print('The R-square is: ', lm.score(x, y))
print('The mean square error of price and predicted value is: ', mse)


#MLR
print('\n ///////////////////   Multiple Linear Regression ////////////// \n')

lm_multiple = LinearRegression()
Z = df[["CPU_frequency","RAM_GB","Storage_GB_SSD","CPU_core","OS","GPU","Category"]]

lm_multiple.fit(Z,y)
ZHat = lm_multiple.predict(Z)
print(ZHat[0:5])


#distplot will be deprecated
# ax1 = sns.distplot(y,hist=False,color="r", label = "Actual value")
# sns.distplot(Yhat,hist=False, color="b",label="Fitted Values", ax=ax1)
# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Price')
# plt.ylabel('Proportion of laptops')
# plt.legend(['Actual Value', 'Predicted Value'])
# plt.ylim(0)
# plt.show()

#Polynomial regression
print('\n ///////////////////   Polinomial Regression ////////////// \n')
x = x.to_numpy().flatten()
f1 = np.polyfit(x,y,1)
#Converting matrix to array
f1 = f1.ravel()
p1 = np.poly1d(f1)

f2 = np.polyfit(x,y,3)
#Converting matrix to array
f2 = f2.ravel()
p2 = np.poly1d(f2)

f3 = np.polyfit(x,y,5)
#Converting matrix to array
f3 = f3.ravel()
p3 = np.poly1d(f3)

def plotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')

r_squared_1 = r2_score(y, p1(x))
print('The R-square value for 1st degree polynomial is: ', r_squared_1)
print('The MSE value for 1st degree polynomial is: ', mean_squared_error(y,p1(x)))
r_squared_3 = r2_score(y, p2(x))
print('The R-square value for 3rd degree polynomial is: ', r_squared_3)
print('The MSE value for 3rd degree polynomial is: ', mean_squared_error(y,p2(x)))
r_squared_5 = r2_score(y, p3(x))
print('The R-square value for 5th degree polynomial is: ', r_squared_5)
print('The MSE value for 5th degree polynomial is: ', mean_squared_error(y,p3(x)))

# ax1 = sns.distplot(y,hist=False,color="r", label = "Actual value")
# sns.distplot(ZHat,hist=False, color="b",label="Fitted Values", ax=ax1)
# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Price')
# plt.ylabel('Proportion of laptops')
# plt.legend(['Actual Value', 'Predicted Value'])
# plt.ylim(0)
# plt.show()

# plotPolly(p1,x,y,'CPU_frequency')
# plotPolly(p2,x,y,'CPU_frequency')
# plotPolly(p3,x,y,'CPU_frequency')
# plt.show()
print('\n ///////////////////   Pipeline ////////////// \n')
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
print(ypipe[0:5])

print('MSE for multi-variable polynomial pipeline is: ', mean_squared_error(y, ypipe))
print('R^2 for multi-variable polynomial pipeline is: ', r2_score(y, ypipe))