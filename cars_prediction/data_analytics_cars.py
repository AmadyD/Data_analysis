import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from flask import Flask, render_template

#app = Flask(__name__)

file_name = 'automobileEDA.csv'
df = pd.read_csv(file_name)

df.replace('?',np.nan, inplace=True)
#plt.switch_backend('agg')


print(df.head(0))

# checking non usable data
for elt in df.head(0):
    isNan = df[elt].isnull().values.any()
    types = df[elt].dtypes
    if(isNan):
        print(f'{elt} contains nan : {isNan} | {types}')

# get correlation between numerical column
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correl = numeric_df.corr()

print (df[['bore','stroke','compression-ratio','horsepower']].corr())

sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0)

print(df[['engine-size','price']].corr())

sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0)

print(df[['highway-mpg', 'price']].corr())

sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0)

print(df[['stroke','price']].corr())
sns.regplot(x="stroke", y="price", data=df)
plt.ylim(0)


# Categoriacal variables ----> using boxplot to visualize

sns.boxplot(x="body-style", y="price", data=df)


sns.boxplot(x="engine-location", y="price", data=df)

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.ylim(0)
plt.show()

# Descriptive statistics

print(df.describe())

# Applying the describe method on type object

print(df.describe(include=['object']))

print(df['drive-wheels'].value_counts())
print(df['drive-wheels'].value_counts().to_frame())

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.reset_index(inplace=True)
drive_wheels_counts = drive_wheels_counts.rename(columns={'drive-wheels': 'value-counts'})
drive_wheels_counts.index.name = 'drive-wheels'

print(drive_wheels_counts)


# Basics of grouping 

print(df['drive-wheels'].unique())

df_group_one = df[['drive-wheels', 'body-style', 'price']]

#Let's calculate the average price for each categorie
df_group_price_mean = df_group_one.groupby(['drive-wheels'], as_index=False).agg({'price':'mean'})
print(df_group_price_mean)

df_group_by_mean = df_group_one.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
print(df_group_by_mean)

#converting dataframe to pivot table
df_group_pivot = df_group_by_mean.pivot(index='drive-wheels', columns='body-style')
df_group_pivot.fillna(0,inplace=True)

df_cars = df[['body-style', 'price']]
df_cars_group = df_cars.groupby(['body-style'], as_index=False).mean()
print(df_cars_group)

plt.pcolor(df_group_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(df_group_pivot, cmap='RdBu')

#label names
row_labels = df_group_pivot.columns.levels[1]
col_labels = df_group_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(df_group_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_group_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

#Correlation and causation

print(df.select_dtypes(include=['number']).corr())

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


'''
def plot_figure():
    global df
    sns.regplot(x="engine-size", y="price", data=df)
    plt.ylim(0)
    return plt

@app.get('/')
def single_converter():
    # Get the matplotlib plot 
    plot = plot_figure()

    # Save the figure in the static directory 
    plot.savefig(os.path.join('static', '../cars_prediction', 'plot.png'))

    return render_template('matplotlib-plot1.html')

if __name__ == '__main__':
   app.run(debug=True)

'''