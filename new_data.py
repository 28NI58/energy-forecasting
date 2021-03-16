import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas.tests.frame.test_validate import dataframe
import pytest


from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.cluster import KMeans



# Put dataset on my github repo
df = pd.read_csv(r'C:\Users\asus\PycharmProjects\phase\school_ds.csv')
# create a dataframe
print(df)

# selecting rows based on condition
# getting a series object containing
# minimum value from each column
# of given dataframe


#minvalue_series = df.min()
#print(minvalue_series)
#select_color = df.loc[df['active'] == 1]


# print(df.loc[df['air_temperature'] == -6.7])

plt.scatter(df['air_temperature'],df['val(KWh)'])
plt.title("Scatterplot ")
plt.xlabel("categorized temperature")
plt.ylabel("Energy(KWh)")
plt.show()

#     **************    CREATE CATEGORIES OF TEMPERATURE INTERVALS       ***********************
# create a list of our conditions
conditions = [
    (df['air_temperature'] <=-7),
	(df['air_temperature'] >=-7 ) & (df['air_temperature'] <= -4),
	(df['air_temperature'] >=-4 ) & (df['air_temperature'] <= -1),
	(df['air_temperature'] >=-1 ) & (df['air_temperature'] <= 2),
	(df['air_temperature'] >= 2) & (df['air_temperature'] <= 5),
	(df['air_temperature'] >= 5 ) & (df['air_temperature'] <= 8),
	(df['air_temperature'] >= 8 ) & (df['air_temperature'] <= 11),
	(df['air_temperature'] >= 11 ) & (df['air_temperature'] <= 14),
	(df['air_temperature'] >= 14 ) & (df['air_temperature'] <= 17),
	(df['air_temperature'] >= 17 ) & (df['air_temperature'] <= 20),
	(df['air_temperature'] >= 20 ) & (df['air_temperature'] <= 23),
	(df['air_temperature'] >= 23 ) & (df['air_temperature'] <= 26),
	(df['air_temperature'] >= 26 ) & (df['air_temperature'] <= 29),
	(df['air_temperature'] >= 29 ) & (df['air_temperature'] <= 32),
	(df['air_temperature'] > 32)
    ]

# create a list of the values we want to assign for each condition
values = [-10,-5.5,-2.5,0.5,3.5,6.5,9.5,12.5,15.5,18.5,21.5,24.5,27.5,30.5,34]

# create a new column and use np.select to assign values to it using our lists as arguments
df['temp'] = np.select(conditions, values)
print(df.head())

#     **************    Classify as working and non working hours     ***********************
# create a list of our conditions

conditions2 = [
    (df['Hour'] <8) & (df['Hour'] >19 ),
	(df['Hour'] >=8 ) & (df['Hour'] <= 19),
	]

values2 = [0,1]
# create a new column and use np.select to assign values to it using our lists as arguments
df['working'] = np.select(conditions2, values2)
print(df.head())

# MAKE TEMP AND WORKING TO DATASET
df.to_csv("raw.csv", index=False)