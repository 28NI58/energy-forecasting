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
df = pd.read_csv(r'C:\Users\asus\PycharmProjects\phase\fill1.csv')
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
sf = df.loc[df['working'] == 1]
print (sf)

plt.scatter(sf['air_temperature'],sf['val(KWh)'])
plt.title("Scatterplot ")
plt.xlabel("categorized temperature")
plt.ylabel("Energy(KWh)")
plt.show()



X = sf[['temp','val(KWh)']].to_numpy()
# k-means clustering
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
# define the model
model = KMeans(n_clusters=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot

plt.title('K means Clustering for categorized complete data(2 clusters) ')
pyplot.show()
