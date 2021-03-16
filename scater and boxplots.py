import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas.tests.frame.test_validate import dataframe
import pytest

from scipy.stats import pearsonr
from scipy.stats import spearmanr
# Put dataset on my github repo
my_df = pd.read_csv(r'C:\Users\asus\PycharmProjects\phase\fill1.csv')
# create a dataframe
print(my_df)

df = pd.DataFrame(my_df, columns= ['air_temperature','val(KWh)','active','temp','Hour','working'])
print(df)


select_color = df.loc[df['working'] == 1]
print (select_color)

# *************** BOX PLOT *******************

sns.boxplot(x='temp', y='val(KWh)', data=select_color)
# my_df.boxplot(column = 'vall', by = 'temperature')
plt.title('Box plot')
plt.show()

# *********** SCATTER PLOT **********************
plt.scatter(df['temp'],df['val(KWh)'])
plt.title("Scatterplot ")
plt.xlabel("categorized temperature")
plt.ylabel("Energy(KWh)")
plt.show()

