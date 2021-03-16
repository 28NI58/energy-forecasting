import dcor
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import pearsonr, kendalltau
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


from minepy import MINE

# Put dataset on my github repo
my_df = pd.read_csv(r'C:\Users\asus\PycharmProjects\phase\raw_ds.csv')

data1 = my_df.air_temperature
data2 = my_df.energy

print(' PEARSSSON ')
print(my_df.corr(method ='spearman'))
print(1 - distance.correlation(data1, data2))

corr, _ = pearsonr(data1, data2)

print('Pearsons correlation coefficient: %.3f' % corr)
result = [corr]

corr, p = spearmanr(data1, data2)
print('Spearmans correlation coefficient: %.3f' % corr)
result.append(corr)
alpha = 0.05
#print(p)
#if p > alpha:
   # print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
#else:
   # print('Samples are correlated (reject H0) p=%.3f' % p)

data3 = np.array(data1)
data4 = np.array(data2)
data3 = data3.reshape(-1, 1)


mi = mutual_info_regression(data3, data4)
#mi /= np.max(mi)
print("MUTUAL INFORMATION ")
print('MUTUAL INFORMATION: %.3f' % mi)
print(mi)
result.append(mi[0])

def distcorr(X, Y):
    """ Compute the distance correlation function


    0.762676242417
    """

    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

print('Distance Corelation')
#print(distcorr(data1, data2))
print('Distance Correlation: %.3f' % distcorr(data1,data2))
dit=format(dcor.distance_correlation(data1, data2))
dit=float(dit)
result.append(dit)
#print("distance correlation = {:.2f}".format(dcor.distance_correlation(data1, data2)))
#print("p-value = {:.7f}".format(dcor.independence.distance_covariance_test(data1,
#                                                                          data2,
#                                                                         exponent=1.0,
#                                                                        num_resamples=100)[0]))

mine = MINE(alpha=0.6, c=15)
mine.compute_score(data1,data2)
print('MAXIMAL INFORMATION COEFFICIENT: %.3f' % mine.mic())
minn=mine.mic()
print(type(minn))
result.append(minn)
yax=[0.02,0.04,0.06,0.08,0.10,0.12,0.14]
res=[0.076,0.050,0.134,0.142,0.106]
ress=[corr,dcor]
# resss = np.reshape(ress)
print(result)
print(res)
re=['PEARSON','SPEARMAN','MUTUAL INFORMATION','DISTANCE CORRELATION','MAXIMAL INFORMATION']
New_Colors = ['teal','blue','brown','orange','red']
plt.bar(re,result,color=New_Colors)
plt.title('CORRELATION COEFFICIENTS')
plt.xlabel('')
plt.xticks(rotation = 90)
plt.ylabel('COEFFICIENT')
plt.show()



