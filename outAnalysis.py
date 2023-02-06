import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv('csv/report.csv', delimiter=';', decimal=',')

print(data.head(5))
print(data.info())

# распределение и мин, макс и тд
columns = ['Line Probe 8: Temperature (K)',
           'Line Probe 8: Mass Fraction of Nitrogen Oxide Emission',
           'Line Probe 8: Mass Fraction of CO',
           'Line Probe 8: Mass Fraction of H2O']

for column in columns:

    y = data[column]

    print(f'--{column}--')
    print(y.describe())

    sns_plot = sns.histplot(y)
    fig = sns_plot.get_figure()
    plt.show()

# классификация

x = data.copy()
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit_predict(x)

print(kmeans)

