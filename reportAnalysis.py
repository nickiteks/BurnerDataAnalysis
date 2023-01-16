import pandas as pd
from dataPreparation import DataPreparation
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('report.csv', delimiter=';', decimal=',')

preparation = DataPreparation()

data = data.loc[:, ~data.columns.str.contains('^Unnamed: 0')]
data = data.loc[:, ~data.columns.str.contains('^Unnamed: 8')]
data = data.loc[:, ~data.columns.str.contains('^Unnamed: 9')]
data = data.loc[:, ~data.columns.str.contains('^Unnamed: 10')]

data = preparation.delete_columns(['L1',
                                   'L2',
                                   'a2',
                                   'N3',
                                   'L3',
                                   'S2',
                                   'NН3',
                                   'Н2',
                                   'Line Probe 8: Direction [-1,0,0] (m)',
                                   'Line Probe 8: Direction [-1,0,0] (m).1',
                                   'Line Probe 8: Direction [-1,0,0] (m).2',
                                   'Line Probe 8: Direction [-1,0,0] (m).3']
                                  , data)

print(data.info())
print(data.isnull().sum())

sns.set()
ax = sns.heatmap(data.corr(), annot=True, fmt='.1f')
plt.show()
