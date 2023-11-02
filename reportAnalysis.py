import pandas as pd
from sklearn.model_selection import train_test_split

from dataPreparation import DataPreparation
from models import Models

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('report.csv', delimiter=';', decimal=',')

preparation = DataPreparation()
models = Models()

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

columns = ['Line Probe 8: Temperature (K)',
           'Line Probe 8: Mass Fraction of Nitrogen Oxide Emission',
           'Line Probe 8: Mass Fraction of CO',
           'Line Probe 8: Mass Fraction of H2O']

for i in columns:
    print(f'---{i}---')
    print('Mean--',data[i].mean())
    print('Min--', data[i].min())
    print('Max--', data[i].max())



# sns.set()
# ax = sns.heatmap(data.corr(), annot=True, fmt='.1f')
# plt.show()

data_temperature = preparation.delete_columns(['Line Probe 8: Mass Fraction of Nitrogen Oxide Emission',
                                               'Line Probe 8: Mass Fraction of CO',
                                               'Line Probe 8: Mass Fraction of H2O'], data)

x = data_temperature.drop('Line Probe 8: Temperature (K)', axis=1)
y = data_temperature['Line Probe 8: Temperature (K)']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

models.cat_boost_regression(X_train, X_test, y_train, y_test)
models.xg_boost_regression(X_train, X_test, y_train, y_test)
models.random_forest_regression(X_train, X_test, y_train, y_test)

# data_Nox = preparation.delete_columns(['Line Probe 8: Temperature (K)',
#                                        'Line Probe 8: Mass Fraction of CO',
#                                        'Line Probe 8: Mass Fraction of H2O'], data)
#
# x = data_Nox.drop('Line Probe 8: Mass Fraction of Nitrogen Oxide Emission', axis=1)
# y = data_Nox['Line Probe 8: Mass Fraction of Nitrogen Oxide Emission']
#
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
#
# models.cat_boost_regression(X_train, X_test, y_train, y_test)
# models.xg_boost_regression(X_train, X_test, y_train, y_test)
# models.random_forest_regression(X_train, X_test, y_train, y_test)

# data_Co = preparation.delete_columns(['Line Probe 8: Temperature (K)',
#                                       'Line Probe 8: Mass Fraction of Nitrogen Oxide Emission',
#                                       'Line Probe 8: Mass Fraction of H2O'], data)
#
# x = data_Co.drop('Line Probe 8: Mass Fraction of CO', axis=1)
# y = data_Co['Line Probe 8: Mass Fraction of CO']
#
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
#
# models.cat_boost_regression(X_train, X_test, y_train, y_test)
# models.xg_boost_regression(X_train, X_test, y_train, y_test)
# models.random_forest_regression(X_train, X_test, y_train, y_test)

# data_H2O = preparation.delete_columns(['Line Probe 8: Temperature (K)',
#                                        'Line Probe 8: Mass Fraction of Nitrogen Oxide Emission',
#                                        'Line Probe 8: Mass Fraction of CO'], data)
#
# x = data_H2O.drop('Line Probe 8: Mass Fraction of H2O', axis=1)
# y = data_H2O['Line Probe 8: Mass Fraction of H2O']
#
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
#
# models.cat_boost_regression(X_train, X_test, y_train, y_test)
# models.xg_boost_regression(X_train, X_test, y_train, y_test)
# models.random_forest_regression(X_train, X_test, y_train, y_test)

# model = CatBoostRegressor(loss_function='RMSE')
#
# train_dataset = Pool(x_learn,y_learn)
#
# grid = {'iterations': [100, 1200, 2000],
#         'learning_rate': [0.03, 0.1, 0.01, 0.05],
#         'depth': [10, 20, 50, 100, 500]}
#
# model.grid_search(grid, train_dataset)
#
# pred = model.predict(x.iloc[[970]])
#
#
# print(pred)
# print(y.iloc[[970]])
#
# feature_importance = model.feature_importances_
# sorted_idx = np.argsort(feature_importance)
# fig = plt.figure(figsize=(12, 6))
# plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
# plt.yticks(range(len(sorted_idx)), np.array(x.columns)[sorted_idx])
# plt.title('Feature Importance')
# plt.show()
