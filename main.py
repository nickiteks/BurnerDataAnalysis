import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree._criterion import MSE

from dataPreparation import DataPreparation
from models import Models

import seaborn as sns
from catboost import CatBoostClassifier
from catboost import Pool
from catboost import CatBoostRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve, RocCurveDisplay, roc_auc_score, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv('book.csv', delimiter=';', decimal=',')

# очистка данных

preparation = DataPreparation()
models = Models()

data = preparation.delete_columns(['O2, %',
                                   'Тип амбразуры',
                                   'Модель горения',
                                   'Модель турбулентности',
                                   'Параметр'],
                                  data)

print(data.head())
print(data.info())
print(data.isnull().sum())
sns.set()
ax = sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

# распределение
# NOX
# preparation.nox_distribution(data)
# # CO
# preparation.co_distribution(data)
# # Tyx
# preparation.tyx_distribution(data)
# # Tcore
# preparation.tcore_distribution(data)


# preparation.correlation('Nox, мг/м3', 'Нагрузка, т/ч',data)
# preparation.correlation('Nox, мг/м3', 'Температура исходного воздуха, К',data)
#
# preparation.correlation('CO, мг/м3', 'Нагрузка, т/ч',data)
# preparation.correlation('CO, мг/м3', 'Температура исходного воздуха, К',data)
#
data = data.drop(['CO, мг/м3', 'Тух, К', 'Тядра, K'], axis=1)
x = data.drop('Nox, мг/м3', axis=1)
y = data['Nox, мг/м3']
# standard scaler
# x = preparation.standard_scaler(x)
# # minMax scaler
#x = preparation.min_max_scaler(x)
# robust scaler
# x = preprocessing.robust_scale(x)
# quantile
#x = preprocessing.quantile_transform(x)

#y = preparation.nox_to_classes(y)

# X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25)
# models.cat_boost_ROC(X_train, X_valid, y_train, y_valid)

# classification
# accuracy_cat = []
# accuracy_xg = []
# accuracy_rf = []
#
# for i in range(50):
#     try:
#         X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25)
#
#         accuracy_xg.append(models.xg_boost_classifer(X_train, X_valid, y_train, y_valid))
#
#         accuracy_cat.append(models.cat_boost_classifier(X_train, X_valid, y_train, y_valid))
#
#         accuracy_rf.append(models.random_forest_classifer(X_train, X_valid, y_train, y_valid))
#     except:
#         pass
#
# x_label = [i for i in range(len(accuracy_xg))]
#
# fig, ax = plt.subplots()
# cat_patch = mpatches.Patch(color='y', label='CatBoost')
# xg_patch = mpatches.Patch(color='b', label='XGBoost')
# rf_patch = mpatches.Patch(color='g', label='Random Forest')
# ax.legend(handles=[cat_patch, xg_patch, rf_patch])
#
# plt.title('Accuracy')
# plt.ylabel('accuracy score')
# plt.xlabel('experiment number')
# plt.grid(True)
#
# plt.plot(x_label, accuracy_cat, color='y')
# plt.plot(x_label, accuracy_xg, color='b')
# plt.plot(x_label, accuracy_rf, color='g')
#
# plt.show()
#
# X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25)
#
# # regression
#
# models.cat_boost_regression(X_train, X_valid, y_train, y_valid)
#
# models.xg_boost_regression(X_train, X_valid, y_train, y_valid)
#
# models.random_forest_regression(X_train, X_valid, y_train, y_valid)
