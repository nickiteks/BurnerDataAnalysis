import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv('book.csv', delimiter=';', decimal=',')

# очистка данных

data = data.drop('O2, %', axis=1)
data = data.drop('Тип амбразуры', axis=1)
data = data.drop('Модель горения', axis=1)
data = data.drop('Модель турбулентности', axis=1)

print(data.head())
print(data.info())
print(data.isnull().sum())


# распределение
# NOX
# NoX_count = [0, 0, 0]
# NoX_groups =['<125','125-290','`>290']
#
# for i in range(len(data['Nox, мг/м3'])):
#     if data['Nox, мг/м3'][i] < 125:
#         NoX_count[0] = NoX_count[0] + 1
#     if 125 <= data['Nox, мг/м3'][i] <= 290:
#         NoX_count[1] = NoX_count[1] + 1
#     if data['Nox, мг/м3'][i] > 290:
#         NoX_count[2] = NoX_count[2] + 1
#
#
# plt.bar(range(len(NoX_groups)),NoX_count)
# plt.title('Распределение NoX')
# plt.ylabel('Количество значений')
# plt.xticks(range(len(NoX_groups)),NoX_groups)
# plt.show()
# #
# #CO
# CO_count = [0, 0, 0]
# CO_groups =['<100','100-300','>300']
#
# for i in range(len(data['CO, мг/м3'])):
#     if data['CO, мг/м3'][i] < 100:
#         CO_count[0] = CO_count[0] + 1
#     if 100 <= data['CO, мг/м3'][i] <= 300:
#         CO_count[1] = CO_count[1] + 1
#     if data['CO, мг/м3'][i] > 300:
#         CO_count[2] = CO_count[2] + 1
#
# plt.bar(range(len(CO_groups)),CO_count)
# plt.title('Распределение CO')
# plt.ylabel('Количество значений')
# plt.xticks(range(len(CO_groups)),CO_groups)
# plt.show()
#
# #Tyx
# Tyx_count = [0, 0, 0]
# Tyx_groups =['343-381','381-393','>393']
#
# for i in range(len(data['Тух, К'])):
#     if 343 <= data['Тух, К'][i] < 381:
#         Tyx_count[0] = Tyx_count[0] + 1
#     if 381 <= data['Тух, К'][i] <= 393:
#         Tyx_count[1] = Tyx_count[1] + 1
#     if data['Тух, К'][i] > 393:
#         Tyx_count[2] = Tyx_count[2] + 1
#
# plt.bar(range(len(Tyx_groups)),Tyx_count)
# plt.title('Распределение Tyx')
# plt.ylabel('Количество значений')
# plt.xticks(range(len(Tyx_groups)),Tyx_groups)
# plt.show()
#
# #Tcore
# Tcore_count = [0, 0, 0]
# Tcore_groups =['1600-1700','1700-2100','>2100']
#
# for i in range(len(data['Тядра, K'])):
#     if 1600 <= data['Тядра, K'][i] < 1700:
#         Tcore_count[0] = Tcore_count[0] + 1
#     if 1700 <= data['Тядра, K'][i] <= 2100:
#         Tcore_count[1] = Tcore_count[1] + 1
#     if data['Тядра, K'][i] > 2100:
#         Tcore_count[2] = Tcore_count[2] + 1
#
# plt.bar(range(len(Tcore_groups)),Tcore_count)
# plt.title('Распределение Tядра')
# plt.ylabel('Количество значений')
# plt.xticks(range(len(Tcore_groups)),Tcore_groups)
# plt.show()

def correlation(param1, param2):
    """
    Кореляция зависимоти одного параметра от другого
    :param param1: первый параметр кореляции
    :param param2: второй параметр кореляции
    :return: график кореляции
    """
    xs = data[param1]
    ys = data[param2]
    pd.DataFrame(np.array([xs, ys]).T).plot.scatter(0, 1, grid=True)
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.show()


# correlation('Nox, мг/м3', 'Нагрузка, т/ч')
# correlation('Nox, мг/м3', 'Температура исходного воздуха, К')
#
# correlation('CO, мг/м3', 'Нагрузка, т/ч')
# correlation('CO, мг/м3', 'Температура исходного воздуха, К')

data = data.drop(['CO, мг/м3', 'Тух, К', 'Тядра, K'], axis=1)
x = data.drop('Nox, мг/м3', axis=1)
y = data['Nox, мг/м3']
# scaler = preprocessing.StandardScaler().fit(x)
# x = scaler.transform(x)
print(x)

#y.loc[y['Nox, мг/м3'] < 125,'Nox, мг/м3'] = 0


for i in range(len(y)):
    if y[i] < 125:
        y[i] = int(0)
    if 290 >= y[i] >= 125:
        y[i] = int(1)
    if y[i] > 290:
        y[i] = int(2)

X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25,random_state=42)
# le = LabelEncoder()
# y_train = le.fit_transform(y_train)
# y_valid = le.fit_transform(y_valid)
#
model = CatBoostClassifier(iterations=1500,
                           learning_rate=0.1,
                           depth=2,
                           loss_function='MultiClass')

model.fit(X_train, y_train)

preds_class = model.predict(X_valid)
preds_proba = model.predict_proba(X_valid)
print("class = ", preds_class)
print(y_valid)
print("proba = ", preds_proba)
print(accuracy_score(y_valid,preds_class))
# #
# bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')#change
# # fit model
# bst.fit(X_train, y_train)
# # make predictions
# preds = bst.predict(X_valid)
#
# print("class = ", preds)
# print(y_valid)
# #
# print(accuracy_score(y_valid,preds))
# #
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(X_train, y_train)
#
# pred = clf.predict(X_valid)
# print(accuracy_score(y_valid,pred))

# train_dataset = Pool(X_train, y_train)
# test_dataset = Pool(X_valid, y_valid)
#
# model = CatBoostRegressor(loss_function='RMSE')
#
# grid = {'iterations': [100, 150, 200],
#         'learning_rate': [0.03, 0.1],
#         'depth': [2, 4, 6, 8],
#         'l2_leaf_reg': [0.2, 0.5, 1, 3]}
# model.grid_search(grid, train_dataset)
#
# pred = model.predict(X_valid)
# rmse = (np.sqrt(mean_squared_error(y_valid, pred)))
# r2 = r2_score(y_valid, pred)
#
# print(rmse)
# print(r2)

