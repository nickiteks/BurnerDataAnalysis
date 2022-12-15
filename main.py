import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


correlation('Nox, мг/м3', 'Нагрузка, т/ч')
correlation('Nox, мг/м3', 'Температура исходного воздуха, К')

correlation('CO, мг/м3', 'Нагрузка, т/ч')
correlation('CO, мг/м3', 'Температура исходного воздуха, К')
