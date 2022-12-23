import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np


class DataPreparation:

    def delete_columns(self, columns, data):
        for i in columns:
            data = data.drop(i, axis=1)

        return data

    def nox_distribution(self, data):
        NoX_count = [0, 0, 0]
        NoX_groups = ['<125', '125-290', '`>290']

        for i in range(len(data['Nox, мг/м3'])):
            if data['Nox, мг/м3'][i] < 125:
                NoX_count[0] = NoX_count[0] + 1
            if 125 <= data['Nox, мг/м3'][i] <= 290:
                NoX_count[1] = NoX_count[1] + 1
            if data['Nox, мг/м3'][i] > 290:
                NoX_count[2] = NoX_count[2] + 1

        plt.bar(range(len(NoX_groups)), NoX_count)
        plt.title('Распределение NoX')
        plt.ylabel('Количество значений')
        plt.xticks(range(len(NoX_groups)), NoX_groups)
        plt.show()

    def co_distribution(self, data):
        CO_count = [0, 0, 0]
        CO_groups = ['<100', '100-300', '>300']

        for i in range(len(data['CO, мг/м3'])):
            if data['CO, мг/м3'][i] < 100:
                CO_count[0] = CO_count[0] + 1
            if 100 <= data['CO, мг/м3'][i] <= 300:
                CO_count[1] = CO_count[1] + 1
            if data['CO, мг/м3'][i] > 300:
                CO_count[2] = CO_count[2] + 1

        plt.bar(range(len(CO_groups)), CO_count)
        plt.title('Распределение CO')
        plt.ylabel('Количество значений')
        plt.xticks(range(len(CO_groups)), CO_groups)
        plt.show()

    def tyx_distribution(self, data):
        Tyx_count = [0, 0, 0]
        Tyx_groups = ['343-381', '381-393', '>393']

        for i in range(len(data['Тух, К'])):
            if 343 <= data['Тух, К'][i] < 381:
                Tyx_count[0] = Tyx_count[0] + 1
            if 381 <= data['Тух, К'][i] <= 393:
                Tyx_count[1] = Tyx_count[1] + 1
            if data['Тух, К'][i] > 393:
                Tyx_count[2] = Tyx_count[2] + 1

        plt.bar(range(len(Tyx_groups)), Tyx_count)
        plt.title('Распределение Tyx')
        plt.ylabel('Количество значений')
        plt.xticks(range(len(Tyx_groups)), Tyx_groups)
        plt.show()

    def tcore_distribution(self, data):
        Tcore_count = [0, 0, 0]
        Tcore_groups = ['1600-1700', '1700-2100', '>2100']

        for i in range(len(data['Тядра, K'])):
            if 1600 <= data['Тядра, K'][i] < 1700:
                Tcore_count[0] = Tcore_count[0] + 1
            if 1700 <= data['Тядра, K'][i] <= 2100:
                Tcore_count[1] = Tcore_count[1] + 1
            if data['Тядра, K'][i] > 2100:
                Tcore_count[2] = Tcore_count[2] + 1

        plt.bar(range(len(Tcore_groups)), Tcore_count)
        plt.title('Распределение Tядра')
        plt.ylabel('Количество значений')
        plt.xticks(range(len(Tcore_groups)), Tcore_groups)
        plt.show()

    def correlation(self, param1, param2, data):
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

    def standard_scaler(self, x):
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
        return x

    def min_max_scaler(self, x):
        x = np.array(x)
        min_max_scaler = preprocessing.MinMaxScaler()
        x = min_max_scaler.fit_transform(x)
        return x

    def nox_to_classes(self, y):
        for i in range(len(y)):
            if y[i] < 125:
                y[i] = int(0)
            if 290 >= y[i] >= 125:
                y[i] = int(1)
            if y[i] > 290:
                y[i] = int(2)

        return y
