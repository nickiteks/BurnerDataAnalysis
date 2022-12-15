import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('book.csv', delimiter=';')

# очистка данных

data = data.drop('O2, %', axis=1)
data = data.drop('Тип амбразуры', axis=1)
data = data.drop('Модель горения', axis=1)
data = data.drop('Модель турбулентности', axis=1)

print(data.head())
print(data.info())
print(data.isnull().sum())

# распределение
#NOX
NoX_count = [0, 0, 0]
NoX_groups =['<125','125-290','`>290']

for i in range(len(data['Nox, мг/м3'])):
    if float(data['Nox, мг/м3'][i].replace(',','.')) < 125:
        NoX_count[0] = NoX_count[0] + 1
    if 125 <= float(data['Nox, мг/м3'][i].replace(',','.')) <= 290:
        NoX_count[1] = NoX_count[1] + 1
    if float(data['Nox, мг/м3'][i].replace(',','.')) > 290:
        NoX_count[2] = NoX_count[2] + 1


plt.bar(range(len(NoX_groups)),NoX_count)
plt.title('Распределение NoX')
plt.ylabel('Количество значений')
plt.xticks(range(len(NoX_groups)),NoX_groups)
plt.show()

#CO
CO_count = [0, 0, 0]
CO_groups =['<100','100-300','>300']

for i in range(len(data['CO, мг/м3'])):
    if float(data['CO, мг/м3'][i].replace(',','.')) < 100:
        CO_count[0] = CO_count[0] + 1
    if 100 <= float(data['CO, мг/м3'][i].replace(',','.')) <= 300:
        CO_count[1] = CO_count[1] + 1
    if float(data['CO, мг/м3'][i].replace(',','.')) > 300:
        CO_count[2] = CO_count[2] + 1

plt.bar(range(len(CO_groups)),CO_count)
plt.title('Распределение CO')
plt.ylabel('Количество значений')
plt.xticks(range(len(CO_groups)),CO_groups)
plt.show()

#Tyx
Tyx_count = [0, 0, 0]
Tyx_groups =['343-381','381-393','>393']

for i in range(len(data['Тух, К'])):
    if 343 <= float(data['Тух, К'][i].replace(',','.')) < 381:
        Tyx_count[0] = Tyx_count[0] + 1
    if 381 <= float(data['Тух, К'][i].replace(',','.')) <= 393:
        Tyx_count[1] = Tyx_count[1] + 1
    if float(data['Тух, К'][i].replace(',','.')) > 393:
        Tyx_count[2] = Tyx_count[2] + 1

plt.bar(range(len(Tyx_groups)),Tyx_count)
plt.title('Распределение Tyx')
plt.ylabel('Количество значений')
plt.xticks(range(len(Tyx_groups)),Tyx_groups)
plt.show()

#Tcore
Tcore_count = [0, 0, 0]
Tcore_groups =['1600-1700','1700-2100','>2100']

for i in range(len(data['Тядра, K'])):
    if 1600 <= float(data['Тядра, K'][i].replace(',','.')) < 1700:
        Tcore_count[0] = Tcore_count[0] + 1
    if 1700 <= float(data['Тядра, K'][i].replace(',','.')) <= 2100:
        Tcore_count[1] = Tcore_count[1] + 1
    if float(data['Тядра, K'][i].replace(',','.')) > 2100:
        Tcore_count[2] = Tcore_count[2] + 1

plt.bar(range(len(Tcore_groups)),Tcore_count)
plt.title('Распределение Tядра')
plt.ylabel('Количество значений')
plt.xticks(range(len(Tcore_groups)),Tcore_groups)
plt.show()