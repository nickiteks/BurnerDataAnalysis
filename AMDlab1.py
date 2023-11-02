import pandas as pd
from sklearn import preprocessing


data = pd.read_csv('report.csv', delimiter=';', decimal=',')
data40 = pd.read_csv('report40.csv', delimiter=';', decimal=',')

new_dataframe = pd.concat([data, data40], ignore_index=True)
new_dataframe.to_csv('classification.csv')

working_mode = []

for index in range(len(new_dataframe['Line Probe 8: Temperature (K)'])):

    working_mode.append([])

    if 1600 <= new_dataframe['Line Probe 8: Temperature (K)'][index] < 1700:
        working_mode[index].append(0)

    if 1700 <= new_dataframe['Line Probe 8: Temperature (K)'][index] < 1900:
        working_mode[index].append(1)

    if 1900 <= new_dataframe['Line Probe 8: Temperature (K)'][index] < 2100:
        working_mode[index].append(2)

    if 2100 <= new_dataframe['Line Probe 8: Temperature (K)'][index]:
        working_mode[index].append(3)

    if 0 <= new_dataframe['Line Probe 8: Mass Fraction of Nitrogen Oxide Emission'][index] < 1.13113043689738E-06:
        working_mode[index].append(0)

    if 1.13113043689738E-06 <= new_dataframe['Line Probe 8: Mass Fraction of Nitrogen Oxide Emission'][
        index] < 2.65859896775309E-06:
        working_mode[index].append(1)

    if 2.65859896775309E-06 <= new_dataframe['Line Probe 8: Mass Fraction of Nitrogen Oxide Emission'][
        index] < 0.0000577810121664682:
        working_mode[index].append(2)

    if 0.0000577810121664682 <= new_dataframe['Line Probe 8: Mass Fraction of Nitrogen Oxide Emission'][index]:
        working_mode[index].append(3)

working_mode_result = []

print(working_mode)

for model in working_mode:
    working_mode_result.append(min(model))

# for index in range(len(working_mode_result)):
#     if working_mode_result[index] == 0:
#         working_mode_result[index] = 'Great'
#
#     if working_mode_result[index] == 1:
#         working_mode_result[index] = 'Optimal'
#
#     if working_mode_result[index] == 2:
#         working_mode_result[index] = 'Satisfactory'
#
#     if working_mode_result[index] == 3:
#         working_mode_result[index] = 'Not a satisfactory'

new_dataframe['working_mode'] = working_mode_result
new_dataframe = new_dataframe[new_dataframe.working_mode != 0]
new_dataframe.to_csv('ddd.csv')


columns = ['working_mode']

dist = pd.DataFrame(new_dataframe['working_mode'].values,columns=['a'])

print(new_dataframe['Line Probe 8: Temperature (K)'])

column_to_normalize = new_dataframe['Line Probe 8: Temperature (K)']

# Примените normalize к столбцу
normalized_column = preprocessing.normalize([column_to_normalize])
new_dataframe['Line Probe 8: Temperature (K)'] = normalized_column[0]

print(new_dataframe['Line Probe 8: Temperature (K)'])
