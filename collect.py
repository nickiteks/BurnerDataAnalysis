import os
import openpyxl
import re
import pandas as pd
import numpy as np


class ReportCollector:

    def __init__(self, column, exclel_path, csv_path):
        self.COLUMN = column
        self.HEADER = True
        self.excel_path = exclel_path
        self.csv_path = csv_path

    def collect(self):
        wb_obj = openpyxl.load_workbook(self.excel_path)
        sheet_obj = wb_obj.active

        oldpwd = os.getcwd()

        os.chdir(self.csv_path)

        files = os.listdir()

        if 'report.csv' in files:
            os.remove('report.csv')
            files = os.listdir()

        files = sorted(files, key=lambda s: int(re.search(r'\d+', s).group()))

        for i in files:
            row = re.findall(r'\d+', i)
            row = int(row[0])

            excel_data = []
            for j in range(21):
                excel_data.append(sheet_obj.cell(row=row, column=self.COLUMN + j).value)

            data = pd.read_csv(i, sep=';', decimal=',', encoding="ISO-8859-1")

            list_of_max = []
            list_of_collums = data.columns.values.tolist()

            for i in range(len(list_of_collums)):
                if i % 2 != 0:
                    max_value = data[list_of_collums[i]].max()
                    max_value_index = data[data[list_of_collums[i]] == max_value].index.to_list()
                    direction = data[list_of_collums[i - 1]][max_value_index[0]]
                    list_of_max.append(direction)
                    list_of_max.append(max_value)

            header = ['N1', 'L1', 'N2', 'L2', 'a2', 'N3', 'L3', '', '', '', 'Air blades', 'Air inlet', 'CH4',
                      'Temperature', 'S2', 'NH3', 'H2', 'O2', 'N2', 'CO2', 'H20',
                      'Line Probe 8: Direction [-1,0,0] (m)',
                      'Line Probe 8: Temperature (K)',
                      'Line Probe 8: Direction [-1,0,0] (m)',
                      'Line Probe 8: Mass Fraction of Nitrogen Oxide Emission',
                      'Line Probe 8: Direction [-1,0,0] (m)',
                      'Line Probe 8: Mass Fraction of CO',
                      'Line Probe 8: Direction [-1,0,0] (m)',
                      'Line Probe 8: Mass Fraction of H2O']

            csv_data = [[i] for i in excel_data + list_of_max]

            for i in range(len(csv_data)):
                for j in range(len(csv_data[i])):
                    csv_data[i][j] = str(csv_data[i][j]).replace('.', ',')

            df2 = pd.DataFrame(np.array(csv_data).transpose(),
                               columns=header, index=[row])

            df2.to_csv('report.csv', sep=';', mode='a', decimal=',', float_format='%.3f', header=self.HEADER)

            self.HEADER = False

        os.chdir(oldpwd)


collector = ReportCollector(6, 'C:/Users/NULS/Desktop/StarAutomation/ExcelData/Ex.xlsx', 'csv')
collector.collect()
