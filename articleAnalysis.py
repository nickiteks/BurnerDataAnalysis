import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import torch.nn as nn

from dataPreparation import DataPreparation
from models import Models

data = pd.read_csv('reportArticle.csv', delimiter=';', decimal=',')

print(data.head())
print(data.info())

preparation = DataPreparation()
models = Models()

data = preparation.delete_columns(['Unnamed: 0',
                                   'Unnamed: 8',
                                   'Unnamed: 9',
                                   'Unnamed: 10',
                                   'S2',
                                   'NH3',
                                   'H2'],
                                  data)

print(data.info())

columns = ['Line Probe 8: Temperature (K)',
           'Line Probe 8: Mass Fraction of Nitrogen Oxide Emission',
           'Line Probe 8: Mass Fraction of CO',
           'Line Probe 8: Mass Fraction of H2O']
#
# for column in columns:
#     y = data[column]
#
#     print(f'--{column}--')
#     print(y.describe())
#
#     sns_plot = sns.histplot(y)
#     fig = sns_plot.get_figure()
#     plt.show()

x = data.drop(['Line Probe 8: Direction [-1,0,0] (m)',
               'Line Probe 8: Temperature (K)',
               'Line Probe 8: Direction [-1,0,0] (m).1',
               'Line Probe 8: Mass Fraction of Nitrogen Oxide Emission',
               'Line Probe 8: Direction [-1,0,0] (m).2',
               'Line Probe 8: Mass Fraction of CO',
               'Line Probe 8: Direction [-1,0,0] (m).3',
               'Line Probe 8: Mass Fraction of H2O',
               'L3',
               'N3',
               'L1']
              , axis=1)

y = data['Line Probe 8: Mass Fraction of Nitrogen Oxide Emission']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# models.cat_boost_regression(X_train, X_test, y_train, y_test)


X_train = torch.tensor(X_train.values)
y_train = torch.tensor(y_train.values).unsqueeze_(1)
X_test = torch.tensor(X_test.values)
y_test = torch.tensor(y_test.values).unsqueeze_(1)


class optimalNet(nn.Module):
    def __init__(self, n_hid_n):
        super(optimalNet, self).__init__()
        self.fc1 = nn.Linear(12, n_hid_n)
        self.act1 = nn.ReLU()
        self.fc3 = nn.Linear(n_hid_n, 1)
        self.act3 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc3(x)
        x = self.act3(x)
        return x


optimalNet = optimalNet(48).double()

loss = nn.MSELoss()
optimiser = torch.optim.SGD(optimalNet.parameters(), lr=0.01)

epoch = 100

for e in range(epoch):
    optimiser.zero_grad()
    y_pred = optimalNet.forward(X_train.double())
    loss_val = loss(y_pred, y_train)

    print(loss_val)

    loss_val.backward()
    optimiser.step()

print('---Test---')
pred = optimalNet.forward(X_test.double())
pred_loss = loss(pred, y_test)

print(pred_loss)
