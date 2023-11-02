import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv('distribution.csv', delimiter=';', decimal='.')

print(data.head())
print(data.info())

data = data.sort_values(by=['Line Probe 43: Direction [-1,0,0] (m)'])

fig, ax = plt.subplots()
ax.plot(data['Line Probe 43: Direction [-1,0,0] (m)'],data['Line Probe 43: Mass Fraction of CH4'],  label='CH4',
        color='g',linestyle = 'dashed')

data = data.sort_values(by=['Line Probe 43: Direction [-1,0,0] (m).1'])
ax.plot( data['Line Probe 43: Direction [-1,0,0] (m).1'],data['Line Probe 43: Mass Fraction of CO'], label='CO',
        color='r',linestyle = 'dashed')

data = data.sort_values(by=['Line Probe 43: Direction [-1,0,0] (m).2'])
ax.plot(data['Line Probe 43: Direction [-1,0,0] (m).2'],data['Line Probe 43: Mass Fraction of CO2'],  label='CO2',
        color='black',linestyle = 'dashed')

data = data.sort_values(by=['Line Probe 43: Direction [-1,0,0] (m).3'])
ax.plot(data['Line Probe 43: Direction [-1,0,0] (m).3'],data['Line Probe 43: Mass Fraction of H2O'],  label='H2O',
        color='blue',linestyle = 'dashed')

data = data.sort_values(by=['Line Probe 43: Direction [-1,0,0] (m).4'])
ax.plot(data['Line Probe 43: Direction [-1,0,0] (m).4'],data['Line Probe 43: Mass Fraction of O2'],  label='O2',
        color='orange',linestyle = 'dashed')


ax.set(xlabel='Direction, m', ylabel='Mass Fraction',
       title='Burner process')
ax.grid()
ax.legend()

plt.show()
