import pandas as pd
import numpy as np
import catboost as cb
import optuna

from dataPreparation import DataPreparation
from models import Models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

data = data.drop(['CO, мг/м3', 'Тух, К', 'Тядра, K'], axis=1)
x = data.drop('Nox, мг/м3', axis=1)
y = data['Nox, мг/м3']

y = preparation.nox_to_classes(y)

X = np.array(x)
y = np.array(y)


def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.3)

    param = {
        'iterations': 1500,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),
        'depth': 2,
        'loss_function': 'MultiClass'
    }

    model = cb.CatBoostClassifier(**param)

    model.fit(train_x, train_y)

    preds = model.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(valid_y, pred_labels)
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))