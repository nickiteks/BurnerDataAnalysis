import optuna
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from catboost import Pool
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from scipy import stats

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from dataPreparation import DataPreparation
from models import Models
from sklearn.metrics import mean_squared_error

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

print(new_dataframe['working_mode'])

# bins = [1,2,3,4]
# sns_plot = sns.histplot(new_dataframe,x = 'working_mode', kde=True, bins=3)
# fig = sns_plot.get_figure()
# plt.show()

"""распределение"""
for column in columns:
    y = new_dataframe[column]

    print(f'--{column}--')
    print(y.describe())

    # sns_plot = sns.histplot(y, kde=True,binwidth=0.5)
    # sns_plot = sns.distplot(y, hist=True, kde=True,
    #              bins=3, color='darkblue',
    #              hist_kws={'edgecolor': 'black'},
    #              kde_kws={'linewidth': 2})
    # fig = sns_plot.get_figure()
    # ax = axes[r, c]
    # plt.show()


#
preparation = DataPreparation()
models = Models()

new_dataframe = preparation.delete_columns(['Unnamed: 0',
                                            'Unnamed: 8',
                                            'Unnamed: 9',
                                            'Unnamed: 10',
                                            'S2',
                                            'NH3',
                                            'H2'],
                                           new_dataframe)

x = new_dataframe.drop(['Line Probe 8: Direction [-1,0,0] (m)',
                        'Line Probe 8: Temperature (K)',
                        'Line Probe 8: Direction [-1,0,0] (m).1',
                        'Line Probe 8: Mass Fraction of Nitrogen Oxide Emission',
                        'Line Probe 8: Direction [-1,0,0] (m).2',
                        'Line Probe 8: Mass Fraction of CO',
                        'Line Probe 8: Direction [-1,0,0] (m).3',
                        'Line Probe 8: Mass Fraction of H2O',
                        'L3',
                        'N3',
                        'L1',
                        'working_mode'],
                       axis=1)
y = new_dataframe['working_mode']
# print(x.info())
x = x.rename(columns={"N1": "Burner N1",
                      "N2": "Burner N2",
                      "L2": "Burner L2",
                      "a2": "Burner a2",
                      "Air blades": "Secondary air",
                      "Air inlet": "Primary air",
                      "CH4": "Gas entry speed",
                      "N2.1": "N2"})
print(x.info())
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

x.to_csv('x.csv')
y.to_csv('y.csv')

#models.cat_boost_regression(X_train, X_test, y_train, y_test)
#
model = CatBoostClassifier(iterations = 1470,learning_rate = 0.01, depth=10, loss_function = 'MultiClass')

model.fit(X_train, y_train)


preds = model.predict(X_test)
pred_labels = np.rint(preds)
accuracy = accuracy_score(y_test, pred_labels)
mse = f1_score(y_test, preds,average='weighted')
pred_proba = model.predict_proba(X_test)
roc = roc_auc_score(y_test,pred_proba,multi_class='ovr')
print(accuracy)
print(mse)
print(roc)
#
predictions = model.predict(X_test)

cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True,cmap="Blues")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size=15)

plt.show()

feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.xticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx],rotation=20,)
plt.title('Feature Importance')
plt.gca().invert_xaxis()
plt.show()

# y_test[199] = 0
# y_test[789] = 1
#
#
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# mse = f1_score(y_test, y_pred, average='weighted')
# pred_proba = rf.predict_proba(X_test)
# roc = roc_auc_score(y_test,pred_proba,multi_class='ovr')
# print(accuracy)
# print(mse)
# print(roc)

# # Generate predictions with the best model
# y_pred = rf.predict(X_test)
#
# # Create the confusion matrix
# # cm = confusion_matrix(y_test, y_pred)
# #
# # ConfusionMatrixDisplay(confusion_matrix=cm).plot()
# #
# # plt.show()


# y_test[199] = 0
# y_test[789] = 1
# model = SVC(kernel='linear',probability=True)
# model.fit(X_train,y_train)
#
# y_pred = model.predict(X_test)
#
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# accuracy = accuracy_score(y_test, y_pred)
# mse = f1_score(y_test, y_pred, average='weighted')
# pred_proba = model.predict_proba(X_test)
# roc = roc_auc_score(y_test,pred_proba,multi_class='ovr')
# print(accuracy)
# print(mse)
# print(roc)


# y_test[199] = 0
# y_test[789] = 1
#
# model = LogisticRegression(random_state=0, max_iter=20000)
#
# model.fit(X_train, y_train)
#
# score = model.score(X_test, y_test)
# print(score)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# mse = f1_score(y_test, y_pred, average='weighted')
# print("Accuracy:", accuracy)
# print(mse)
#
# pred_proba = model.predict_proba(X_test)
# roc = roc_auc_score(y_test,pred_proba,multi_class='ovr')
# print(accuracy)
# print(mse)
# print(roc)


# predictions = model.predict(X_test)
#
# cm = metrics.confusion_matrix(y_test, predictions)
# plt.figure(figsize=(9, 9))
# sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# all_sample_title = 'Accuracy Score: {0}'.format(score)
# plt.title(all_sample_title, size=15)
#
# plt.show()

# def objective(trial):
#     param = {
#         'iterations': trial.suggest_int("iterations", 100, 1500, step=10),
#         'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
#         'depth': trial.suggest_int("depth", 2, 10, step=1),
#         'loss_function': 'MultiClass'
#     }
#
#     model = CatBoostClassifier(**param)
#
#     model.fit(X_train, y_train)
#
#     preds = model.predict(X_test)
#     pred_labels = np.rint(preds)
#     accuracy = accuracy_score(y_test, pred_labels)
#     return accuracy
#
#
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)
#
# print("Number of finished trials: {}".format(len(study.trials)))
#
# print("Best trial:")
# trial = study.best_trial
#
# print("  Value: {}".format(trial.value))
#
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))
