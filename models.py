from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from catboost import Pool
from catboost import CatBoostRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import shap


class Models:

    def cat_boost_classifier(self, X_train, X_valid, y_train, y_valid):
        """
        Классификация с помошью CatBOOST
        """
        model = CatBoostClassifier(iterations=1500,
                                   learning_rate=0.1,
                                   depth=2,
                                   loss_function='MultiClass')

        model.fit(X_train, y_train)

        preds_class = model.predict(X_valid)
        preds_proba = model.predict_proba(X_valid)
        print(accuracy_score(y_valid, preds_class))
        return accuracy_score(y_valid, preds_class)

    def xg_boost_classifer(self, X_train, X_valid, y_train, y_valid):
        """
        Классификация с помощью XgBOOST
        """
        bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='multi:softprob')
        bst.fit(X_train, y_train)
        preds = bst.predict(X_valid)
        print(accuracy_score(y_valid, preds))
        return accuracy_score(y_valid, preds)

    def random_forest_classifer(self, X_train, X_valid, y_train, y_valid):
        """
        Классификация с помощью случайного леса
        """
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_valid)
        print(accuracy_score(y_valid, pred))
        return accuracy_score(y_valid, pred)

    def cat_boost_regression(self, X_train, X_test, y_train, y_test):
        """
        Регрессия с помощью CatBOOST
        """
        train_dataset = Pool(X_train, y_train)
        test_dataset = Pool(X_test, y_test)

        model = CatBoostRegressor(loss_function='RMSE')

        grid = {'iterations': [100,200],
                'learning_rate': [0.03, 0.1,0.01,0.05],
                'depth': [10, 20,50,100, 500]}
        model.grid_search(grid, train_dataset)

        y_test = np.array(y_test)

        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        rmse = (np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)

        # график важности параметров

        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        fig = plt.figure(figsize=(12, 6))
        plt.bar(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.xticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx],rotation=20,)
        plt.title('Feature Importance')
        plt.gca().invert_xaxis()
        plt.show()

        print('--CatBoost--')
        print("MSE : % f" % (mse))
        print("RMSE : % f" % (rmse))
        print("R2 : % f" % (r2))

        fig, ax = plt.subplots()
        ax.plot(pred, color='g')
        ax.plot(y_test, color='r')

        ax.set(
            title='prev vs test')
        ax.grid()

        plt.show()

        print(pred)
        print(y_test)

    def xg_boost_regression(self, X_train, X_valid, y_train, y_valid):
        """
        Регрессия с помощью XgBOOST
        """
        xgb_r = XGBRegressor(objective='reg:squarederror',
                             n_estimators=1000, max_depth=25, eta=0.1, subsample=0.7, colsample_bytree=0.8)

        # Fitting the model
        xgb_r.fit(X_train, y_train)

        # Predict the model
        pred = xgb_r.predict(X_valid)

        # RMSE Computation
        mse = mean_squared_error(y_valid, pred)
        rmse = np.sqrt(mean_squared_error(y_valid, pred))
        r2 = r2_score(y_valid, pred)

        # график важности параметров

        feature_importance = xgb_r.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        fig = plt.figure(figsize=(12, 6))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(X_valid.columns)[sorted_idx])
        plt.title('Feature Importance')
        plt.show()

        print('--XGBoost--')
        print("MSE : % f" % (mse))
        print("RMSE : % f" % (rmse))
        print("R2 : % f" % (r2))

    def random_forest_regression(self, X_train, X_valid, y_train, y_valid):
        """
        Регрессия с помощью случайного леса
        """
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        regressor.fit(X_train, y_train)
        pred = regressor.predict(X_valid)

        mse = mean_squared_error(y_valid, pred)
        rmse = np.sqrt(mean_squared_error(y_valid, pred))
        r2 = r2_score(y_valid, pred)

        # график важности параметров

        feature_importance = regressor.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        fig = plt.figure(figsize=(12, 6))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(X_valid.columns)[sorted_idx])
        plt.title('Feature Importance')
        plt.show()

        print('--RF--')
        print("MSE : % f" % (mse))
        print("RMSE : % f" % (rmse))
        print("R2 : % f" % (r2))

    def cat_boost_ROC(self, X_train, X_valid, y_train, y_valid):
        model = CatBoostClassifier(iterations=1500,
                                   learning_rate=0.1,
                                   depth=2,
                                   loss_function='MultiClass')
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_valid)

        y_valid = label_binarize(y_valid, classes=[0, 1, 2])

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_valid[:, i], pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i in range(3):
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def xg_boost_ROC(self, X_train, X_valid, y_train, y_valid):
        model = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='multi:softprob')
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_valid)

        y_valid = label_binarize(y_valid, classes=[0, 1, 2])

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_valid[:, i], pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i in range(3):
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def random_forest_ROC(self, X_train, X_valid, y_train, y_valid):
        model = RandomForestClassifier(max_depth=2, random_state=0)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_valid)

        y_valid = label_binarize(y_valid, classes=[0, 1, 2])

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_valid[:, i], pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i in range(3):
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
