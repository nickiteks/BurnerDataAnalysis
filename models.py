from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from catboost import Pool
from catboost import CatBoostRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class Models:

    def cat_boost_classifier(self,X_train, X_valid, y_train, y_valid):
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

    def xg_boost_classifer(self,X_train, X_valid, y_train, y_valid):
        """
        Классификация с помощью XgBOOST
        """
        bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='multi:softprob')
        bst.fit(X_train, y_train)
        preds = bst.predict(X_valid)
        print(accuracy_score(y_valid, preds))
        return accuracy_score(y_valid, preds)

    def random_forest_classifer(self,X_train, X_valid, y_train, y_valid):
        """
        Классификация с помощью случайного леса
        """
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_valid)
        print(accuracy_score(y_valid, pred))
        return accuracy_score(y_valid, pred)

    def cat_boost_regression(self,X_train, X_valid, y_train, y_valid):
        """
        Регрессия с помощью CatBOOST
        """
        train_dataset = Pool(X_train, y_train)
        test_dataset = Pool(X_valid, y_valid)

        model = CatBoostRegressor(loss_function='RMSE')

        grid = {'iterations': [100, 150, 200],
                'learning_rate': [0.03, 0.1],
                'depth': [2, 4, 6, 8],
                'l2_leaf_reg': [0.2, 0.5, 1, 3]}
        model.grid_search(grid, train_dataset)

        pred = model.predict(X_valid)
        rmse = (np.sqrt(mean_squared_error(y_valid, pred)))
        r2 = r2_score(y_valid, pred)

        print("RMSE : % f" % (rmse))
        print(r2)

    def xg_boost_regression(self,X_train, X_valid, y_train, y_valid):
        """
        Регрессия с помощью XgBOOST
        """
        xgb_r = XGBRegressor(objective='reg:squarederror',
                             n_estimators=10, seed=123)

        # Fitting the model
        xgb_r.fit(X_train, y_train)

        # Predict the model
        pred = xgb_r.predict(X_valid)

        # RMSE Computation
        rmse = np.sqrt(mean_squared_error(y_valid, pred))
        print("RMSE : % f" % (rmse))

    def random_forest_regression(self,X_train, X_valid, y_train, y_valid):
        """
        Регрессия с помощью случайного леса
        """
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        regressor.fit(X_train, y_train)
        pred = regressor.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, pred))
        print("RMSE : % f" % (rmse))