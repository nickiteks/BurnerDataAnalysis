from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class Models:

    def cat_boost_classifier(self,X_train, X_valid, y_train, y_valid):
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
        bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='multi:softprob')
        bst.fit(X_train, y_train)
        preds = bst.predict(X_valid)
        print(accuracy_score(y_valid, preds))
        return accuracy_score(y_valid, preds)

    def random_forest_classifer(self,X_train, X_valid, y_train, y_valid):
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_valid)
        print(accuracy_score(y_valid, pred))
        return accuracy_score(y_valid, pred)