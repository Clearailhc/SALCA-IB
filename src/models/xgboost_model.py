import xgboost as xgb

class DynamicXGBoost:
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100):
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)