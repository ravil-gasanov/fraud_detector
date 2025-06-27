from sklearn.base import BaseEstimator, TransformerMixin


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["V4xV11"] = X["V4"] * X["V11"]

        X["V7_is_negative"] = (X["V7"] < 0).astype("Int64")

        return X
