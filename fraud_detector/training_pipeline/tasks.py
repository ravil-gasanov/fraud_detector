import pandas as pd
from prefect import task
from sklearn.metrics import f1_score

from fraud_detector.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


@task
def load_X_y(data_path):
    data = pd.read_csv(data_path)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    return X, y


@task
def load_model_from_model_registry():
    pass


@task
def train_model(model, X, y):
    model.fit(X, y)

    return model


@task
def eval_model_on_test(model, test_X, test_y):
    predictions = model.predict(test_X)

    metric = f1_score(
        y_true=test_y,
        y_pred=predictions,
    )

    return metric


@task
def register_model(model, model_name):
    pass
