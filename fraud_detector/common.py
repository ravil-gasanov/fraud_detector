import pandas as pd
from prefect import task

from fraud_detector.config import FEATURE_COLUMNS, TARGET_COLUMN


@task
def load_X_y(data_path):
    data = pd.read_csv(data_path)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    return X, y
