from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from loguru import logger
import pandas as pd
from prefect import task

from fraud_detector.common import load_X_y
from fraud_detector.config import FEATURE_COLUMNS, PREDICTIONS_PATH, TEST_PATH, TRAIN_PATH


@task
def load_reference_data():
    data = pd.read_csv(TRAIN_PATH)
    return data


@task
def load_new_data():
    data = pd.read_csv(TEST_PATH)
    return data


@task
def load_predictions_on_new_data():
    predictions = pd.read_csv(PREDICTIONS_PATH)
    return predictions


@task
def calculate_metrics(reference_data, new_data, predictions):
    schema = DataDefinition(numerical_columns=FEATURE_COLUMNS)

    reference_dataset = Dataset.from_pandas(reference_data, data_definition=schema)
    new_dataset = Dataset.from_pandas(new_data, data_definition=schema)

    report = Report([DataDriftPreset()])

    metrics = report.run(new_dataset, reference_dataset)

    return metrics


@task
def save_metrics(metrics):
    metrics.save_html("reports/data_drift_report.html")
