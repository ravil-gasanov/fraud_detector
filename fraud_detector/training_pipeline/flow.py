import os

from prefect import flow

from fraud_detector.training_pipeline.tasks import (
    eval_model_on_test,
    load_model_from_model_registry,
    load_test_X_y,
    load_train_X_y,
    register_model,
    train_model,
)

METRIC_THRESHOLD = os.environ["METRIC_THRESHOLD"]


@flow
def train_flow():
    X, y = load_train_X_y()
    test_X, test_y = load_test_X_y()

    model = load_model_from_model_registry()

    model = train_model(model=model, X=X, y=y)

    metric = eval_model_on_test(model=model, test_X=test_X, test_y=test_y)

    if metric <= METRIC_THRESHOLD:
        register_model(model=model, model_name="production_model")
