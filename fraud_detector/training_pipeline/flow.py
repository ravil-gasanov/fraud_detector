from loguru import logger
from prefect import flow

from fraud_detector.common import load_X_y
from fraud_detector.config import METRIC_THRESHOLD, MLFLOW_EXPERIMENT_NAME, TEST_PATH, TRAIN_PATH
from fraud_detector.training_pipeline.tasks import (
    eval_model_on_test,
    load_best_model_from_experiment,
    register_model,
    train_model,
)


@flow
def train_flow():
    logger.info("Starting the training flow...")

    logger.info(f"Loading training data from {TRAIN_PATH}")
    X, y = load_X_y(data_path=TRAIN_PATH)

    logger.info(f"Loading test data from {TEST_PATH}")
    test_X, test_y = load_X_y(data_path=TEST_PATH)

    logger.info(f"Loading the best model from experiment: {MLFLOW_EXPERIMENT_NAME}")
    model = load_best_model_from_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)

    logger.info("Training the model...")
    model = train_model(model=model, X=X, y=y)

    logger.info("Evaluating the model on the test set...")
    metric = eval_model_on_test(model=model, test_X=test_X, test_y=test_y)

    if metric >= METRIC_THRESHOLD:
        logger.info(
            f"Model met the metric threshold of {METRIC_THRESHOLD}. "
            f"Current metric: {metric}. Registering the model."
        )
        register_model(model=model, model_name="fraud_detector_production_model")
    else:
        logger.warning(
            f"Model did not meet the metric threshold of {METRIC_THRESHOLD}. "
            f"Current metric: {metric}. Model will not be registered."
        )
