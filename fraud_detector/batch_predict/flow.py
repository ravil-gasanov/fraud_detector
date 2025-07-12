from loguru import logger
from prefect import flow

from fraud_detector.batch_predict.tasks import (
    load_production_model,
    make_predictions,
    save_predictions,
)
from fraud_detector.common import load_X_y
from fraud_detector.config import TEST_PATH


@flow
def batch_predict_flow():
    logger.info("Starting batch prediction flow")

    logger.info("Loading data for batch prediction")
    test_X, test_y = load_X_y(data_path=TEST_PATH)

    logger.info("Loading production model")
    model = load_production_model()

    logger.info("Making predictions")
    predictions = make_predictions(model=model, X=test_X)

    logger.info("Saving predictions")
    save_predictions(predictions=predictions)

    logger.info("Batch prediction flow completed successfully")
