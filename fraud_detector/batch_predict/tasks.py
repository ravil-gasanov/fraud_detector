from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
from prefect import task
import sqlalchemy

from fraud_detector.config import DATABASE_URL, MLFLOW_TRACKING_URI, PREDICTIONS_PATH


@task
def load_production_model():
    """
    Load the production model for batch prediction.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_name = "fraud_detector_production_model"
    model_version = "latest"

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    return model


@task
def make_predictions(model, X):
    """
    Make predictions using the provided model and input data.

    :param model: The trained model to use for predictions.
    :param X: Input data for which predictions are to be made.
    :return: Predictions made by the model.
    """
    y_pred = model.predict(X)

    predictions = pd.DataFrame({"prediction": y_pred})

    return predictions


@task
def save_predictions(predictions, table_name="predictions"):
    """
    Save predictions to SQL database.

    Args:
        predictions: List or array of prediction values
        table_name: Name of the database table
    """
    # Save to database
    try:
        engine = sqlalchemy.create_engine(DATABASE_URL)
        predictions.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
        )
    except Exception as e:
        logger.error(f"Error saving predictions to database: {e}")

    logger.info("Saving to a local csv file")
    predictions.to_csv(PREDICTIONS_PATH, index=False)
