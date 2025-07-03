from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
from prefect import task
from sklearn.metrics import f1_score

from fraud_detector.config import MLFLOW_TRACKING_URI


@task
def load_best_model_from_experiment(experiment_name: str):
    """
    Load the best model from an MLflow experiment based on highest F1 score.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        Loaded sklearn model object
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # Initialize client
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Search for runs ordered by F1 score (highest first)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found in the experiment")

    best_run = runs[0]

    # Print info about the best run
    logger.info(f"Best run ID: {best_run.info.run_id}")
    logger.info(f"Best F1 score: {best_run.data.metrics.get('best_cv_score', 'N/A')}")

    # Load the sklearn model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    return model


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
def register_model(model, model_name: str):
    """
    Register an sklearn model to MLflow Model Registry.

    Args:
        model: Trained sklearn model object
        model_name: Name to register the model under in the registry

    Returns:
        ModelVersion object from MLflow
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        with mlflow.start_run():
            # Log the sklearn model
            mlflow.sklearn.log_model(
                sk_model=model, artifact_path="model", registered_model_name=model_name
            )
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise e
