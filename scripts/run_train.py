from fraud_detector.training_pipeline.flow import train_flow
from fraud_detector.training_pipeline.tasks import load_best_model_from_experiment

from loguru import logger

logger.add(
    "logs/train.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO",
    format="{time} {level} {message}",
)

if __name__ == "__main__":
    train_flow()
