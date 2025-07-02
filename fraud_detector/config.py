FEATURE_COLUMNS = ["V4", "V11", "V7", "Amount"]
TARGET_COLUMN = "Class"

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

RANDOM_STATE = 42

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "fraud_detection_experiment"

METRIC_THRESHOLD = 0.6  # Example threshold for F1 score