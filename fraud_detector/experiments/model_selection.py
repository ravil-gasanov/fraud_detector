import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from fraud_detector.experiments.feature_transformer import FeatureTransformer

RANDOM_STATE = 42

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "fraud_detection_experiment"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def get_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def get_models():
    models = [
        ("logreg", LogisticRegression()),
        ("rfc", RandomForestClassifier()),
    ]

    return models


def get_model_params(model_name):
    params = {
        "logreg": [
            {
                "logreg__penalty": ["l1", "l2", "elasticnet"],
                "logreg__max_iter": [100, 500, 1000],
                "logreg__solver": ["liblinear"],
            },
        ],
        "rfc": [{}],
    }

    return params[model_name]


def build_pipeline(model_name, model):
    steps = [
        ("FeatureTransformer", FeatureTransformer()),
        (model_name, model),
    ]

    return Pipeline(steps=steps)


def load_train_X_y():
    FEATURE_COLUMNS = ["V4", "V11", "V7", "Amount"]
    TARGET_COLUMN = "Class"

    data = pd.read_csv("data/train.csv")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    return X, y


def run_experiments():
    X, y = load_train_X_y()

    cv = get_cv()

    models = get_models()

    mlflow.sklearn.autolog()

    for model_name, model in models:
        with mlflow.start_run(run_name=model_name):
            pipeline = build_pipeline(model_name=model_name, model=model)
            param_grid = get_model_params(model_name=model_name)

            gridcv = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="f1",
            )

            gridcv.fit(X=X, y=y)

            print(f"Best estimator: {gridcv.best_estimator_}")
            print(f"Mean test f1-score: {gridcv.best_score_}")
