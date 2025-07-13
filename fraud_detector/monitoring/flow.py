from prefect import flow

from fraud_detector.monitoring.tasks import (
    calculate_metrics,
    load_new_data,
    load_predictions_on_new_data,
    load_reference_data,
    save_metrics,
)


@flow(name="monitoring_flow")
def monitoring_flow():
    # Load reference data
    reference_data = load_reference_data()

    # Load new data
    new_data = load_new_data()

    # Load predictions on new data
    predictions = load_predictions_on_new_data()

    # Calculate metrics based on the loaded data and predictions
    metrics = calculate_metrics(reference_data, new_data, predictions)

    # Save the calculated metrics
    save_metrics(metrics)
