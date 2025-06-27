from prefect import task


@task
def load_train_X_y():
    pass


@task
def load_test_X_y():
    pass


@task
def load_model_from_model_registry():
    pass


@task
def train_model(model, X, y):
    pass


@task
def eval_model_on_test(model, test_X, test_y):
    pass


@task
def register_model(model, model_name):
    pass
