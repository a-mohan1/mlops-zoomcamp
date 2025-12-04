import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.sklearn.autolog(disable=True)

experiment_name = "random-forest-hyperopt"
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if experiment and experiment.lifecycle_stage == 'deleted':
    # Restore the deleted experiment
    client.restore_experiment(experiment.experiment_id)
    print(f"Restored experiment: {experiment_name}")

# create and set the experiment
mlflow.set_experiment(experiment_name)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):

        with mlflow.start_run(nested=True):

            params_int = {
                'max_depth': int(params['max_depth']),
                'n_estimators': int(params['n_estimators']),
                'min_samples_split': int(params['min_samples_split']),
                'min_samples_leaf': int(params['min_samples_leaf']),
                'random_state': 42
            }
            rf = RandomForestRegressor(**params_int)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_params(rf.get_params())
            mlflow.sklearn.log_model(rf, artifact_path="model")

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    }

    with mlflow.start_run(run_name="hyperopt_optimization"):
        rstate = np.random.default_rng(42)
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=Trials(),
            rstate=rstate
        )
        
        # Log best parameters
        best_params_int = {
            'max_depth': int(best_params['max_depth']),
            'n_estimators': int(best_params['n_estimators']),
            'min_samples_split': int(best_params['min_samples_split']),
            'min_samples_leaf': int(best_params['min_samples_leaf']),
        }
        mlflow.log_params({f"best_{k}": v for k, v in best_params_int.items()})
        print(f"Best parameters: {best_params_int}")


if __name__ == '__main__':
    run_optimization()

    """
    # Get the experiment
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id    

    # Search for runs with the lowest RMSE
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    """
