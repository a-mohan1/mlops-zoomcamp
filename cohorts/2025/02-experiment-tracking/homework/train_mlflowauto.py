import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(".", data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(".", data_path, "val.pkl"))
        rf_params = {'max_depth': 10, 'random_state': 0}

        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
    
        rmse = root_mean_squared_error(y_val, y_pred)

    
if __name__ == '__main__':
    TRACKING_SERVER_HOST = "127.0.0.1"
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

    experiment_name = "nyc-taxi-experiment2"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()
    
    run_train()
