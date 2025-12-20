#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

from datetime import date

import mlflow
#from mlflow.tracking import MlflowClient
from prefect import flow, task, get_run_logger


EXPERIMENT_NAME = "nyc-taxi-experiment"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
#client = MlflowClient()

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


@task(retries=3, retry_delay_seconds=5)
def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    #df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


@task(log_prints=True)
def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task
def train_model(X_train, y_train, X_val, y_val, dv):
    logger = get_run_logger()

    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        logger.info(f"Validation rmse: {rmse}")

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="model")

        return run.info.run_id


@flow
def main_flow():
    logger = get_run_logger()

    cur_month = date.today().month
    cur_year = date.today().year
    # Calculate 2 months ago
    if cur_month==1:
        month, year = 11, cur_year-1
    elif cur_month==2:
        month, year = 12, cur_year-1
    else:
        month, year = cur_month - 2, cur_year

    df_train = read_dataframe(year=year, month=month)

    val_year = year if month < 12 else year + 1
    val_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=val_year, month=val_month)
    #df_test = read_dataframe(year=cur_year, month=cur_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)
    #X_test, _ = create_X(df_test, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    #y_test = df_test[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    logger.info(f"MLflow run_id: {run_id}")

    with open("run_id.txt", "w") as f:
        f.write(run_id)

    #experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    model_uri = f"runs:/{run_id}/model"  
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    #y_test_pred = loaded_model.predict(X_test)
    #test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

    # Register model
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name='nyc-taxi'
    )
    logger.info(f"Registered model from run {run_id}")
    #print(f"Registered model from run {run_id} with test_rmse: {test_rmse}")
    return run_id


if __name__ == "__main__":
    
    """
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)
    """
    
    run_id = main_flow()