"""
# Install Prefect
pip install prefect

# Run the pipeline
python prefect_pipeline.py

# Start Prefect UI (optional)
prefect server start

# Deploy with schedule
prefect deployment build prefect_pipeline.py:training_pipeline \
  -n "monthly-training" \
  --cron "0 0 1 * *" \
  --apply
"""
# Access UI at http://localhost:6789


"""
Prefect Pipeline for NYC Taxi ML Training
Install: pip install prefect

Run: python prefect_pipeline.py
Deploy: prefect deployment build prefect_pipeline.py:training_pipeline -n "nyc-taxi-training"
"""

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
import pickle
from pathlib import Path
from typing import Tuple, Dict


@task(retries=3, retry_delay_seconds=60)
def read_dataframe(year: int, month: int) -> pd.DataFrame:
    """Load data from S3 with retry logic"""
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    return df


@task
def create_features(df_train: pd.DataFrame, df_val: pd.DataFrame) -> Dict:
    """Create features using DictVectorizer"""
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    
    # Training features
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_dicts)
    
    # Validation features
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    
    y_train = df_train['duration'].values
    y_val = df_val['duration'].values
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'dv': dv
    }


@task(log_prints=True)
def train_model(features: Dict, year: int, month: int) -> Tuple[str, float]:
    """Train XGBoost model and log to MLflow"""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)
    
    X_train = features['X_train']
    X_val = features['X_val']
    y_train = features['y_train']
    y_val = features['y_val']
    dv = features['dv']
    
    with mlflow.start_run() as run:
        # Add Prefect metadata
        mlflow.set_tag("orchestrator", "prefect")
        mlflow.set_tag("data_year", year)
        mlflow.set_tag("data_month", month)
        
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
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        # Save artifacts
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        
        run_id = run.info.run_id
        
        print(f"✅ Training completed - Run ID: {run_id}, RMSE: {rmse:.4f}")
        
        return run_id, rmse


@task
def save_run_id(run_id: str):
    """Save run ID to file"""
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"Run ID saved to run_id.txt")


@task(log_prints=True)
def register_model_if_good(run_id: str, rmse: float, threshold: float = 10.0):
    """Register model only if RMSE is below threshold"""
    if rmse < threshold:
        mlflow.set_tracking_uri("http://localhost:5000")
        model_uri = f"runs:/{run_id}/models_mlflow"
        mlflow.register_model(model_uri=model_uri, name="nyc-taxi-duration")
        print(f"✅ Model registered: {model_uri}")
    else:
        print(f"⚠️ Model not registered - RMSE {rmse:.4f} exceeds threshold {threshold}")


@flow(
    name="NYC Taxi Duration Training",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
def training_pipeline(year: int = 2024, month: int = 1):
    """Main training pipeline"""
    print(f"🚀 Starting training pipeline for {year}-{month:02d}")
    
    # Load data
    df_train = read_dataframe(year, month)
    
    # Calculate next month
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(next_year, next_month)
    
    # Create features
    features = create_features(df_train, df_val)
    
    # Train model
    run_id, rmse = train_model(features, year, month)
    
    # Save results
    save_run_id(run_id)
    
    # Conditionally register model
    register_model_if_good(run_id, rmse, threshold=10.0)
    
    print(f"✅ Pipeline completed successfully!")
    return {"run_id": run_id, "rmse": rmse}


# Schedule the flow (monthly retraining)
@flow
def scheduled_training():
    """Run training for multiple months"""
    from datetime import datetime
    
    current_date = datetime.now()
    year = current_date.year
    month = current_date.month
    # Calculate 2 months ago
    if month==1:
        month, year = 11, year-1
    elif month==2:
        month, year = 12, year-1
    else:
        month-=2
    
    # Train on two months ago, validation data is last month
    training_pipeline(year=year, month=month)


if __name__ == "__main__":
    # Run once
    result = training_pipeline(year=2024, month=1)
    print(f"Final result: {result}")
    
    # Or deploy for scheduling:
    # prefect deployment build prefect_pipeline.py:training_pipeline \
    #   -n "monthly-training" \
    #   -cron "0 0 1 * *" \
    #   --apply