#from prefect.deployments import Deployment
#from prefect.orion.schemas.schedules import CronSchedule

from prefect import flow
from score import ride_duration_prediction

# Deprecated
"""
deployment = Deployment.build_from_flow(
    flow=ride_duration_prediction,
    name="ride_duration_prediction",
    parameters={
        "taxi_type": "green",
        "run_id": "e1efc53e9bd149078b0c12aeaa6365df",
    },
    schedule=CronSchedule(cron="0 3 2 * *"),
    work_queue_name="ml",
)
"""

# 
"""
if __name__ == "__main__":
    ride_duration_prediction.from_source(
        source="./04-deployment/batch/",  # Current directory
        entrypoint="score.py:ride_duration_prediction"
    ).deploy(
        name="ride_duration_prediction",
        parameters={
            "taxi_type": "green",
            "run_id": "e1efc53e9bd149078b0c12aeaa6365df",
        },
        cron="0 3 2 * *",  # 3 AM on the 2nd of each month
        work_pool_name="zoompool",
    )
"""

# For local deployment
if __name__ == "__main__":
    ride_duration_prediction.serve(
        name="ride_duration_prediction",
        parameters={
            "taxi_type": "green",
            "run_id": "e1efc53e9bd149078b0c12aeaa6365df",
        },
        cron="0 3 2 * *",
    )
