"""Atelier 3, step 5: run some flows automatically once a day.

Terminal 1: prefect server start
Terminal 2: prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
            python deploiement_prefect.py
Terminal 3: prefect deployment ls
            prefect deployment run 'all/ml-pipeline-all'
"""

from prefect import serve

from pipeline_prefect import all_flow, evaluate_flow, train_flow

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(name="ml-pipeline-all", cron="0 8 * * *"),  # every day 08:00
        train_flow.to_deployment(name="ml-pipeline-train", cron="0 20 * * *"),  # every day 20:00
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate"),  # manual only
    )
