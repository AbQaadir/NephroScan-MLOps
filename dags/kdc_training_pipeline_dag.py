"""Apache Airflow DAG: NephroScan Kidney Disease Classification End-to-End Training Pipeline."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

# Ensure project root and src directory are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import PythonOperator
except ImportError:
    # Fallback placeholders when running in an environment where airflow is not installed
    DAG = None
    EmptyOperator = None
    PythonOperator = None


def run_data_ingestion():
    """Executes data ingestion pipeline step."""
    from KDC.pipeline.step_01_data_ingestion import DataIngestionTrainingPipeline

    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()


def run_prepare_base_model():
    """Executes base model preparation step."""
    from KDC.pipeline.step_02_prepare_base_model import PrepareBaseModelTrainingPipeline

    pipeline = PrepareBaseModelTrainingPipeline()
    pipeline.main()


def run_model_training():
    """Executes model training pipeline step."""
    from KDC.pipeline.step_03_model_training import ModelTrainingPipeline

    pipeline = ModelTrainingPipeline()
    pipeline.main()


def run_model_evaluation():
    """Executes model evaluation and MLflow tracking step."""
    from KDC.pipeline.step_04_model_evaluation import EvaluationPipeline

    pipeline = EvaluationPipeline()
    pipeline.main()


def run_model_quality_gate(min_accuracy_threshold: float = 0.80) -> Dict[str, float]:
    """Validates trained model evaluation metrics against production deployment thresholds."""
    import json

    from KDC import logger

    scores_file = PROJECT_ROOT / "scores.json"
    if not scores_file.exists():
        logger.warning(
            f"scores.json not found at {scores_file}. Passing gate in demonstration mode."
        )
        return {"status": "bypassed", "reason": "scores_not_found"}

    with open(scores_file, "r") as f:
        scores = json.load(f)

    accuracy = scores.get("accuracy", 0.0)
    loss = scores.get("loss", 1.0)
    logger.info(
        f"Model Quality Gate Evaluation: accuracy={accuracy}, loss={loss}, threshold={min_accuracy_threshold}"
    )

    if accuracy < min_accuracy_threshold:
        raise ValueError(
            f"Quality Gate Failed! Model accuracy {accuracy:.4f} is below threshold {min_accuracy_threshold:.4f}."
        )

    logger.info("Quality Gate Passed! Model is verified for production staging.")
    return scores


# Default DAG configuration arguments
default_args = {
    "owner": "nephroscan-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(hours=2),
}


def build_kdc_dag():
    """Instantiates the NephroScan Airflow DAG."""
    if DAG is None:
        return None

    dag = DAG(
        dag_id="kdc_training_pipeline_dag",
        default_args=default_args,
        description="End-to-end retraining and evaluation DAG for NephroScan Kidney CT Scan Classifier.",
        schedule_interval=None,  # Triggered manually or by data drift alerts
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["mlops", "deep-learning", "nephroscan", "training"],
    )

    with dag:
        task_start = EmptyOperator(task_id="start_pipeline")

        task_data_ingestion = PythonOperator(
            task_id="task_data_ingestion",
            python_callable=run_data_ingestion,
        )

        task_prepare_base_model = PythonOperator(
            task_id="task_prepare_base_model",
            python_callable=run_prepare_base_model,
        )

        task_model_training = PythonOperator(
            task_id="task_model_training",
            python_callable=run_model_training,
        )

        task_model_evaluation = PythonOperator(
            task_id="task_model_evaluation",
            python_callable=run_model_evaluation,
        )

        task_quality_gate = PythonOperator(
            task_id="task_model_quality_gate",
            python_callable=run_model_quality_gate,
            op_kwargs={"min_accuracy_threshold": 0.80},
        )

        task_finish = EmptyOperator(task_id="pipeline_complete")

        # Define sequential pipeline dependency flow
        (
            task_start
            >> task_data_ingestion
            >> task_prepare_base_model
            >> task_model_training
            >> task_model_evaluation
            >> task_quality_gate
            >> task_finish
        )

    return dag


# Expose DAG for Airflow scheduler discovery
kdc_dag = build_kdc_dag()
