"""Unit tests for Airflow DAG structure and pipeline tasks."""

import json
from pathlib import Path

import pytest

from dags.kdc_training_pipeline_dag import run_model_quality_gate


def test_quality_gate_passes_above_threshold(tmp_path: Path, monkeypatch):
    test_scores = {"loss": 0.15, "accuracy": 0.92}
    scores_file = tmp_path / "scores.json"
    scores_file.write_text(json.dumps(test_scores))

    # Point PROJECT_ROOT to tmp_path
    monkeypatch.setattr("dags.kdc_training_pipeline_dag.PROJECT_ROOT", tmp_path)

    res = run_model_quality_gate(min_accuracy_threshold=0.85)
    assert res["accuracy"] == 0.92
    assert res["loss"] == 0.15


def test_quality_gate_fails_below_threshold(tmp_path: Path, monkeypatch):
    test_scores = {"loss": 0.65, "accuracy": 0.60}
    scores_file = tmp_path / "scores.json"
    scores_file.write_text(json.dumps(test_scores))

    monkeypatch.setattr("dags.kdc_training_pipeline_dag.PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Quality Gate Failed"):
        run_model_quality_gate(min_accuracy_threshold=0.80)


def test_airflow_dag_structure():
    pytest.importorskip("airflow")
    from dags.kdc_training_pipeline_dag import build_kdc_dag

    dag = build_kdc_dag()
    if dag is None:
        pytest.skip("Airflow not installed in this environment.")

    assert dag.dag_id == "kdc_training_pipeline_dag"

    expected_tasks = {
        "start_pipeline",
        "task_data_ingestion",
        "task_prepare_base_model",
        "task_model_training",
        "task_model_evaluation",
        "task_model_quality_gate",
        "pipeline_complete",
    }
    assert set(dag.task_dict.keys()) == expected_tasks

    # Verify linear dependency order
    assert (
        dag.get_task("task_prepare_base_model")
        in dag.get_task("task_data_ingestion").downstream_list
    )
    assert (
        dag.get_task("task_model_training")
        in dag.get_task("task_prepare_base_model").downstream_list
    )
    assert (
        dag.get_task("task_model_evaluation") in dag.get_task("task_model_training").downstream_list
    )
    assert (
        dag.get_task("task_model_quality_gate")
        in dag.get_task("task_model_evaluation").downstream_list
    )
