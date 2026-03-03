from pathlib import Path

from lifecycle.src.runner import CreditPipelineRunner


def test_credit_pipeline_runner_end_to_end(tmp_path):
    runner = CreditPipelineRunner()
    raw_dataset_path = Path("lifecycle/data/raw/risco_credito.csv")
    artifact_dir = tmp_path / "artifacts"

    artifacts = runner.run_training_pipeline(raw_dataset_path, artifact_dir)

    assert artifacts["clean_dataset"].exists()
    assert artifacts["raw_features"].exists()
    assert artifacts["normalized_features"].exists()
    assert artifacts["featurizer_state"].exists()
    assert artifacts["model"].exists()

    predictions = runner.run_inference_pipeline(
        raw_dataset_path,
        artifact_dir,
        prediction_path=tmp_path / "predictions.csv",
    )

    assert "predicted_inadimplente" in predictions.columns
    assert "score_inadimplencia" in predictions.columns
    assert len(predictions) == 10
