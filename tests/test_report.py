from pathlib import Path

from llm_eval.report import generate_report, summarize_rows
from llm_eval.results import load_result_rows, write_apst_results_csv


def test_write_csv_and_generate_report(tmp_path: Path):
    payload = {
        "results": [
            {
                "model": "mock",
                "prompt_id": "p1",
                "prompt_type": "safety_harmful",
                "prompt_domain": "fraud",
                "temperature": 0.7,
                "n_samples": 4,
                "n_failures": 2,
                "failure_rate": 0.5,
                "empirical_failure_probability": 0.5,
                "reliability": 0.5,
                "apst_risk_horizon": 10,
                "apst_risk_at_10": 0.999,
                "failure_mode_distribution": {"safe_coherent": 2, "harmful": 1, "non_refusal": 1},
            }
        ]
    }
    csv_path = write_apst_results_csv(payload, tmp_path / "results.csv")
    rows = load_result_rows(csv_path)
    report_path = generate_report(results_path=csv_path, lang="both")

    assert rows[0]["model"] == "mock"
    assert summarize_rows(rows)["n_failures"] == 2
    assert report_path.exists()
    assert "APST Starter Kit Report" in report_path.read_text(encoding="utf-8")
