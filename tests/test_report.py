from countdown_rl.report import build_report


def test_report_is_self_contained_and_escapes_model_name():
    result = {
        "model": "unsafe/<script>",
        "metrics": {
            "strict_accuracy": 0.5,
            "parseable_rate": 0.75,
            "exact_number_usage_rate": 0.6,
            "format_rate": 0.9,
            "failure_taxonomy": {"correct": 1, "wrong_value": 1},
        },
    }
    report = build_report([result], "Test")
    assert "<!doctype html>" in report
    assert "&lt;script&gt;" in report
    assert "50.0%" in report

