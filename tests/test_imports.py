from __future__ import annotations


def test_core_modules_import():
    import src.api.app  # noqa: F401
    import src.evaluation.evaluate_over_prob_holdout  # noqa: F401
    import src.evaluation.market_readiness  # noqa: F401
    import src.inference.scan_slate_with_model  # noqa: F401
    import src.models.build_points_regression  # noqa: F401
    import src.warehouse.materialize  # noqa: F401
