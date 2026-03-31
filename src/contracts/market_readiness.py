"""Default market readiness labels used by scoring and API layers."""

from __future__ import annotations

from typing import Any, Dict


DEFAULT_MARKET_READINESS: Dict[str, Dict[str, Any]] = {
    "player_points": {
        "status": "production",
        "tier": "beta_primary",
        "label": "Production",
        "summary": (
            "Primary beta market. Best current holdout accuracy and calibrated "
            "probability quality among the live prop models."
        ),
    },
    "player_rebounds": {
        "status": "experimental",
        "tier": "beta_secondary",
        "label": "Experimental",
        "summary": (
            "Model beats simple rebounding baselines marginally, but still needs "
            "real-line backtests and readiness gating."
        ),
    },
    "player_assists": {
        "status": "experimental",
        "tier": "beta_secondary",
        "label": "Experimental",
        "summary": (
            "Model improves on rolling baselines slightly, but not enough yet for "
            "default product exposure."
        ),
    },
    "player_threes": {
        "status": "experimental",
        "tier": "beta_secondary",
        "label": "Experimental",
        "summary": (
            "Model is directionally useful, but variance and market-readiness work "
            "are still outstanding."
        ),
    },
    "game_moneyline": {
        "status": "planned",
        "tier": "future",
        "label": "Planned",
        "summary": "Game-level moneyline modeling has not been implemented yet.",
    },
    "game_spread": {
        "status": "planned",
        "tier": "future",
        "label": "Planned",
        "summary": "Game-level spread modeling has not been implemented yet.",
    },
    "game_total": {
        "status": "planned",
        "tier": "future",
        "label": "Planned",
        "summary": "Game-level total modeling has not been implemented yet.",
    },
}


def get_market_readiness(market_key: str) -> Dict[str, Any]:
    """Return readiness metadata for a market with a safe default."""
    return DEFAULT_MARKET_READINESS.get(
        market_key,
        {
            "status": "experimental",
            "tier": "unknown",
            "label": "Experimental",
            "summary": "No explicit readiness rule has been defined for this market.",
        },
    )

