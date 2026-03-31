"""Helpers for versioned model bundles and scored output contracts."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.contracts.market_readiness import get_market_readiness
from src.contracts.versions import MODEL_BUNDLE_SCHEMA_VERSION


def utc_now_iso() -> str:
    """Return a compact UTC timestamp string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: object, prefix: Optional[str] = None) -> str:
    """Create a stable short identifier from deterministic string parts."""
    raw = "||".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}" if prefix else digest


def default_readiness_status(target: str) -> str:
    """Infer a default readiness status from the training target."""
    if target == "target_pts":
        return get_market_readiness("player_points")["status"]
    if target == "target_reb":
        return get_market_readiness("player_rebounds")["status"]
    if target == "target_ast":
        return get_market_readiness("player_assists")["status"]
    if target == "target_fg3m":
        return get_market_readiness("player_threes")["status"]
    return "experimental"


def build_training_window(
    train_min: Optional[int] = None,
    train_max: Optional[int] = None,
    val_min: Optional[int] = None,
    val_max: Optional[int] = None,
) -> Dict[str, Optional[int]]:
    """Create a normalized training window payload."""
    return {
        "train_min_season": train_min,
        "train_max_season": train_max,
        "val_min_season": val_min,
        "val_max_season": val_max,
    }


def build_model_bundle_metadata(
    *,
    target: str,
    training_window: Dict[str, Optional[int]],
    readiness_status: Optional[str] = None,
    model_type: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build shared metadata for saved model bundles."""
    metadata: Dict[str, Any] = {
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "target": target,
        "training_window": training_window,
        "readiness_status": readiness_status or default_readiness_status(target),
        "artifact_created_at": utc_now_iso(),
    }
    if model_type:
        metadata["model_type"] = model_type
    if extra:
        metadata.update(extra)
    return metadata


def derive_model_version(model_path: Path, bundle: Optional[Dict[str, Any]] = None) -> str:
    """Return a stable model version string from bundle metadata or filesystem info."""
    if bundle:
        created_at = bundle.get("artifact_created_at")
        target = bundle.get("target", model_path.stem)
        if created_at:
            return f"{target}:{created_at}"

    try:
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
        return f"{model_path.stem}:{mtime.replace(microsecond=0).isoformat()}"
    except FileNotFoundError:
        return f"{model_path.stem}:unknown"
