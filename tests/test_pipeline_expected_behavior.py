"""
Tests for expected pipeline and analysis behavior.

- generate_all produces fig_survival_all_in_one and does NOT produce
  fig3_survival_curves, fig_survival_km_ci, fig_survival_1_minus_censoring_err,
  fig_survival_joint_censoring.
- get_attribute_inference_target: sensitive_attributes[0] when present, else primary_target.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── Attribute inference default ───────────────────────────────────────────

def test_get_attribute_inference_target_prefers_sensitive_attributes():
    from metrics.privacy.attribute_inference import get_attribute_inference_target
    schema = {"sensitive_attributes": ["status"], "target_spec": {"primary_target": "time"}}
    assert get_attribute_inference_target(schema) == "status"


def test_get_attribute_inference_target_fallback_primary_target():
    from metrics.privacy.attribute_inference import get_attribute_inference_target
    schema = {"target_spec": {"primary_target": "status"}}
    assert get_attribute_inference_target(schema) == "status"


def test_get_attribute_inference_target_empty_sensitive_ignored():
    from metrics.privacy.attribute_inference import get_attribute_inference_target
    schema = {"sensitive_attributes": [], "target_spec": {"primary_target": "status"}}
    assert get_attribute_inference_target(schema) == "status"


def test_get_attribute_inference_target_missing_returns_none():
    from metrics.privacy.attribute_inference import get_attribute_inference_target
    schema = {"target_spec": {}}
    assert get_attribute_inference_target(schema) is None


# ─── generate_all task list (expected figures) ───────────────────────────

def test_generate_all_includes_survival_all_in_one():
    """Pipeline must produce fig_survival_all_in_one.pdf."""
    source = (ROOT / "analysis" / "generate_all.py").read_text()
    assert "fig_survival_all_in_one.pdf" in source
    assert "Figure: Survival metrics all-in-one" in source


def test_generate_all_does_not_include_removed_survival_figures():
    """Pipeline must NOT produce fig3_survival_curves, fig_survival_km_ci, etc."""
    source = (ROOT / "analysis" / "generate_all.py").read_text()
    removed_outputs = [
        "fig3_survival_curves.pdf",
        "fig_survival_km_ci.pdf",
        "fig_survival_1_minus_censoring_err.pdf",
        "fig_survival_joint_censoring.pdf",
    ]
    for name in removed_outputs:
        assert name not in source, f"Pipeline should not generate {name}"


