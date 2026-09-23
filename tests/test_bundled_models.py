"""发布模型必须独立于 artifacts，并保持版本与权重一致。"""
import hashlib
import json
from pathlib import Path

import pytest
from casmi import formula
from formula_adapter import FormulaAdapter


@pytest.mark.parametrize("preset,trees", [("multi", 263), ("enveda-only", 227)])
def test_bundled_model_loads_outside_workspace(preset, trees, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path(formula.__file__).resolve().parent / "models"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["models"][preset]
    for name, expected in entry["sha256"].items():
        assert hashlib.sha256((root / preset / name).read_bytes()).hexdigest() == expected
    adapter = FormulaAdapter() if preset == "multi" else FormulaAdapter(preset)
    assert adapter.model_path == root / preset
    assert adapter.model_info["num_trees"] == trees
    assert adapter.model_info["model_version"] == entry["model_version"]
    result = adapter.predict([{
        "precursor_mz": 181.0706646, "adduct": "[M+H]+",
        "ms2_mzs": [163.0601, 145.0495], "ms2_normalized_intensities": [1., .4],
    }], molecule_id="cloud-smoke", top_k=3)
    assert result["status"] == "ok"
    assert len(result["candidates"]) == 3
