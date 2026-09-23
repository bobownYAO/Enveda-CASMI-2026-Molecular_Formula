"""Adapter 的模型路由、结果兼容性和文件边界。"""

import json

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

import formula_adapter as adapter
from casmi.formula import FormulaConfig, FormulaPredictor
from casmi.formula.features import FEATURE_NAMES


SPECTRUM = dict(
    precursor_mz=181.0706646,
    adduct="[M+H]+",
    ms2_mzs=[163.0601, 145.0495],
    ms2_normalized_intensities=[1.0, 0.4],
)


@pytest.fixture
def models(tmp_path, monkeypatch):
    """不依赖未纳入版本控制的本地实验模型。"""
    cfg = FormulaConfig(
        element_limits=dict(C=6, H=12, N=0, O=6, P=0, S=0, F=0, Cl=0, Br=0, I=0)
    )
    rng = np.random.default_rng(42)
    dataset = lgb.Dataset(
        rng.random((20, len(FEATURE_NAMES))),
        label=[1, 0] * 10,
        group=[2] * 10,
        feature_name=FEATURE_NAMES,
    )
    booster = lgb.train(
        dict(
            objective="lambdarank",
            verbosity=-1,
            num_threads=1,
            min_data_in_leaf=1,
            num_leaves=3,
        ),
        dataset,
        num_boost_round=2,
    )
    presets = {}
    for name in ("multi", "enveda-only"):
        folder = tmp_path / name
        folder.mkdir()
        booster.save_model(str(folder / "model.txt"))
        (folder / "metadata.json").write_text(
            json.dumps(
                dict(
                    artifact_version=1,
                    feature_names=FEATURE_NAMES,
                    config=cfg.to_dict(),
                    config_version=cfg.version,
                    model_version=name,
                )
            ),
            encoding="utf-8",
        )
        presets[name] = name
    monkeypatch.setattr(adapter, "MODEL_PRESETS", presets)
    monkeypatch.setattr(adapter, "_ADAPTER_DIR", tmp_path)
    monkeypatch.setattr(
        adapter, "CONFIG", dict(lightgbm_model="multi", top_k=3, workers=1)
    )
    return tmp_path


def test_model_selection_config_override_and_working_directory(
    models, monkeypatch, tmp_path
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert adapter.FormulaAdapter().model_info["model_version"] == "multi"
    adapter.CONFIG["lightgbm_model"] = "enveda-only"
    assert adapter.FormulaAdapter().model_info["model_version"] == "enveda-only"
    assert adapter.FormulaAdapter("multi").model_info["model_version"] == "multi"
    model = adapter.FormulaAdapter(models / "multi" / "model.txt")
    direct = (
        FormulaPredictor.load(models / "multi").predict([SPECTRUM], "x", 3).to_dict()
    )
    result = model.predict([SPECTRUM], molecule_id="x")
    assert result["candidates"] == direct["candidates"]
    assert result["model_version"] == direct["model_version"]
    assert result["status"] == direct["status"]
    json.dumps(result, allow_nan=False)


def test_invalid_input_and_model_errors_remain_visible(models):
    model = adapter.FormulaAdapter()
    assert (
        model.predict([dict(SPECTRUM, ms2_normalized_intensities=[])])["status"]
        == "invalid_input"
    )
    assert (
        model.predict([dict(SPECTRUM, adduct="[2M+H]+")])["status"]
        == "unsupported_input"
    )
    with pytest.raises(ValueError, match="top_k"):
        model.predict([SPECTRUM], top_k=False)
    with pytest.raises(ValueError, match="workers"):
        adapter.FormulaAdapter(workers=0)
    with pytest.raises(FileNotFoundError):
        adapter.FormulaAdapter("nonexistent")
    metadata = models / "multi" / "metadata.json"
    data = json.loads(metadata.read_text())
    data["feature_names"] = ["wrong"]
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incompatible"):
        adapter.FormulaAdapter("multi")


def test_parquet_groups_and_jsonl_preserve_failures_and_existing_files(models):
    path = models / "input.parquet"
    pl.DataFrame(
        [
            dict(SPECTRUM, molecule_id="good"),
            dict(SPECTRUM, molecule_id="bad", adduct="unsupported"),
        ]
    ).write_parquet(path)
    output = models / "results.jsonl"
    model = adapter.FormulaAdapter()
    results = model.predict_parquet(path, output_path=output)
    assert [r["molecule_id"] for r in results] == ["bad", "good"]
    assert results[0]["status"] == "unsupported_input"
    assert results == [json.loads(line) for line in output.read_text().splitlines()]
    assert (
        results[1]["candidates"]
        == adapter.predict_formula([SPECTRUM], lightgbm_model="multi")["candidates"]
    )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        model.predict_parquet(path, output_path=path)
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        model.predict_parquet(path, output_path=output)
