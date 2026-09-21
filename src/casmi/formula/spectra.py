"""谱图输入校验、合峰和归一化。

这里把外部行记录转换成内部 ``Spectrum``，后续候选搜索和特征提取只依赖
这个已经校验过的表示。清洗规则必须在训练和推理之间保持一致。
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .chemistry import ADDUCTS, Adduct


class UnsupportedInput(ValueError):
    """表示格式合法但加合物等处理规则不在本项目支持范围内。"""

    pass


@dataclass
class Spectrum:
    """一条清洗后的谱图，同时保留原始峰用于指纹和去重。"""

    spectrum_id: str
    precursor_mz: float
    adduct: Adduct
    mzs: np.ndarray
    intensities: np.ndarray
    raw_mzs: np.ndarray
    raw_intensities: np.ndarray
    collision_energy: tuple[float, ...] | None
    instrument: str | None
    base_peak_intensity: float | None

    @property
    def fingerprint(self):
        """根据输入观测和元数据生成精确重复谱去重用指纹。"""
        values = [
            self.precursor_mz,
            self.adduct.name,
            self.raw_mzs.tolist(),
            self.raw_intensities.tolist(),
            self.collision_energy,
            self.instrument,
            self.base_peak_intensity,
        ]
        return hashlib.sha256(json.dumps(values, allow_nan=False).encode()).hexdigest()


def normalize_spectrum(row, config, index=0):
    """校验并规范化一条外部谱图记录。

    处理顺序是：验证加合物和离子模式、验证数组、合并相同 m/z、按基峰归一化、
    过滤弱峰并保留最多 ``config.max_peaks`` 个峰，最后校验碰撞能量和仪器元数据。
    原始峰仍保存在返回对象中，因此重复谱判断不会被清洗后的排序影响。
    """
    name = row.get("adduct")
    if name not in ADDUCTS:
        raise UnsupportedInput(f"Unsupported adduct: {name}")
    adduct = ADDUCTS[name]
    mode = "positive" if adduct.charge > 0 else "negative"
    if row.get("ionization_mode") not in (None, mode):
        raise ValueError("ionization_mode conflicts with adduct")
    mz = float(row["precursor_mz"])
    raw_mzs = np.asarray(row["ms2_mzs"], dtype=float)
    raw_ints = np.asarray(row["ms2_normalized_intensities"], dtype=float)
    if not np.isfinite(mz) or mz <= 0:
        raise ValueError("precursor_mz must be positive and finite")
    if raw_mzs.ndim != 1 or raw_ints.ndim != 1 or len(raw_mzs) != len(raw_ints):
        raise ValueError("Peak arrays must be aligned one-dimensional arrays")
    if (
        not np.isfinite(raw_mzs).all()
        or not np.isfinite(raw_ints).all()
        or (raw_mzs <= 0).any()
        or (raw_ints < 0).any()
    ):
        raise ValueError("Invalid peak mass or intensity")
    # 相同 m/z 的观测峰先合并，避免同一物理峰因重复记录被重复计权。
    unique, inverse = np.unique(raw_mzs, return_inverse=True)
    intensities = np.bincount(inverse, weights=raw_ints, minlength=len(unique))
    if len(intensities) and intensities.max() > 0:
        intensities /= intensities.max()
        keep = (intensities >= config.intensity_floor) & (intensities > 0)
        unique, intensities = unique[keep], intensities[keep]
        # 强度阈值和峰数上限只影响碎片证据；候选质量搜索仍由前体质量控制。
        keep = np.argsort(-intensities, kind="stable")[: config.max_peaks]
        keep = keep[np.argsort(unique[keep])]
        unique, intensities = unique[keep], intensities[keep]
    else:
        unique, intensities = np.array([], dtype=float), np.array([], dtype=float)
    # 可选元数据只作为特征，不能绕过上面的物理输入校验。
    ce = row.get("collision_energy_ev")
    if ce is not None:
        ce = np.asarray(ce, dtype=float)
        if ce.ndim != 1 or not np.isfinite(ce).all() or (ce < 0).any():
            raise ValueError("collision_energy_ev must be a finite nonnegative list")
        ce = tuple(float(v) for v in ce) or None
    base = row.get("base_peak_intensity")
    if base is not None:
        base = float(base)
        if not np.isfinite(base) or base < 0:
            raise ValueError("Invalid base_peak_intensity")
    instrument = row.get("instrument_type")
    if instrument is not None and not isinstance(instrument, str):
        raise ValueError("instrument_type must be a string or null")
    return Spectrum(
        str(row.get("spectrum_id") or f"spectrum_{index}"),
        mz,
        adduct,
        unique,
        intensities,
        raw_mzs.copy(),
        raw_ints.copy(),
        ce,
        instrument,
        base,
    )
