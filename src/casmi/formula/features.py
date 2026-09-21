"""训练与推理共用的候选条件碎片证据特征。

给定候选中性分子式和已归一化谱图，本模块计算离子组成能否解释碎片峰，
并把逐谱结果聚合成固定顺序的模型输入。训练和预测必须使用同一个特征提取器。
"""

import numpy as np
from numba import njit

from .candidates import FormulaEnumerator
from .chemistry import ATOM_MASS, ELECTRON_MASS, ELEMENTS, MASSES

# 这组名字同时参与模型元数据校验；增删特征必须提升模型/缓存兼容版本。
PER_SPECTRUM = (
    "abs_ppm",
    "supported",
    "peak_fraction",
    "intensity_fraction",
    "fragment_error_fraction",
    "water_loss_fraction",
    "co_loss_fraction",
    "co2_loss_fraction",
    "ammonia_loss_fraction",
    "log_peak_count",
    "positive",
    "ce_mean",
    "ce_min",
    "ce_max",
    "ce_missing",
    "log_base_peak",
    "base_missing",
    "timsTOF",
    "orbitrap",
    "instrument_missing",
)
FEATURE_NAMES = (
    [f"count_{e}" for e in ELEMENTS]
    + ["dbe", "h_c", "hetero_c", "mass"]
    + [f"{name}_{agg}" for agg in ("mean", "max", "min") for name in PER_SPECTRUM]
)


@njit(cache=True)
def _match(
    ions,
    assignments,
    errors,
    offsets,
    intensities,
    neutral_losses,
    loss_mode,
    assignment_masses,
    ion_masses,
    targets,
    tolerances,
):
    """在 Numba 中把候选离子与每个峰的原子组成匹配结果汇总。

    ``loss_mode`` 为真时匹配的是中性丢失，否则匹配直接离子；峰强度始终保留在
    分母中，所以无法被任何候选解释的峰不会被静默删除。
    """
    n = len(ions)
    p = len(intensities)
    output = np.zeros((n, 7))
    for c in range(n):
        if np.min(ions[c]) < 0:
            continue
        count = 0
        explained = 0.0
        total_error = 0.0
        loss_counts = np.zeros(4)
        for peak in range(p):
            if targets[peak] > ion_masses[c] + tolerances[peak] + 1e-9:
                continue
            best = 2.0
            best_index = -1
            for a in range(offsets[peak], offsets[peak + 1]):
                error = (
                    abs(ion_masses[c] - assignment_masses[a] - targets[peak])
                    / tolerances[peak]
                    if loss_mode[peak]
                    else errors[a]
                )
                if error > 1.0 + 1e-8 or error >= best:
                    continue
                fits = True
                for e in range(12):
                    if assignments[a, e] > ions[c, e]:
                        fits = False
                        break
                if fits:
                    best = error
                    best_index = a
                    # 直接离子候选按质量误差排序，首个满足原子守恒者就是最优匹配。
                    if not loss_mode[peak]:
                        break
            if best_index >= 0:
                count += 1
                explained += intensities[peak]
                total_error += best
                for loss in range(4):
                    fits = True
                    for e in range(12):
                        loss_count = (
                            assignments[best_index, e]
                            if loss_mode[peak]
                            else ions[c, e] - assignments[best_index, e]
                        )
                        if loss_count != neutral_losses[loss, e]:
                            fits = False
                            break
                    if fits:
                        loss_counts[loss] += intensities[peak]
        denominator = max(np.sum(intensities), 1e-12)
        output[c, 0] = count / max(p, 1)
        output[c, 1] = explained / denominator
        output[c, 2] = total_error / count if count else 1.0
        output[c, 3:] = loss_counts / denominator
    return output


def fragment_features(vectors, spectrum, config):
    """为每个候选计算碎片解释率、强度覆盖率和中性丢失比例。

    对无法超过所有候选母体质量的峰跳过组成枚举，但仍保留其强度分母；这既
    避免浪费搜索预算，也保持不同候选之间的特征可比性。金属元素通过加合物
    的离子计数进入匹配，最终仍要求候选离子原子计数非负。
    """
    ions = spectrum.adduct.ion_counts(vectors)
    limits = np.maximum(ions.max(axis=0), 0)
    engine = FormulaEnumerator(
        dict(zip(ELEMENTS, map(int, limits[:10]))), config.max_search_results
    )
    atom_masses = np.r_[MASSES, ATOM_MASS["Na"], ATOM_MASS["K"]]
    ion_masses = ions @ atom_masses
    valid = ions.min(axis=1) >= 0
    lower = float(ion_masses[valid].min()) if valid.any() else 0.0
    upper = float(ion_masses[valid].max()) if valid.any() else 0.0
    assignments = []
    errors = []
    offsets = [0]
    modes = []
    targets = []
    tolerances = []
    for peak in spectrum.mzs:
        tolerance = max(config.fragment_da, peak * config.fragment_ppm * 1e-6)
        target = peak + spectrum.adduct.charge * ELECTRON_MASS
        if target > upper + tolerance + 1e-9:
            # 原子子集不可能超过所有候选母体；跳过枚举但保留强度分母。
            modes.append(False)
            targets.append(target)
            tolerances.append(tolerance)
            offsets.append(offsets[-1])
            continue
        # 当候选母体质量很接近时枚举较小的一侧，并在 _match 中按候选精确复核；
        # 先扩大窗口只是优化搜索，不改变召回边界。
        use_loss = upper - lower <= 0.05 and 0 < upper - target < target
        search_target = (upper + lower) / 2 - target if use_loss else target
        search_tol = tolerance + (upper - lower) / 2 if use_loss else tolerance
        modes.append(use_loss)
        targets.append(target)
        tolerances.append(tolerance)
        peak_hits = []
        peak_errors = []
        for na in range(int(limits[10]) + 1):
            for k in range(int(limits[11]) + 1):
                remaining = search_target - na * ATOM_MASS["Na"] - k * ATOM_MASS["K"]
                hits = engine.search(remaining, search_tol)
                # 中性搜索排除零向量，但纯金属离子或纯中性丢失仍可能合法。
                if (na or k or use_loss) and abs(remaining) <= search_tol:
                    hits = np.vstack([hits, np.zeros(10, dtype=np.int64)])
                if len(hits):
                    peak_hits.append(
                        np.column_stack(
                            [hits, np.full(len(hits), na), np.full(len(hits), k)]
                        )
                    )
                    peak_errors.append(np.abs(hits @ MASSES - remaining) / search_tol)
        if peak_hits:
            hits = np.concatenate(peak_hits)
            err = np.concatenate(peak_errors)
            order = np.argsort(err, kind="stable")
            assignments.append(hits[order])
            errors.append(err[order])
            offsets.append(offsets[-1] + len(hits))
        else:
            offsets.append(offsets[-1])
    losses = np.zeros((4, 12), dtype=np.int64)
    losses[0, [1, 3]] = [2, 1]
    losses[1, [0, 3]] = [1, 1]
    losses[2, [0, 3]] = [1, 2]
    losses[3, [1, 2]] = [3, 1]
    assignments = (
        np.concatenate(assignments).astype(np.int64)
        if assignments
        else np.empty((0, 12), dtype=np.int64)
    )
    errors = np.concatenate(errors) if errors else np.empty(0)
    return _match(
        ions,
        assignments,
        errors,
        np.asarray(offsets),
        spectrum.intensities,
        losses,
        np.asarray(modes),
        assignments @ atom_masses,
        ion_masses,
        np.asarray(targets),
        np.asarray(tolerances),
    )


def extract_features(vectors, spectra, config, errors, support):
    """把逐谱碎片证据和分子式描述聚合成固定列顺序的特征矩阵。

    前四类基础列描述元素计数、DBE、元素比例和质量；其余列对每条谱图的
    特征计算 mean/max/min，确保多谱训练与多谱推理使用完全相同的布局。
    """
    n = len(vectors)
    masses = vectors @ MASSES
    c, h, nit, o, p, s, f, cl, br, i = vectors.T
    base = np.column_stack(
        [
            vectors,
            1 + c + (nit + p - h - f - cl - br - i) / 2,
            h / np.maximum(c, 1),
            (nit + o + p + s + f + cl + br + i) / np.maximum(c, 1),
            masses,
        ]
    )
    all_features = []
    for index, spec in enumerate(spectra):
        fragments = fragment_features(vectors, spec, config)
        ce = spec.collision_energy
        instrument = (spec.instrument or "").lower()
        metadata = [
            np.log1p(len(spec.mzs)),
            float(spec.adduct.charge > 0),
            float(np.mean(ce)) if ce else 0.0,
            min(ce) if ce else 0.0,
            max(ce) if ce else 0.0,
            float(ce is None),
            np.log1p(spec.base_peak_intensity or 0),
            float(spec.base_peak_intensity is None),
            float("timstof" in instrument),
            float("orbitrap" in instrument),
            float(not instrument),
        ]
        per = np.column_stack(
            [
                np.minimum(np.abs(errors[:, index]), 1e6),
                support[:, index],
                fragments,
                np.tile(metadata, (len(vectors), 1)),
            ]
        )
        all_features.append(per)
    stacked = np.stack(all_features)
    features = np.column_stack(
        [base, stacked.mean(axis=0), stacked.max(axis=0), stacked.min(axis=0)]
    )
    assert features.shape[1] == len(FEATURE_NAMES)
    return features
