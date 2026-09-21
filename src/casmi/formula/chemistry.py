"""质谱化学基础：单同位素质量、分子式向量和受支持的加合物。

本模块把字符串分子式转换成固定顺序的元素计数向量，并负责在中性分子与
带电离子之间换算质量。候选生成、谱图特征和预测器都依赖这里的元素顺序，
因此修改 ``ELEMENTS`` 或加合物表时必须同步检查模型特征和产物兼容性。
"""

from dataclasses import dataclass
import re

import numpy as np

# 这个顺序是整个项目的向量协议：数据、特征、搜索和模型都按它解释计数。
ELEMENTS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
ATOM_MASS = dict(
    zip(
        ELEMENTS,
        (
            12.0,
            1.00782503223,
            14.00307400443,
            15.99491461957,
            30.97376199842,
            31.9720711744,
            18.99840316273,
            34.968852682,
            78.9183376,
            126.9044719,
        ),
    )
)
ATOM_MASS.update(Na=22.9897692820, K=38.9637064864)
MASSES = np.array([ATOM_MASS[e] for e in ELEMENTS])
ELECTRON_MASS = 0.000548579909065
TOKEN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")


def parse_formula(formula: str) -> np.ndarray:
    """解析中性分子式，返回与 ``ELEMENTS`` 对齐的整数计数向量。

    电荷、未知元素、小数计数和空分子式都会被拒绝；这里不尝试“修正”输入，
    以免把带电结构标签或域外元素静默地当成中性分子式。
    """
    if not isinstance(formula, str) or not formula:
        raise ValueError("A non-empty neutral molecular formula is required")
    counts = np.zeros(len(ELEMENTS), dtype=np.int64)
    end = 0
    for m in TOKEN.finditer(formula):
        if m.start() != end or m[1] not in ELEMENTS:
            raise ValueError(f"Unsupported neutral formula: {formula}")
        counts[ELEMENTS.index(m[1])] += int(m[2] or 1)
        end = m.end()
    if end != len(formula) or not counts.any():
        raise ValueError(f"Unsupported neutral formula: {formula}")
    return counts


def format_formula(counts) -> str:
    """按稳定的 Hill 风格顺序把元素计数格式化成分子式字符串。"""
    order = (
        ["C", "H"] + sorted(set(ELEMENTS) - {"C", "H"})
        if counts[0]
        else sorted(ELEMENTS)
    )
    return "".join(
        e
        + (str(int(counts[ELEMENTS.index(e)])) if counts[ELEMENTS.index(e)] > 1 else "")
        for e in order
        if counts[ELEMENTS.index(e)] > 0
    )


def exact_mass(counts) -> float:
    """用项目固定的单同位素原子质量计算中性分子的精确质量（Da）。"""
    return float(np.dot(counts, MASSES))


def dbe(counts) -> float:
    """计算用于排序特征的双键当量；它不是本版本的硬召回过滤规则。"""
    c, h, n, o, p, s, f, cl, br, i = counts
    return float(1 + c + (n + p - h - f - cl - br - i) / 2)


@dataclass(frozen=True)
class Adduct:
    """描述一个明确支持的离子加合物及其质量/元素变换规则。

    ``delta`` 表示相对中性分子的原子计数变化，``charge`` 用于 m/z 换算，
    ``multiplier`` 保留多聚体接口。负离子产生的负元素计数会由调用方判为无效。
    """

    name: str
    delta: dict[str, int]
    charge: int
    multiplier: int = 1

    @property
    def shift(self):
        """返回加合物和电子质量造成的 m/z 平移量（Da）。"""
        return (
            sum(ATOM_MASS[e] * n for e, n in self.delta.items())
            - self.charge * ELECTRON_MASS
        )

    def mz(self, neutral_mass):
        """把中性质量换算成该加合物的观测 m/z。"""
        return (self.multiplier * neutral_mass + self.shift) / abs(self.charge)

    def neutral_mass(self, mz):
        """把观测 m/z 反解为中性质量；调用方负责检查输入范围。"""
        return (mz * abs(self.charge) - self.shift) / self.multiplier

    def ion_counts(self, counts):
        """把中性元素向量扩展成含 Na/K 的离子原子计数向量。"""
        counts = np.asarray(counts)
        result = np.zeros(counts.shape[:-1] + (12,), dtype=np.int64)
        result[..., :10] = counts * self.multiplier
        for e, n in self.delta.items():
            result[..., (ELEMENTS + ("Na", "K")).index(e)] += n
        return result


# 只有这里列出的加合物会进入训练和推理，新增项需要补质量与输入测试。
ADDUCTS = {
    a.name: a
    for a in [
        Adduct("[M+H]+", {"H": 1}, 1),
        Adduct("[M+NH4]+", {"N": 1, "H": 4}, 1),
        Adduct("[M-H2O+H]+", {"H": -1, "O": -1}, 1),
        Adduct("[M-2H2O+H]+", {"H": -3, "O": -2}, 1),
        Adduct("[M+Na]+", {"Na": 1}, 1),
        Adduct("[M+K]+", {"K": 1}, 1),
        Adduct("[M-H]-", {"H": -1}, -1),
        Adduct("[M-H2O-H]-", {"H": -3, "O": -1}, -1),
        Adduct("[M+CH2O2-H]-", {"C": 1, "H": 1, "O": 2}, -1),
        Adduct("[M+Cl]-", {"Cl": 1}, -1),
    ]
}
