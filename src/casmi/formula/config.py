"""分子式推理配置及其兼容性校验。

配置对象会被固化到 prepared 数据和模型元数据中；其稳定哈希用于阻止用不同
的质量窗口、元素边界或特征参数解释已有产物。
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

from .chemistry import ELEMENTS


@dataclass
class FormulaConfig:
    """控制候选搜索、谱图处理、特征提取和训练的版本化参数。"""

    schema_version: int = 1
    precursor_ppm: float = 10.0
    fragment_ppm: float = 10.0
    fragment_da: float = 0.002
    intensity_floor: float = 0.001
    max_peaks: int = 128
    max_candidates: int = 10000
    max_search_results: int = 250000
    seed: int = 42
    # 由指定训练划分推导；prepare 会根据新的训练数据重新固化这个边界。
    element_limits: dict[str, int] = field(
        default_factory=lambda: dict(
            C=155, H=268, N=35, O=65, P=4, S=8, F=44, Cl=8, Br=8, I=8
        )
    )

    def __post_init__(self):
        """拒绝会导致不可复现或无法安全解释产物的配置。"""
        if self.schema_version != 1:
            raise ValueError("Incompatible formula configuration version")
        for name in ("precursor_ppm", "fragment_ppm", "fragment_da"):
            v = getattr(self, name)
            if not isinstance(v, (float, int)) or not 0 < v < float("inf"):
                raise ValueError(f"{name} must be positive and finite")
        if self.precursor_ppm >= 1e6:
            raise ValueError("precursor_ppm must be below 1000000")
        if not 0 <= self.intensity_floor <= 1:
            raise ValueError("intensity_floor must be in [0,1]")
        for name in ("max_peaks", "max_candidates", "max_search_results"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.element_limits or any(
            e not in ELEMENTS or type(n) is not int or n < 0
            for e, n in self.element_limits.items()
        ):
            raise ValueError("Invalid element limits")

    def to_dict(self):
        """返回可序列化的完整配置，供 manifest 和模型元数据保存。"""
        return asdict(self)

    @property
    def version(self):
        """返回排序稳定的配置内容哈希，用作配置兼容性标识。"""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]

    @classmethod
    def load(cls, path):
        """加载裸配置或包含 ``formula`` 包装对象的 JSON 文件。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data.get("formula", data))
