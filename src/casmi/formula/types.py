"""公共预测结果类型及 JSON 序列化协议。"""

from dataclasses import asdict, dataclass, field
import json


@dataclass
class FormulaCandidate:
    """一个已排序的分子式候选及其质量误差和谱图支持数。"""

    rank: int
    formula: str
    exact_mass: float
    score: float
    mass_errors_ppm: dict[str, float]
    supporting_spectra: int


@dataclass
class FormulaPrediction:
    """一次分子级预测的状态、候选、警告、版本和诊断信息。

    ``status`` 区分正常结果、候选截断、无候选和输入/资源失败；失败记录仍可
    通过 JSONL 输出，方便批量任务统计而不丢失分子标识。
    """

    molecule_id: str
    status: str
    candidates: list[FormulaCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_version: str = "baseline-v1"
    config_version: str = ""
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        """转换为只含标准 Python 对象的字典。"""
        return asdict(self)

    def to_json(self):
        """生成保留中文且拒绝 NaN 的稳定 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
