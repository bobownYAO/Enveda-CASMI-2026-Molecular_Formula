"""最小公共 API 示例：用两条谱图生成一个分子的分子式候选。

运行：``python examples/predict_formula.py``。示例故意使用 baseline，不依赖训练
模型，便于先验证环境、输入字段和 JSON 输出格式。
"""
from casmi.formula import FormulaPredictor


def main():
    predictor=FormulaPredictor.baseline()
    prediction=predictor.predict([
        {'precursor_mz':181.07066456,'adduct':'[M+H]+',
         'ms2_mzs':[163.06009988,145.04953520],
         'ms2_normalized_intensities':[1.,.4]},
        {'precursor_mz':179.05611163,'adduct':'[M-H]-',
         'ms2_mzs':[161.04554695],
         'ms2_normalized_intensities':[1.]},
    ],molecule_id='glucose-example',top_k=5)
    print(prediction.to_json())


if __name__=='__main__':main()
