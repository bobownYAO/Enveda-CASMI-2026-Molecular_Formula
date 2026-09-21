"""``casmi formula`` 命令行编排器。

CLI 只负责解析参数、合并配置、调用领域 API 和写出 JSON；算法规则仍由
``FormulaConfig``、``FormulaPredictor``、``data`` 与 ``training`` 模块负责。
"""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

from .formula import FormulaConfig, FormulaPredictor
from .formula.data import audit, prepare, write_json
from .formula.training import train, evaluate


def main(argv=None):
    """解析命令并执行审计、准备、训练、评估或预测流程。

    命令行路径优先于配置文件路径；训练和模型预测会额外校验保存的配置版本，
    防止用户用不同参数解释已有 prepared 数据或模型。
    """
    parser = argparse.ArgumentParser(prog="casmi")
    root = parser.add_subparsers(dest="module", required=True)
    formula = root.add_parser("formula").add_subparsers(dest="action", required=True)
    for name in ("audit", "prepare", "train", "evaluate", "predict"):
        command = formula.add_parser(name)
        command.add_argument(
            "--config", help="JSON config; formula settings plus optional paths"
        )
        command.add_argument("--input")
        command.add_argument("--output")
        if name in ("predict", "evaluate"):
            command.add_argument("--model")
        if name == "predict":
            command.add_argument("--top-k", type=int, default=25)
            command.add_argument("--workers", type=int, default=1)
        if name in ("train", "evaluate"):
            command.add_argument("--max-molecules", type=int)
        if name == "train":
            command.add_argument("--max-validation-molecules", type=int)
            command.add_argument("--n-estimators", type=int, default=1000)
        if name == "evaluate":
            command.add_argument(
                "--split", choices=["train", "validation", "test"], default="test"
            )
    args = parser.parse_args(argv)
    try:
        # 配置文件可同时承载路径和 formula 参数，但未知字段必须显式报错。
        settings = (
            json.loads(Path(args.config).read_text(encoding="utf-8"))
            if args.config
            else {}
        )
        path = args.input or settings.get("input")
        output = args.output or settings.get("output")
        if not path:
            raise ValueError("--input or input in config is required")
        if output and Path(path).resolve() == Path(output).resolve():
            raise ValueError("Output must not overwrite the input file")
        field_names = {f.name for f in fields(FormulaConfig)}
        unknown = set(settings) - field_names - {"formula", "input", "output", "model"}
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        has_formula = "formula" in settings or bool(field_names & settings.keys())
        config_data = settings.get(
            "formula",
            {
                k: v
                for k, v in settings.items()
                if k not in ("input", "output", "model")
            },
        )
        config = FormulaConfig(**config_data) if has_formula else FormulaConfig()
        if args.action in ("train", "evaluate") and has_formula:
            model = getattr(args, "model", None) or settings.get("model")
            expected = (
                FormulaPredictor.load(model).config
                if model
                else FormulaConfig.load(Path(path) / "config.json")
            )
            if config.version != expected.version:
                raise ValueError("Config differs from saved preparation/model config")
        # 下面的分派保持 CLI 薄层：各阶段返回机器可读字典或逐分子预测对象。
        if args.action == "audit":
            result = audit(path)
        elif args.action == "prepare":
            if not output:
                raise ValueError("--output is required")
            result = prepare(path, output, config)
        elif args.action == "train":
            if not output:
                raise ValueError("--output is required")
            result = train(
                path,
                output,
                args.max_molecules,
                args.max_validation_molecules,
                args.n_estimators,
            )
        elif args.action == "evaluate":
            result = evaluate(
                path,
                args.model or settings.get("model"),
                args.split,
                args.max_molecules,
            )
        else:
            if not output:
                raise ValueError("--output is required")
            model = args.model or settings.get("model")
            predictor = (
                FormulaPredictor.load(model)
                if model
                else FormulaPredictor.baseline(config)
            )
            if model and has_formula and predictor.config.version != config.version:
                raise ValueError(
                    "Prediction config differs from the saved model config"
                )
            dest = Path(output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            # JSONL 每个分子一行，失败分子也写出，便于批量结果与诊断对齐。
            with dest.open("w", encoding="utf-8") as stream:
                for prediction in predictor.iter_predict_parquet(
                    path, args.top_k, args.workers
                ):
                    stream.write(prediction.to_json() + "\n")
                    stream.flush()
                    count += 1
                    if count % 25 == 0:
                        print(
                            f"predicted {count} molecules", file=sys.stderr, flush=True
                        )
            print(
                json.dumps(
                    {"molecules": count, "output": str(dest)}, ensure_ascii=False
                )
            )
            return 0
        if output and args.action in ("audit", "evaluate"):
            write_json(output, result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        print(f"casmi: {error}", file=sys.stderr)
        return 2
