"""Export only the supported CRNN, explicitly fixing input shape and CPU device."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("-c", "--config", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.engine))
    import paddle
    import paddle.jit.dy2static.utils as static_utils
    from ppocr.modeling.architectures import build_model

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["Architecture"]["algorithm"] != "CRNN":
        raise ValueError("Only CRNN export is supported")
    paddle.set_device("cpu")
    # Paddle 2.5 hardcodes ~/.cache for generated Python. Keep it inside this run.
    cache = args.config.parent / "static_cache"
    cache.mkdir(parents=True, exist_ok=True)
    static_utils.get_temp_dir = lambda: str(cache)
    characters = Path(config["Global"]["character_dict_path"]).read_text(encoding="utf-8-sig").splitlines()
    config["Architecture"]["Head"]["out_channels"] = len(characters) + 1
    model = build_model(config["Architecture"])
    checkpoint = config["Global"]["pretrained_model"]
    state = paddle.load(checkpoint + ".pdparams")
    expected = model.state_dict()
    if set(state) != set(expected) or any(list(state[key].shape) != list(value.shape) for key, value in expected.items()):
        raise ValueError("Checkpoint architecture/dictionary mismatch; refusing partial export")
    model.set_state_dict(state)
    model.eval()
    output = Path(config["Global"]["save_inference_dir"])
    output.mkdir(parents=True, exist_ok=True)
    model = paddle.jit.to_static(model, input_spec=[paddle.static.InputSpec([None, 3, 48, 256], "float32")])
    paddle.jit.save(model, str(output / "inference"))
    (output / "characters.txt").write_text("\n".join(characters) + "\n", encoding="utf-8")
    print("Exported CRNN [N, 3, 48, 256]:", output)


if __name__ == "__main__":
    main()
