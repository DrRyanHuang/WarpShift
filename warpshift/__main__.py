"""WarpShift launch and control CLI. Never downloads models or dependencies."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def serve(args: argparse.Namespace) -> None:
    source_root = ROOT / "sglang" / "python"
    if not (source_root / "sglang").is_dir():
        raise RuntimeError("Initialize the sglang submodule before serving")
    patch = ROOT / "patches/sglang-tp-disk-update.patch"
    apply_command = ["git", "-C", str(ROOT / "sglang"), "apply"]
    # Check only our patch; unrelated upstream edits remain allowed.
    patched = subprocess.run([*apply_command, "--reverse", "--check", str(patch)], capture_output=True)
    if patched.returncode:
        applied = subprocess.run([*apply_command, str(patch)], capture_output=True, text=True)
        if (
            applied.returncode
            and subprocess.run([*apply_command, "--reverse", "--check", str(patch)], capture_output=True).returncode
        ):
            raise RuntimeError(f"Cannot apply SGLang TP disk-update patch: {applied.stderr}")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(source_root), str(ROOT), os.environ.get("PYTHONPATH", "")])
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source_root))
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.entrypoints import http_server
    from sglang.srt.utils import kill_process_tree
    from warpshift.server import ModelSwitcher, install
    from warpshift.weights import LOADER, ModelWeightClient, _validate_models

    models = _validate_models(json.loads(Path(args.models).read_text()))
    if args.source == "delta" and len(models) != 2:
        raise ValueError("Delta source currently requires exactly two models")
    initial = args.initial or next(iter(models))
    if initial not in models:
        raise ValueError("Initial model not in manifest")
    native_args = args.sglang_args[1:] if args.sglang_args[:1] == ["--"] else args.sglang_args
    server_args = prepare_server_args(
        [
            "--model-path",
            models[initial],
            "--host",
            "127.0.0.1",
            "--dtype",
            "bfloat16",
            "--skip-server-warmup",
            "--custom-weight-loader",
            LOADER,
            *native_args,
        ]
    )
    if str(Path(server_args.model_path).resolve()) != models[initial]:
        raise ValueError("Choose the checkpoint with --initial")
    # One supported deployment contract, checked once at launch.
    for name in ("dp_size", "pp_size", "ep_size", "nnodes", "tokenizer_worker_num"):
        if getattr(server_args, name, 1) != 1:
            raise ValueError(f"WarpShift requires {name}=1")
    for name in (
        "use_ray",
        "encoder_only",
        "smg_grpc_mode",
        "grpc_mode",
        "grpc_port",
        "ssl_keyfile",
        "fastapi_root_path",
        "enable_lora",
        "speculative_algorithm",
        "quantization",
        "enable_dp_attention",
    ):
        if getattr(server_args, name, None):
            raise ValueError(f"WarpShift does not support {name}")
    if server_args.dtype not in ("bfloat16", "bf16") or server_args.weight_cache_mode != "off":
        raise ValueError("Require BF16 and weight_cache_mode=off")
    if LOADER not in server_args.custom_weight_loader:
        raise ValueError(f"Custom loaders must include {LOADER}")
    host = server_args.host
    host = "127.0.0.1" if host == "0.0.0.0" else ("::1" if host == "::" else host)
    url = f"http://{'[' + host + ']' if ':' in host else host}:{server_args.port}"
    client = ModelWeightClient(
        url, server_args.tp_size, args.timeout, server_args.admin_api_key or server_args.api_key
    )
    client.current = initial  # SGLang loads this checkpoint during startup.
    install(http_server.app, ModelSwitcher(client, models, args.source, initial, args.timeout))
    print(
        json.dumps(
            {
                "event": "launch",
                "url": url,
                "source": args.source,
                "sglang_path": str(source_root),
                "models": list(models),
            }
        ),
        flush=True,
    )
    try:
        http_server.launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve_parser = subcommands.add_parser("serve")
    serve_parser.add_argument("--models", required=True, help="JSON: model ID -> local checkpoint directory")
    serve_parser.add_argument("--initial")
    serve_parser.add_argument(
        "--source",
        choices=("disk", "pageable", "pinned", "delta", "page_cache"),
        default="pinned",
        help="Weight source; page_cache is a legacy alias for disk (file reads may hit OS page cache)",
    )
    serve_parser.add_argument("--timeout", type=float, default=1200)
    serve_parser.add_argument("sglang_args", nargs=argparse.REMAINDER)
    for name in ("prepare", "switch", "status"):
        control_parser = subcommands.add_parser(name)
        control_parser.add_argument("--url", default="http://127.0.0.1:30000")
        control_parser.add_argument("--timeout", type=float, default=1200)
        control_parser.add_argument("--api-key", default=os.environ.get("WARPSHIFT_API_KEY"))
        if name == "switch":
            control_parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.command == "serve":
        args.source = "disk" if args.source == "page_cache" else args.source
        serve(args)
        return
    body = {"model": args.model} if args.command == "switch" else {}
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    http_request = Request(
        args.url.rstrip("/") + "/warpshift/" + args.command,
        data=None if args.command == "status" else json.dumps(body).encode(),
        headers=headers,
    )
    with urlopen(http_request, timeout=args.timeout) as response:
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
