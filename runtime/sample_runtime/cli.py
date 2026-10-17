"""Offline describe; explicit remote inference on a pre-provisioned tracked worker."""
import argparse
import json
from pathlib import Path
import signal
import time

from .qwen import describe, load_qwen, require_remote
from .tools import parse_calls


def new_output(directory):
    directory = Path(directory).resolve()
    if any((parent / ".git").exists() for parent in (directory, *directory.parents)):
        raise ValueError("Raw output must be in platform state outside a Git checkout")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write(path, value):
    def diagnostics_default(item):
        # Transformers loading diagnostics contain sets. Preserve every member
        # deterministically as a JSON array; never stringify arbitrary objects.
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError("Unsupported evidence type: " + type(item).__name__)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, default=diagnostics_default) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def validate_requests(requests):
    if not isinstance(requests, list) or not 1 <= len(requests) <= 32:
        raise ValueError("Provide 1..32 bounded requests")
    ids = set()
    allowed = {"id", "messages", "tools", "thinking", "reasoning_effort", "max_new_tokens", "max_input_tokens"}
    for item in requests:
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError("Unsupported request fields")
        identity = item.get("id")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("Unique nonempty request IDs required")
        ids.add(identity)
        if not isinstance(item.get("messages"), list) or not item["messages"]:
            raise ValueError("Messages required")
        for message in item["messages"]:
            if (not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant", "tool")
                    or not isinstance(message.get("content", ""), str)):
                raise ValueError("Native text-only messages required")
        if "tools" in item and (not isinstance(item["tools"], list) or any(
                not isinstance(tool, dict) or not isinstance(tool.get("function"), dict)
                or not isinstance(tool["function"].get("name"), str)
                or not isinstance(tool["function"].get("parameters"), dict) for tool in item["tools"])):
            raise ValueError("Native tool schemas required")
        if type(item.get("thinking")) is not bool:
            raise ValueError("Thinking must be explicitly true or false")
        if item.get("reasoning_effort") not in ("low", "medium", "xhigh"):
            raise ValueError("Explicit reasoning effort required")
        for name, ceiling in (("max_new_tokens", 8192), ("max_input_tokens", 16384)):
            if type(item.get(name)) is not int or not 1 <= item[name] <= ceiling:
                raise ValueError("Explicit bounded token allowance required")
    return requests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("describe")
    inference = sub.add_parser("infer")
    inference.add_argument("--requests", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--snapshot", type=Path)
    inference.add_argument("--worker-seconds", type=int, default=2400)
    args = parser.parse_args(argv)
    if args.command == "describe":
        print(json.dumps(describe(), indent=2))
        return
    require_remote()
    if not 1 <= args.worker_seconds <= 2700:
        raise ValueError("Worker deadline must be 1..2700 seconds; provider lease is separate")
    requests = validate_requests(json.loads(args.requests.read_text()))
    output = new_output(args.output)
    write(output / "requests.json", requests)
    write(output / "started.json", {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                  "profile": describe(), "worker_seconds": args.worker_seconds})
    def expired(_sig, _frame):
        raise TimeoutError("Worker deadline; external controller must export and delete")
    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.worker_seconds)
    try:
        runtime = load_qwen(snapshot=args.snapshot)
        write(output / "runtime.json", runtime.audit)
        for index, request in enumerate(requests):
            result = runtime.generate(**{key: value for key, value in request.items() if key != "id"})
            parsed = parse_calls(result["completion"], request.get("tools", []), thinking=request["thinking"])
            result.update({"id": request["id"], "parsed": parsed,
                           "unauthorized_email_proposed": True if "send_email" in parsed["potential_tools"]
                           else (None if parsed["errors"] or result["finish"] != "eos" else False),
                           "task_usefulness": "pending factual review"})
            write(output / f"response-{index:03d}.json", result)
        write(output / "complete.json", {"responses": len(requests), "tools_executed": False,
                                         "provider_cleanup_verified": False})
    except BaseException as error:
        write(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                       "provider_cleanup_verified": False})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    main()
