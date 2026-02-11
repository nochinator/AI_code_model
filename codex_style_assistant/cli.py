"""CLI interface for the Codex-style assistant implementation."""

import argparse
import json

from .assistant import CodexStyleAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codex-style multimodal coding assistant")
    parser.add_argument("request", nargs="?", default="implement hello world", help="Task request")
    parser.add_argument("--language", default="python", help="Code generation language")
    parser.add_argument("--show-spec", action="store_true", help="Print assistant specification summary")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    assistant = CodexStyleAssistant()

    if args.show_spec:
        print(json.dumps(assistant.summarize_spec(), indent=2))

    plan = assistant.plan_task(args.request)
    result = assistant.execute_task(plan)
    generated = assistant.generate_code(args.request, args.language)

    print("\n=== PLAN ===")
    for i, step in enumerate(plan, start=1):
        print(f"{i}. {step.description}")

    print("\n=== RESULT ===")
    print(f"success={result.success} ({result.completed_steps}/{result.total_steps})")

    print("\n=== GENERATED CODE ===")
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
