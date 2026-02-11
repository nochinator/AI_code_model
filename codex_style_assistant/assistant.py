"""Core assistant implementation aligned to the model specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from .spec import ModelSpecification, default_specification


@dataclass
class TaskStep:
    """Represents one executable step in a multi-step task."""

    description: str
    command: Optional[str] = None


@dataclass
class ExecutionEvent:
    """Log event for traceability."""

    timestamp: str
    level: str
    message: str


@dataclass
class ExecutionResult:
    """Outcome of task execution."""

    completed_steps: int
    total_steps: int
    events: List[ExecutionEvent] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.completed_steps == self.total_steps


class CodexStyleAssistant:
    """A practical, local implementation of the provided model specification.

    This implementation intentionally uses deterministic template-based generation
    and no external APIs so it can run in constrained environments.
    """

    def __init__(self, specification: Optional[ModelSpecification] = None) -> None:
        self.specification = specification or default_specification()
        self._events: List[ExecutionEvent] = []

    def _log(self, level: str, message: str) -> None:
        self._events.append(
            ExecutionEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                level=level,
                message=message,
            )
        )

    def summarize_spec(self) -> Dict[str, object]:
        """Expose a machine-readable summary of active model properties."""

        return {
            "name": self.specification.name,
            "purpose": self.specification.purpose,
            "primary_goal": self.specification.primary_goal,
            "capability_count": len(self.specification.capabilities),
            "architecture_layers": [layer.name for layer in self.specification.architecture],
            "safety_controls": list(self.specification.safety_controls),
        }

    def generate_code(self, prompt: str, language: str = "python") -> str:
        """Natural-language to code synthesis (template baseline)."""

        self._log("info", f"Generating {language} code from prompt")
        language_key = language.strip().lower()

        if language_key == "python":
            return (
                "def solve():\n"
                f"    \"\"\"Auto-generated solution for: {prompt}\"\"\"\n"
                "    # TODO: implement domain logic\n"
                "    return {'status': 'not_implemented', 'prompt': "
                + repr(prompt)
                + "}\n"
            )
        if language_key in {"javascript", "js"}:
            return (
                "function solve() {\n"
                f"  // Auto-generated solution for: {prompt}\n"
                "  // TODO: implement domain logic\n"
                "  return { status: 'not_implemented', prompt: "
                + repr(prompt)
                + " };\n"
                "}\n"
            )
        return (
            f"-- Auto-generated pseudo-code for: {prompt}\n"
            "-- TODO: implement domain logic\n"
        )

    def understand_code(self, source: str) -> Dict[str, object]:
        """Provide lightweight structural understanding of source code."""

        lines = [line for line in source.splitlines() if line.strip()]
        functions = [line.strip() for line in lines if line.strip().startswith(("def ", "function "))]
        imports = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
        self._log("info", "Performed structural code understanding")
        return {
            "non_empty_lines": len(lines),
            "detected_functions": functions,
            "detected_imports": imports,
        }

    def plan_task(self, request: str) -> List[TaskStep]:
        """Create a simple multi-step plan for an engineering request."""

        self._log("info", f"Planning task: {request}")
        return [
            TaskStep("Analyze request and constraints"),
            TaskStep("Generate initial implementation"),
            TaskStep("Run validation checks"),
            TaskStep("Summarize outputs and next actions"),
        ]

    def execute_task(self, steps: Iterable[TaskStep]) -> ExecutionResult:
        """Execute planned steps with traceable event logs."""

        self._events.clear()
        planned = list(steps)
        for index, step in enumerate(planned, start=1):
            self._log("info", f"Step {index}/{len(planned)}: {step.description}")
            if step.command:
                self._log("warning", "Command execution skipped in safe local mode")

        self._log("info", "Task execution completed")
        return ExecutionResult(
            completed_steps=len(planned),
            total_steps=len(planned),
            events=list(self._events),
        )
