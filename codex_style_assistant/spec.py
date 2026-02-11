"""System specification objects for a Codex-style multimodal coding assistant."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Capability:
    name: str
    details: List[str]


@dataclass(frozen=True)
class ArchitectureLayer:
    name: str
    details: List[str]


@dataclass(frozen=True)
class ModelSpecification:
    name: str
    purpose: str
    primary_goal: str
    capabilities: List[Capability] = field(default_factory=list)
    architecture: List[ArchitectureLayer] = field(default_factory=list)
    training_data_sources: List[str] = field(default_factory=list)
    objective_functions: List[str] = field(default_factory=list)
    benchmarks: List[str] = field(default_factory=list)
    safety_controls: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    workflow_integrations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    future_directions: List[str] = field(default_factory=list)


def default_specification() -> ModelSpecification:
    """Return a concrete representation of MODEL_SPECIFICATION.md."""

    return ModelSpecification(
        name="Codex-Style Model",
        purpose=(
            "An agentic, multimodal AI assistant optimized for software engineering "
            "workflows, reasoning tasks, and tool interaction."
        ),
        primary_goal=(
            "Generate, understand, debug, and reason about code across languages "
            "with autonomous task execution and workflow integration."
        ),
        capabilities=[
            Capability(
                name="Code Generation & Understanding",
                details=[
                    "Natural language to code synthesis",
                    "Multi-language support",
                    "Context-aware completion and refactoring",
                    "Deep code logic comprehension",
                ],
            ),
            Capability(
                name="Agentic Task Execution",
                details=[
                    "Long-running multi-step coding support",
                    "Autonomous command execution and iteration",
                    "Interactive steering",
                    "Traceable progress logging",
                ],
            ),
            Capability(
                name="Multimodal Inputs",
                details=[
                    "Handles text, code, and image references",
                    "Supports UI/visual-debug context",
                ],
            ),
        ],
        architecture=[
            ArchitectureLayer(
                name="Base Model",
                details=[
                    "Transformer autoregressive architecture",
                    "Code-aware tokenization",
                    "Large mixed code+text training corpora",
                ],
            ),
            ArchitectureLayer(
                name="Agentic Reasoning Layer",
                details=[
                    "Adaptive reasoning depth",
                    "Dynamic context compaction",
                    "Plan/execute/validate modules",
                ],
            ),
            ArchitectureLayer(
                name="Token Handling",
                details=[
                    "Large context windows",
                    "Adaptive token budgeting",
                ],
            ),
        ],
        training_data_sources=[
            "Public code repositories",
            "Forums and documentation",
            "NL↔code intent datasets",
            "Benchmark task data",
        ],
        objective_functions=[
            "Autoregressive language modeling",
            "Code and reasoning fine-tuning",
            "Human-feedback-based alignment",
        ],
        benchmarks=[
            "SWE-Bench Pro",
            "Terminal-Bench",
            "Agentic task success",
            "Coding correctness",
        ],
        safety_controls=[
            "Sandboxed command execution",
            "Malicious request refusal policy hooks",
            "Cybersecurity-sensitive safety hardening",
        ],
        interfaces=["CLI", "IDE", "Web/Desktop", "API"],
        workflow_integrations=[
            "Runs in development environments",
            "Logs and outputs are reviewable",
            "Supports user-gated commit flows",
        ],
        limitations=[
            "Requires human code review",
            "Quality varies by language/task",
            "Resource and token constraints",
            "Modality support may vary by deployment",
        ],
        future_directions=[
            "Broader reasoning with larger models",
            "Improved multimodal robustness",
            "Stronger tool-use verification loops",
        ],
    )
