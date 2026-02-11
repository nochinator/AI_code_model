import unittest

from codex_style_assistant import CodexStyleAssistant


class TestCodexStyleAssistant(unittest.TestCase):
    def setUp(self) -> None:
        self.assistant = CodexStyleAssistant()

    def test_spec_summary_contains_expected_fields(self) -> None:
        summary = self.assistant.summarize_spec()
        self.assertEqual(summary["name"], "Codex-Style Model")
        self.assertGreaterEqual(summary["capability_count"], 3)
        self.assertIn("Base Model", summary["architecture_layers"])

    def test_generate_python_code(self) -> None:
        code = self.assistant.generate_code("build a parser", "python")
        self.assertIn("def solve", code)
        self.assertIn("build a parser", code)

    def test_code_understanding_detects_functions(self) -> None:
        source = """
import os

def run():
    pass
"""
        understanding = self.assistant.understand_code(source)
        self.assertEqual(understanding["non_empty_lines"], 3)
        self.assertTrue(any(item.startswith("def run") for item in understanding["detected_functions"]))
        self.assertIn("import os", understanding["detected_imports"])

    def test_plan_and_execute(self) -> None:
        steps = self.assistant.plan_task("create auth service")
        result = self.assistant.execute_task(steps)
        self.assertTrue(result.success)
        self.assertEqual(result.completed_steps, result.total_steps)
        self.assertGreaterEqual(len(result.events), len(steps))


if __name__ == "__main__":
    unittest.main()
