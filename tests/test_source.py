"""Lightweight checks that do not require TensorFlow or a dataset."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "effiaul_four_class.py"


class SourceChecks(unittest.TestCase):
    def test_source_parses(self):
        ast.parse(SOURCE.read_text(encoding="utf-8"))

    def test_expected_entry_points_exist(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        required = {"main", "scan_master_dataset", "create_fixed_splits",
                    "calculate_influence_scores", "strategy_selector",
                    "run_effiaul_update", "run_experiment"}
        self.assertFalse(required - functions, required - functions)

    def test_no_embedded_absolute_dataset_path(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("EFFIAUL_DATA_DIR", source)
        self.assertIn("EFFIAUL_RESULTS_DIR", source)


if __name__ == "__main__":
    unittest.main()
