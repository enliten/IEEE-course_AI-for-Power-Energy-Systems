import unittest
import warnings
from pathlib import Path

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestNotebookMarkdownIntegrity(unittest.TestCase):
    def test_markdown_cells_do_not_contain_latex_escape_damage(self):
        notebook_paths = sorted(
            path
            for path in REPO_ROOT.glob("**/*.ipynb")
            if ".ipynb_checkpoints" not in path.parts
        )
        self.assertTrue(notebook_paths, "Expected at least one notebook in the repository")

        disallowed_controls = {
            "\f": "form feed, often caused by writing \\frac as \\f + rac",
            "\r": "carriage return, often caused by writing \\right as \\r + ight",
            "\t": "tab, often caused by writing \\text as \\t + ext",
        }

        failures = []
        for path in notebook_paths:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=nbformat.validator.MissingIDFieldWarning)
                notebook = nbformat.read(path, as_version=4)
            for cell_index, cell in enumerate(notebook.cells):
                if cell.get("cell_type") != "markdown":
                    continue
                source = "".join(cell.get("source", ""))
                for control_char, description in disallowed_controls.items():
                    if control_char in source:
                        failures.append(f"{path.relative_to(REPO_ROOT)} cell {cell_index}: {description}")

        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
