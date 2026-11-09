import compileall
import pathlib
import re
import unittest

bad_chars = ["\u00a0", "\u202f", "\u200b", "\ufeff"]
pattern = re.compile("|".join(bad_chars))
root = pathlib.Path("d:/GitFile/engineer_work")


class TestSourceHygiene(unittest.TestCase):
    def test_no_non_ascii_whitespace(self):
        for path in root.rglob("*"):
            if any(part in {".venv", "venv", ".git", ".pytest_cache", ".ruff_cache", "__pycache__"} for part in path.parts):
                continue
            if path.suffix.lower() in {".py", ".yaml", ".yml", ".html", ".md"}:
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                self.assertIsNone(pattern.search(text), f"Bad whitespace in {path}")

    def test_compile_all(self):
        # compileall returns True if no errors
        self.assertTrue(compileall.compile_dir("src", quiet=1))
        self.assertTrue(compileall.compile_dir("web_app", quiet=1))
        self.assertTrue(compileall.compile_dir("tests", quiet=1))


if __name__ == "__main__":
    unittest.main()
