"""Prevent embedded service credentials from entering the learning overlay."""

from pathlib import Path
import ast
import re
import unittest


ROOT = Path(__file__).resolve().parents[1] / "research/patchtst"
TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".txt"}
PATTERNS = {
    "credential assignment": re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)\s*=\s*"
        r"[\"'][^\"'\n]{8,}[\"']"
    ),
    "literal W&B login key": re.compile(
        r"(?i)wandb\.login\([^\n)]*key\s*=\s*[\"']"
    ),
    "common service token": re.compile(
        r"(?i)(?:sk-[A-Za-z0-9_-]{20,}|gh[opsu]_[A-Za-z0-9]{20,})"
    ),
}


class LearningCredentialTest(unittest.TestCase):
    def test_overlay_contains_no_literal_credentials(self):
        findings = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    findings.append(f"{path.relative_to(ROOT)}:{line}: {label}")
        self.assertEqual(findings, [], "\n".join(findings))

    def test_python_ast_contains_no_credential_literals(self):
        credential_name = re.compile(
            r"(?i)(api[_-]?key|access[_-]?token|secret|password)"
        )
        findings = []
        for path in ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "add_argument":
                        option_names = [
                            item.value for item in node.args
                            if isinstance(item, ast.Constant) and isinstance(item.value, str)
                        ]
                        if any(credential_name.search(name) for name in option_names):
                            for keyword in node.keywords:
                                if (
                                    keyword.arg == "default"
                                    and isinstance(keyword.value, ast.Constant)
                                    and isinstance(keyword.value.value, str)
                                    and keyword.value.value
                                ):
                                    findings.append(f"{path.relative_to(ROOT)}:{node.lineno}: credential CLI default")
                if isinstance(node, ast.keyword) and node.arg and credential_name.search(node.arg):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and node.value.value:
                        findings.append(f"{path.relative_to(ROOT)}:{node.lineno}: credential keyword literal")
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    value = node.value
                    if any(isinstance(target, ast.Name) and credential_name.search(target.id) for target in targets):
                        if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value:
                            findings.append(f"{path.relative_to(ROOT)}:{node.lineno}: credential assignment")
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if (
                            isinstance(key, ast.Constant) and isinstance(key.value, str)
                            and credential_name.search(key.value)
                            and isinstance(value, ast.Constant) and isinstance(value.value, str)
                            and value.value
                        ):
                            findings.append(f"{path.relative_to(ROOT)}:{node.lineno}: credential mapping literal")
        self.assertEqual(findings, [], "\n".join(findings))


if __name__ == "__main__":
    unittest.main()
