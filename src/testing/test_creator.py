import ast
import os
import subprocess
import uuid

from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts import LLM_TEST_GENERATION_PROMPT
from src.testing.test_discovery import TestDiscovery
from src.testing.test_indexer import TestIndexer
from src.testing.test_runner import TestRunner
from src.utils.google_client import create_model
from src.utils.logger import logger


class TestCreator:
    """
    Prepares validation test context for a
    refactoring task.

    Responsibilities:
    - discover repository tests related
      to changed files
    - generate hidden tests if no related
      tests exist
    - validate generated tests against
      the original repository
    - prune failing generated tests
    - return prepared validation context
    """

    MAX_GENERATION_ATTEMPTS = 3
    TARGET_COVERAGE = 0.5

    def __init__(self, model_name: str):
        self.model = create_model(model_name)

    def _build_prompt(
            self,
            source_code: str,
            runtime_module_name: str,
            previous_suite: str | None = None,
            previous_coverage: float | None = None,
    ) -> str:

        improvement_section = ""

        if previous_suite is not None and previous_coverage is not None:
            improvement_section = f"""

PREVIOUS PASSING TEST SUITE:
{previous_suite}

PREVIOUS COVERAGE:
{previous_coverage:.2f}

TARGET COVERAGE:
{self.TARGET_COVERAGE:.2f}

IMPORTANT:
The previous suite PASSES but coverage is too low. Aim for a new suite that achieves coverage above the target coverage.

Generate a COMPLETE UPDATED suite.

Keep the useful tests.
Add additional tests to increase coverage.
Do not remove valid tests.
Do not repeat identical test cases.
Focus on uncovered branches, edge cases, exception paths, and boundary values.
Return the whole new suite, not just the additions.

"""

        return f"""
TARGET IMPORT PATH (REQUIRED):
{runtime_module_name}

IMPORTANT:
- The target import path is a Python MODULE path.
- It is NOT a class name.
- It is NOT a function name.
- Do not shorten the path.
- Do not import from parent modules.
- Import symbols from this module.

SOURCE CODE (REFERENCE ONLY - DO NOT COPY):
{source_code}

TASK:
Generate pytest tests for the module at the target import path.

MANDATORY RULES:

* You MUST import code from:
  {runtime_module_name}

* The generated tests MUST use the imported code from:
  {runtime_module_name}

* Every test function MUST call at least one function or method from the imported module.

* The source code is provided only to understand behavior.
  DO NOT copy, recreate, redefine, mock, or reimplement any class,
  function, method, constant, or behavior from the source code.

* DO NOT create replacement implementations.

* If a class named "FooBar" exists in the source code,
  DO NOT define a new class named "FooBar" in the tests.

* If a function named "foo" exists in the source code,
  DO NOT define a new function named "foo" in the tests.

REQUIRED VALIDATION:

Before writing the final answer, verify:

1. The code imports {runtime_module_name}.
2. Every test uses the imported module.
3. No implementation from the source code is recreated.
4. No mock or fallback implementation exists.

{improvement_section}

OUTPUT FORMAT:

* Output ONLY valid Python code.
* No markdown fences.
* No explanations.
* No prose.
* No comments outside normal Python code.

"""

    def _clean_generated_code(self, code: str) -> str:
        code = code.strip()

        if code.startswith("```python"):
            code = code.removeprefix("```python")

        if code.startswith("```"):
            code = code.removeprefix("```")

        if code.endswith("```"):
            code = code.removesuffix("```")

        return code.strip()

    def _resolve_runtime_import_path(
            self,
            repo_path: Path,
            source_file: Path,
            execution_cwd: Path,
    ) -> str:
        candidates = []
        logger.debug(f"Execution_cwd: {execution_cwd}")
        try:
            relative = source_file.relative_to(execution_cwd).with_suffix("")

            for index in range(len(relative.parts)):
                candidates.append(".".join(relative.parts[index:]))

        except Exception:
            pass

        try:
            relative_repo = source_file.relative_to(repo_path).with_suffix("")

            for index in range(len(relative_repo.parts)):
                candidates.append(".".join(relative_repo.parts[index:]))

        except Exception:
            pass

        candidates = list(dict.fromkeys(candidates))

        logger.debug(f"[TestCreationAgent] Import candidates: {candidates}")

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{execution_cwd}:{existing}"

        venv_python = repo_path / "venv" / "bin" / "python3"
        dotenv_python = repo_path / ".venv" / "bin" / "python3"
        python_executable = (
            str(venv_python) if venv_python.exists()
            else str(dotenv_python) if dotenv_python.exists()
            else "python3"
        )
        logger.debug(f"venv python exists? {venv_python.exists()}")
        logger.debug(f"dotenv python exists? {dotenv_python.exists()}")

        for candidate in candidates:
            command = [
                python_executable,
                "-c",
                f"import importlib;importlib.import_module('{candidate}')",
            ]

            result = subprocess.run(
                command,
                cwd=execution_cwd,
                capture_output=True,
                text=True,
                env=env,
            )

            logger.debug(f"Result of import run for cadidate {candidate}: {result}")

            if result.returncode == 0:
                logger.debug(f"[TestCreationAgent] Resolved runtime import path: {candidate}")
                return candidate

        raise RuntimeError(f"Failed to resolve runtime import path for {source_file}")

    def _generate_test_code(
            self,
            source_code: str,
            runtime_module_name: str,
            previous_suite: str | None = None,
            previous_coverage: float | None = None,
    ) -> str:
        prompt = self._build_prompt(
            source_code=source_code,
            runtime_module_name=runtime_module_name,
            previous_suite=previous_suite,
            previous_coverage=previous_coverage,
        )

        # logger.debug(f"[TestCreationAgent] Prompt for test generation:\n{prompt}")

        response = self.model.invoke([
            SystemMessage(content=LLM_TEST_GENERATION_PROMPT),
            HumanMessage(content=prompt),
        ])

        return response.content

    def _write_test_file(
            self,
            output_root: Path,
            module_name: str,
            content: str,
            attempt: int,
    ) -> Path:
        module_dir = output_root / module_name.replace(".", "_")
        module_dir.mkdir(parents=True, exist_ok=True)

        test_path = module_dir / f"test_generated_{attempt}_{uuid.uuid4().hex[:8]}.py"

        test_path.write_text(content, encoding="utf-8")

        return test_path

    def _validate_test_suite(
            self,
            repo_path: Path,
            test_file: Path,
            runtime_module_name: str,
    ) -> tuple[bool, float, str]:

        runner = TestRunner(repo_path)

        result = runner.run_pytest(
            [str(test_file.resolve())],
            repo_path=repo_path,
            coverage=True,
            coverage_target=runtime_module_name,
        )

        logger.debug(f"Generated test validation:\n{result['stdout']}\n{result['stderr']}")

        coverage = result.get("coverage", 0.0)
        combined_output = result["stdout"] + "\n" + result["stderr"]

        return result["success"], coverage, combined_output

    def _extract_failing_tests(self, output: str) -> list[str]:
        failing = []

        for line in output.splitlines():
            stripped = line.strip()

            if not (stripped.startswith("FAILED") or stripped.startswith("ERROR")):
                continue

            parts = stripped.split("::")

            if len(parts) < 2:
                continue

            test_name = parts[-1].split()[0].strip()
            failing.append(test_name)

        return list(set(failing))

    def _prune_failing_tests_ast(self, test_file: Path, failing_tests: list[str]):
        source = test_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        new_body = []

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in failing_tests:
                logger.info(f"[TestCreationAgent] Pruned failing test: {node.name}")
                continue

            if isinstance(node, ast.ClassDef):
                filtered_class_body = []

                for child in node.body:
                    should_remove = isinstance(child, ast.FunctionDef) and child.name in failing_tests

                    if should_remove:
                        logger.info(f"[TestCreationAgent] Pruned failing test: {child.name}")
                        continue

                    filtered_class_body.append(child)

                node.body = filtered_class_body

            new_body.append(node)

        tree.body = new_body
        rewritten = ast.unparse(tree)
        test_file.write_text(rewritten, encoding="utf-8")

    def _count_test_functions(self, source: str) -> int:
        tree = ast.parse(source)
        count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                count += 1

        return count

    def _generate_hidden_tests(
            self,
            repo_path: Path,
            output_root: Path,
            module_name: str,
    ) -> Path:

        logger.info(f"[TestCreationAgent] Generating tests for module: {module_name}")

        source_file = repo_path / Path(module_name.replace(".", "/")).with_suffix(".py")
        execution_cwd = repo_path

        runtime_module_name = self._resolve_runtime_import_path(
            repo_path=repo_path,
            source_file=source_file,
            execution_cwd=execution_cwd,
        )

        source_code = source_file.read_text(encoding="utf-8")

        output_dir = output_root / module_name.replace(".", "_")
        output_dir.mkdir(parents=True, exist_ok=True)

        best_test_file = None
        best_coverage = 0.0
        best_suite_source = None

        for attempt in range(self.MAX_GENERATION_ATTEMPTS):
            logger.info(
                f"[TestCreationAgent] Generation attempt {attempt + 1}/{self.MAX_GENERATION_ATTEMPTS}"
            )

            generated_code = self._generate_test_code(
                source_code=source_code,
                runtime_module_name=runtime_module_name,
                previous_suite=best_suite_source,
                previous_coverage=best_coverage if best_suite_source else None,
            )

            logger.debug(f"Generated test code:\n{generated_code}")

            generated_code = self._clean_generated_code(generated_code)

            if (
                    f"import {runtime_module_name}" not in generated_code
                    and f"from {runtime_module_name}" not in generated_code
            ):
                logger.info("[TestCreationAgent] Discarded suite without target module import.")
                continue

            test_file = self._write_test_file(
                output_root=output_root,
                module_name=module_name,
                content=generated_code,
                attempt=attempt,
            )

            logger.info(f"[TestCreationAgent] Validating generated suite: {test_file.name}")

            success, coverage, output = self._validate_test_suite(
                repo_path=repo_path,
                test_file=test_file,
                runtime_module_name=runtime_module_name,
            )

            if success:
                logger.info(f"[TestCreationAgent] Accepted suite: {test_file.name}")
                logger.info(f"[TestCreationAgent] Coverage: {coverage:.2f}")

                if coverage > best_coverage:
                    if best_test_file and best_test_file.exists():
                        best_test_file.unlink()

                    best_test_file = test_file
                    best_suite_source = test_file.read_text(encoding="utf-8")
                    best_coverage = coverage

                else:
                    test_file.unlink()

            else:
                failing_tests = self._extract_failing_tests(output)

                logger.info(f"[TestCreationAgent] Failing tests detected: {failing_tests}")

                if failing_tests:
                    self._prune_failing_tests_ast(
                        test_file=test_file,
                        failing_tests=failing_tests,
                    )

                    logger.info(
                        f"[TestCreationAgent] Revalidating pruned suite: {test_file.name}"
                    )

                    pruned_source = test_file.read_text(encoding="utf-8")

                    if self._count_test_functions(pruned_source) == 0:
                        logger.info(
                            "[TestCreationAgent] All tests pruned. Discarding suite."
                        )

                        try:
                            test_file.unlink()
                        except Exception:
                            pass

                        continue

                    retry_success, retry_coverage, retry_output = (
                        self._validate_test_suite(
                            repo_path=repo_path,
                            test_file=test_file,
                            runtime_module_name=runtime_module_name,
                        )
                    )

                    if retry_success:
                        logger.info(
                            f"[TestCreationAgent] Accepted pruned suite: {test_file.name}"
                        )
                        logger.info(
                            f"[TestCreationAgent] Coverage after pruning: {retry_coverage:.2f}"
                        )

                        if retry_coverage > best_coverage:
                            if best_test_file and best_test_file.exists():
                                best_test_file.unlink()

                            best_test_file = test_file
                            best_suite_source = test_file.read_text(encoding="utf-8")
                            best_coverage = retry_coverage

                        else:
                            test_file.unlink()

                        if best_coverage >= self.TARGET_COVERAGE:
                            logger.info(
                                "[TestCreationAgent] Coverage target reached."
                            )
                            break

                    else:
                        logger.info(
                            f"[TestCreationAgent] Discarded invalid suite: {test_file.name}"
                        )

                        try:
                            test_file.unlink()
                        except Exception:
                            pass

                else:
                    logger.info(
                        f"[TestCreationAgent] Discarded invalid suite: {test_file.name}"
                    )

                    try:
                        test_file.unlink()
                    except Exception:
                        pass

            if best_coverage >= self.TARGET_COVERAGE:
                logger.info("[TestCreationAgent] Coverage target reached.")
                break

        logger.info(
            f"[TestCreationAgent] Kept {'1' if best_test_file else '0'} valid hidden test suites."
        )
        logger.info(f"[TestCreationAgent] Best coverage: {best_coverage:.2f}")

        test_coverage = best_coverage if best_test_file else 0.0

        return output_dir, test_coverage

    def prepare_test_context(
            self,
            repo_path: Path,
            generated_tests_root: Path,
            changed_files: list[Path],
    ) -> dict:

        repo_path = Path(repo_path)
        generated_tests_root = Path(generated_tests_root)

        indexer = TestIndexer(repo_path)
        test_index = indexer.build_index()

        discovery = TestDiscovery(
            repo_path=repo_path,
            test_index=test_index,
        )

        visible_tests = discovery.find_related_tests(changed_files)

        hidden_generated_tests = []
        test_coverage = 0.0
        coverages = []

        logger.info(f"Found {len(visible_tests)} related tests.")

        if not visible_tests:
            for file_path in changed_files:
                module_name = (
                    file_path
                    .relative_to(repo_path)
                    .with_suffix("")
                    .as_posix()
                    .replace("/", ".")
                )
                separate_modules = module_name.split(".")
                if "test" not in separate_modules and "tests" not in separate_modules and "maint" not in separate_modules and "t" not in separate_modules and "js_tests" not in separate_modules:
                    logger.info(f"[TestCreationAgent] No existing tests found, generating for: {file_path.name}")

                    output_dir, coverage = self._generate_hidden_tests(
                        repo_path=repo_path,
                        output_root=generated_tests_root,
                        module_name=module_name,
                    )

                    coverages.append(coverage)
                    hidden_generated_tests.extend(output_dir.rglob("*test*.py"))
                else:
                    logger.debug(f"[TestCreationAgent] Excluding test generation for: {module_name}")

            test_coverage = sum(coverages) / len(coverages) if coverages else 0.0

        return {
            "visible_tests": visible_tests,
            "hidden_generated_tests": hidden_generated_tests,
            "test_coverage": test_coverage,
        }

    def run(self, *args, **kwargs):
        return self.prepare_test_context(*args, **kwargs)
