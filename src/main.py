import ast
import os
import re
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from src.utils.logger import logger
from src.agents.vanilla_agent import VanillaAgent

load_dotenv()

# Read config from environment variables with defaults
MODEL = os.getenv("MODEL", "gemma-4-26b-a4b-it")
USE_DUMMY = os.getenv("USE_DUMMY", "false").lower() == "true"
USE_SPECIALISED = os.getenv("USE_SPECIALISED", "false").lower() == "true"
USE_VANILLA = os.getenv("USE_VANILLA", "false").lower() == "true"

if USE_DUMMY:
    from DummyBench.scripts.base_mapping import file_mapping
    BENCH_NAME = "DummyBench"
else:
    from RefactorBench.scripts.base_mapping import file_mapping
    BENCH_NAME = "RefactorBench"

from src.agents import GeneralRefactoringAgent
from src.graph import build_graph

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILTERED_PROBLEMS_FILE = (
        PROJECT_ROOT /
        "RefactorBenchFilteredProblems" /
        "mapping.txt"
)


def _cleanup_repo_test_folders(generated: bool = False, repo_tests: bool = False) -> None:
    """Delete generated_tests and/or tests folders inside all benchmark repositories."""
    repos_path = PROJECT_ROOT / BENCH_NAME / "repositories"
    for repo_dir in repos_path.iterdir():
        if not repo_dir.is_dir():
            continue
        if generated:
            folder = repo_dir / "generated_tests"
            if folder.exists():
                shutil.rmtree(folder)
                logger.info(f"[Cleanup] Deleted generated_tests in {repo_dir.name}")
        if repo_tests:
            folder = repo_dir / "tests"
            if folder.exists():
                shutil.rmtree(folder)
                logger.info(f"[Cleanup] Deleted tests in {repo_dir.name}")

def load_allowed_tasks(
        mapping_file: Path,
) -> set[str]:

    allowed = set()

    for line in mapping_file.read_text().splitlines():

        line = line.strip()

        if (
                not line
                or line.startswith("---")
        ):
            continue

        task_name = (
            line.split(",")[0]
            .strip()
        )

        allowed.add(task_name)

    return allowed

ALLOWED_TASKS = (
    load_allowed_tasks(
        FILTERED_PROBLEMS_FILE
    )
)


def count_subtests(test_path: Path) -> int:
    tree = ast.parse(test_path.read_text())

    total = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                        isinstance(target, ast.Name)
                        and target.id == "files_to_check"
                        and isinstance(node.value, ast.List)
                ):
                    total += len(node.value.elts)

    return total


def count_failed_subtests(output: str) -> int:
    return len(
        re.findall(
            r"^(?:FAIL|ERROR): .*?\(file='[^']+'\)",
            output,
            re.MULTILINE,
        )
    )


def _parse_test_output(output: str, test_path: Path) -> tuple[int, int]:
    total_subtests = count_subtests(test_path)

    if total_subtests > 0:
        failed = count_failed_subtests(output)
        return total_subtests - failed, total_subtests

    # fallback for normal unittest tests
    ran_match = re.search(r"Ran (\d+) tests?", output)
    if not ran_match:
        return 0, 0

    total = int(ran_match.group(1))

    failures = int(m.group(1)) if (m := re.search(r"failures=(\d+)", output)) else 0
    errors = int(m.group(1)) if (m := re.search(r"errors=(\d+)", output)) else 0

    return max(0, total - failures - errors), total

def _apply_refactor(agent, test_rel_path, task_rel_path) -> tuple[Path, Path, Path, str] | None:
    """
    Runs the agent on the relevant file and writes the refactored result.
    Returns (test_path, package_root, original_content_path) or None if setup fails.
    """
    repository_name = test_rel_path.split('/')[2]

    full_task_path = PROJECT_ROOT / "RefactorBench" / task_rel_path[3:]
    full_test_path = PROJECT_ROOT / "RefactorBench" / test_rel_path[3:]
    full_repository_path = PROJECT_ROOT / "RefactorBench" / "repositories" /  repository_name

    instruction = full_task_path.read_text().strip()
    test_code = full_test_path.read_text()

    match = re.search(r"file_path = '(\.\./[^']+)'", test_code)
    if not match:
        logger.info(f"Could not find file_path in test: {full_test_path}")
        return None

    relevant_file = full_repository_path / match.group(1)[3:]
    execution_path = full_repository_path / match.group(1)[3:].split('/')[0]

    if not relevant_file.exists():
        logger.info(f"Relevant file not found: {relevant_file}")
        return None

    original_content = relevant_file.read_text()
    refactored = agent.run(instruction=instruction, context=original_content)
    relevant_file.write_text(refactored)

    logger.info(f"Problem:        {repository_name}")
    logger.info(f"Instruction:    {instruction}")
    logger.info(f"Relevant file:  {relevant_file}")
    logger.info(f"Test:           {full_test_path}")
    logger.info(f"Refactored code applied")
    logger.info("-" * 50)

    return full_test_path, execution_path, relevant_file, original_content


def _run_test(test_path: Path, execution_path: Path, repository_name: str, task_name: str) -> tuple[int, int]:
    """Runs a single test file and returns (passes, total)."""
    cmd = ["python", f"../../../tests/{test_path.parent.name}/{test_path.name}"]
    result = subprocess.run(cmd, cwd=execution_path, capture_output=True, text=True)
    output = result.stdout + result.stderr
    logger.debug(output)

    passes, total = _parse_test_output(output, test_path)
    if total == 0:
        logger.info("Could not parse test results")
    else:
        logger.info(f"[{repository_name}] {task_name}: {passes}/{total}")

    return passes, total


def evaluate(agent):
    originals = {}  # relevant_file -> original_content
    jobs = []   # (test_path, package_root)

    # Apply refactorings on tasks and queue tests
    for test_rel_path, task_rel_path in file_mapping.items():
        task_name = Path(task_rel_path).name

        if not USE_DUMMY and task_name not in ALLOWED_TASKS:
            continue

        outcome = _apply_refactor(agent, test_rel_path, task_rel_path)
        if outcome:
            test_path, package_root, relevant_file, original_content = outcome
            originals[relevant_file] = original_content
            jobs.append((test_path, package_root))

    # Run tests
    total_passes = total_tests = 0
    for test_path, package_root in jobs:
        passes, total = _run_test(test_path, package_root)
        total_passes += passes
        total_tests += total
    logger.info(f"Overall: {total_passes}/{total_tests} tests passed")

    # Revert changes made by refactoring agent in benchmark repository
    for file_path, original_content in originals.items():
        file_path.write_text(original_content)
    logger.info("All changes reverted")

def evaluate_graph():
    """Run the full multi-agent graph on benchmark."""
    #If we set the flag to true, we use the dual specialised refactoring agent setup rather than the single general one
    graph = build_graph(MODEL, use_specialised_refactoring_agents=USE_SPECIALISED)
    # graph = build_graph("gemma3:1b", use_specialised_refactoring_agents=False)
    originals = {}
    test_originals = {}
    jobs = []
    results = []

    for test_rel_path, task_rel_path in file_mapping.items():

        task_name = Path(task_rel_path).name

        if not USE_DUMMY and task_name not in ALLOWED_TASKS:
            continue

        repository_name = test_rel_path.split('/')[2]
        full_task_path = PROJECT_ROOT / BENCH_NAME / task_rel_path[3:]
        full_test_path = PROJECT_ROOT / BENCH_NAME / test_rel_path[3:]
        full_repository_path = PROJECT_ROOT / BENCH_NAME / "repositories" / repository_name

        instruction = full_task_path.read_text().strip()
        test_code = full_test_path.read_text()

        match = re.search(r"file_path = '(\.\./[^']+)'", test_code)
        if not match:
            logger.info(f"Could not find file_path in test: {full_test_path}")
            continue

        relevant_file = full_repository_path / match.group(1)[3:]
        execution_path = full_repository_path / match.group(1)[3:].split('/')[0]

        if not relevant_file.exists():
            logger.info(f"Relevant file not found: {relevant_file}")
            continue

        original_content = relevant_file.read_text()

        # Store original test code for all tests in the repository, to ensure we can revert any changes made
        tests_dir = full_repository_path / "tests"

        if tests_dir.exists():

            for test_file in tests_dir.rglob("*.py"):

                if test_file not in test_originals:

                    test_originals[test_file] = (
                        test_file.read_text()
                    )

        # Clean up generated_tests and repo tests from all repos before starting
        _cleanup_repo_test_folders(generated=True)

        try:
            result = graph.invoke({
                "instruction": instruction,
                "repository_path": str(full_repository_path),
                "file_plans": [],
                "visible_tests": [],
                "generated_tests": [],
                "refactored_code": {},
                "compile_error": None,
                "compile_attempts": 0,
                "test_error": None,
                "test_attempts": 0,
                "tests_generated": 0,
                "test_coverage": 0.0,
                "success": False,
            })

            refactored = result["refactored_code"]

            if result["success"] and isinstance(refactored, dict):
                for file_path, code in refactored.items():
                    Path(file_path).write_text(code)

        except Exception as error:
            logger.error(f"[Graph Error] {error}")

            relevant_file.write_text(original_content)

            for file_path, content in test_originals.items():
                file_path.write_text(content)

            result = {
                "success": False,
                "compile_attempts": 0,
                "test_attempts": 0,
                "tests_generated": 0,
                "test_coverage": 0.0,
                "file_plans": [],
                "refactored_code": {},
            }

        logger.info(f"Repository:  {repository_name}")
        logger.info(f"Instruction: {instruction}")
        logger.info(f"Success:     {result['success']}")
        logger.info(f"Attempts:    {result['compile_attempts']}")
        logger.info("-" * 50)

        if relevant_file not in originals:
            originals[relevant_file] = original_content
        jobs.append((full_test_path, execution_path))
        results.append(result)

    _cleanup_repo_test_folders(generated=True)

    logger.info(f"[REFACTORING COMPLETE] Running tests to check refactoring success...")

    with open(f"results/{BENCH_NAME}_results.csv", "w") as f:
        f.write(
            "repository,task,model,refactoring_entity,files_refactored,tests_generated,test_coverage,test_attempts,success,tests_passed,total_tests,passing_rate\n")
        total_passes = total_tests = 0
        for (test_path, execution_path), result in zip(jobs, results):
            repository_name = execution_path.parent.name
            task_name = test_path.stem

            file_plans = result.get("file_plans", [])
            files_refactored = len(set(fp["file"] for fp in file_plans))
            refactoring_entities = set()
            for fp in file_plans:
                for r in fp.get("refactorings", []):
                    entity = r.get("refactoring_entity")
                    if entity:
                        refactoring_entities.add(entity)
            refactoring_entities = ";".join(refactoring_entities) if refactoring_entities else "unknown"

            passes, total = _run_test(test_path, execution_path, repository_name, task_name)
            total_passes += passes
            total_tests += total
            rate = passes / total if total > 0 else 0.0
            f.write(
                f"{repository_name},"
                f"{task_name},"
                f"{MODEL},"
                f"{refactoring_entities},"
                f"{files_refactored},"
                f"{result.get('tests_generated', 0)},"
                f"{result.get('test_coverage', 0.0):.2f},"
                f"{result.get('test_attempts', 0)},"
                f"{result.get('success', False)},"
                f"{passes},"
                f"{total},"
                f"{rate:.2f}\n"
            )

    logger.info(f"\nOverall: {total_passes}/{total_tests} tests passed")

    for file_path, original_content in originals.items():
        file_path.write_text(original_content)
    for file_path, original_content in test_originals.items():
        file_path.write_text(original_content)
    logger.info("All changes reverted")


def evaluate_vanilla_agent():

    agent = VanillaAgent(MODEL)

    originals = {}
    jobs = []

    for test_rel_path, task_rel_path in file_mapping.items():

        task_name = Path(task_rel_path).name

        if not USE_DUMMY and task_name not in ALLOWED_TASKS:
            continue

        repository_name = test_rel_path.split('/')[2]

        full_task_path = PROJECT_ROOT / BENCH_NAME / task_rel_path[3:]
        full_test_path = PROJECT_ROOT / BENCH_NAME / test_rel_path[3:]
        full_repository_path = (
                PROJECT_ROOT /
                BENCH_NAME /
                "repositories" /
                repository_name
        )

        instruction = full_task_path.read_text().strip()

        test_code = full_test_path.read_text()

        match = re.search(
            r"file_path = '(\.\./[^']+)'",
            test_code
        )

        if not match:
            continue

        relevant_file = (
                full_repository_path /
                match.group(1)[3:]
        )

        execution_path = (
                full_repository_path /
                match.group(1)[3:].split('/')[0]
        )

        if not relevant_file.exists():
            continue

        if relevant_file not in originals:
            originals[relevant_file] = (
                relevant_file.read_text()
            )

        try:

            refactored_files = agent.run(
                instruction=instruction,
                repository_path=full_repository_path,
            )

            for file_path, code in refactored_files.items():

                path = Path(file_path)

                if path not in originals:
                    originals[path] = path.read_text()

                path.write_text(code)

        except Exception as error:

            logger.error(
                f"[Vanilla Error] {error}"
            )

        jobs.append(
            (
                full_test_path,
                execution_path,
            )
        )

    logger.info(
        "[REFACTORING COMPLETE] Running tests..."
    )

    total_passes = total_tests = 0

    for test_path, execution_path in jobs:

        repository_name = execution_path.parent.name
        task_name = test_path.stem

        passes, total = _run_test(
            test_path,
            execution_path,
            repository_name,
            task_name,
        )

        total_passes += passes
        total_tests += total

    logger.info(
        f"Overall: {total_passes}/{total_tests}"
    )

    for file_path, content in originals.items():
        file_path.write_text(content)

    logger.info("All changes reverted")

def run_tests_without_refactoring():
    Path("results").mkdir(exist_ok=True)  # ← add this
    test_originals = {}
    jobs = []

    logger.info(f"Running tests without refactoring...")

    for test_rel_path, task_rel_path in file_mapping.items():

        task_name = Path(task_rel_path).name

        if not USE_DUMMY and task_name not in ALLOWED_TASKS:
            continue

        repository_name = test_rel_path.split('/')[2]
        full_test_path = PROJECT_ROOT / BENCH_NAME / test_rel_path[3:]
        full_repository_path = PROJECT_ROOT / BENCH_NAME / "repositories" / repository_name

        test_code = full_test_path.read_text()

        match = re.search(r"file_path = '(\.\./[^']+)'", test_code)
        if not match:
            logger.info(f"Could not find file_path in test: {full_test_path}")
            continue

        relevant_file = full_repository_path / match.group(1)[3:]
        execution_path = full_repository_path / match.group(1)[3:].split('/')[0]

        if not relevant_file.exists():
            logger.info(f"Relevant file not found: {relevant_file}")
            continue

        jobs.append((full_test_path, execution_path))

        # Store original test code for all tests in the repository, to ensure we can revert any changes made
        tests_dir = full_repository_path / "tests"

        if tests_dir.exists():

            for test_file in tests_dir.rglob("*.py"):

                if test_file not in test_originals:
                    test_originals[test_file] = (
                        test_file.read_text()
                    )

    with open(f"results/{BENCH_NAME}_results.csv", "w") as f:
        f.write("repository,task,tests_passed,total_tests,passing_rate\n")
        total_passes = total_tests = 0
        for test_path, package_root in jobs:
            repository_name = package_root.parent.name
            task_name = test_path.stem
            passes, total = _run_test(test_path, package_root, repository_name, task_name)
            total_passes += passes
            total_tests += total
            f.write(f"{repository_name},{task_name},{passes},{total},{passes / total if total > 0 else 0:.2f}\n")

    # assert total_tests == 149, f"Expected 149 total tests, got {total_tests}"
    # assert total_passes == 13, f"Expected 13 passes without refactoring, got {total_passes}"
    # logger.info("✓ Baseline assertion passed: 13/149 tests pass without refactoring")

def evaluate_base_agent():
    evaluate(GeneralRefactoringAgent(MODEL))
    # evaluate(VanillaAgent("gemma3:1b"))


if __name__ == "__main__":
    # evaluate_base_agent()

    _cleanup_repo_test_folders(repo_tests=False)

    if not USE_DUMMY:
        run_tests_without_refactoring()

    logger.info("=" * 50)
    logger.info("[Config] Starting refactoring evaluation with:")
    logger.info(f"  MODEL:          {MODEL}")
    logger.info(f"  BENCH:          {BENCH_NAME}")
    logger.info(f"  USE_SPECIALISED:{USE_SPECIALISED}")
    logger.info(f"  USE_VANILLA:    {USE_VANILLA}")
    logger.info("=" * 50)

    if USE_VANILLA:
        evaluate_vanilla_agent()
    else:
        evaluate_graph()
