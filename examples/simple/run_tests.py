import os
import subprocess
import sys

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_LIST_PATH = os.path.join(SCRIPT_DIR, "tests.list")


def load_test_names(list_path):
    with open(list_path) as fh:
        return [line.strip() for line in fh if line.strip() and not line.strip().startswith("#")]


def run_test(test_name):
    """Run `python3 <test_name>` and check its exit code and stdout. Returns True on success."""
    result = subprocess.run(
        ["python3", test_name],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        print(f"{RED}FAIL: {test_name} exited with code {result.returncode} (expected 0){RESET}")
        return False

    if "PASS" not in result.stdout:
        print(f"{RED}FAIL: {test_name} did not print 'PASS' in its stdout{RESET}")
        return False

    print(f"{GREEN}OK: {test_name}{RESET}")
    return True


def main():
    if not os.path.exists(TESTS_LIST_PATH):
        print(f"{RED}FAIL: test list not found: {TESTS_LIST_PATH}{RESET}")
        sys.exit(1)

    test_names = load_test_names(TESTS_LIST_PATH)
    if not test_names:
        print(f"{RED}FAIL: no tests listed in {TESTS_LIST_PATH}{RESET}")
        sys.exit(1)

    failures = []
    for test_name in test_names:
        print(f"==== Running {test_name} ====")
        if not run_test(test_name):
            failures.append(test_name)
        print()

    total = len(test_names)
    passed = total - len(failures)
    if failures:
        print(f"{RED}{len(failures)}/{total} test(s) FAILED: {', '.join(failures)}{RESET}")
        sys.exit(1)

    print(f"{GREEN}All {passed}/{total} test(s) passed{RESET}")


if __name__ == "__main__":
    main()
