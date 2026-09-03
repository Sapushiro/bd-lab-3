import json
import sys
from pathlib import Path
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_PATH = PROJECT_ROOT / "scenario.json"

def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def run_scenario(client: httpx.Client, scenario: dict) -> str | None:
    name = scenario["name"]
    method = scenario["method"]
    endpoint = scenario["endpoint"]

    requests_arguments = {}
    if "request" in scenario:
        requests_arguments["json"] = scenario["request"]

    if "raw_body" in scenario:
        requests_arguments["content"] = scenario["raw_body"]
        requests_arguments["headers"] = {
            "Content-Type": "application/json"
        }

    response = client.request(method, endpoint, **requests_arguments)
    expected_status = scenario["expected_status"]

    if response.status_code != expected_status:
        return (
            f"{name}: expected status {expected_status}, "
            f"received {response.status_code}."
            f"Response: {response.text}"
        )

    if "expected_response" in scenario:
        actual_response = response.json()
        expected_response = scenario["expected_response"]

        if actual_response != expected_response:
            return (
                f"{name}: expected_response {expected_response},"
                f"received {actual_response}"
            )

    return None


def main() -> None:
    configuration = load_json(SCENARIO_PATH)
    base_url = configuration["base_url"]
    scenario_files = configuration["scenario_files"]

    failures = []
    total = 0

    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        for relative_path in scenario_files:
            scenarios = load_json(PROJECT_ROOT / relative_path)

            for scenario in scenarios:
                total += 1

                try:
                    error = run_scenario(client, scenario)
                except Exception as exception:
                    error = f"{scenario['name']: {exception}}"

                if error is None:
                    print(f"PASS: {scenario['name']}")
                else:
                    print(f"FAIL: {error}")
                    failures.append(error)

    print()
    print(f"Executed: {total}")
    print(f"Passed: {total - len(failures)}")
    print(f"Failed: {len(failures)}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
