# ============================================================
# EXPERIMENT 7
# Automated Testing and Report Generation
# Customer Support RAG Application using FastAPI and Ollama
# ============================================================

import time
import logging
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

# FastAPI server address
BASE_URL = "http://127.0.0.1:8000"

# API endpoints
HEALTH_URL = f"{BASE_URL}/health"
ASK_URL = f"{BASE_URL}/ask"

# Create a folder to store reports and graphs
REPORT_DIR = Path("experiment7_reports")
REPORT_DIR.mkdir(exist_ok=True)

# Store test results here
results = []

# Store response times for graph generation
response_times = []


# ============================================================
# 2. LOGGING CONFIGURATION
# ============================================================

# Save operational information to a log file.
# Do not log API keys or private customer information.
logging.basicConfig(
    filename=REPORT_DIR / "test_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("experiment7_tests")


# ============================================================
# 3. HELPER FUNCTION TO RECORD TEST RESULTS
# ============================================================

def record_result(
    test_case,
    expected_result,
    actual_result,
    passed,
    response_time_ms=None
):
    """Save the outcome of an individual test."""

    results.append({
        "Test case": test_case,
        "Expected result": expected_result,
        "Actual result": actual_result,
        "Response time (ms)": (
            round(response_time_ms, 2)
            if response_time_ms is not None
            else "N/A"
        ),
        "Pass/Fail": "PASS" if passed else "FAIL"
    })

    # Log the outcome without recording customer questions.
    logger.info(
        "Test=%s | Result=%s | Latency_ms=%s",
        test_case,
        "PASS" if passed else "FAIL",
        response_time_ms
    )

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{test_case}: {actual_result}"
    )


# ============================================================
# 4. TEST HEALTH ENDPOINT
# ============================================================

def test_health():
    """Check whether the FastAPI application is running."""

    try:
        response = requests.get(
            HEALTH_URL,
            timeout=5
        )

        elapsed = response.elapsed.total_seconds() * 1000

        passed = (
            response.status_code == 200
            and response.json().get("status") == "healthy"
        )

        record_result(
            test_case="Health endpoint",
            expected_result="API reports running",
            actual_result=f"HTTP {response.status_code}",
            passed=passed,
            response_time_ms=elapsed
        )

    except requests.RequestException:
        record_result(
            test_case="Health endpoint",
            expected_result="API reports running",
            actual_result="API unavailable",
            passed=False
        )


# ============================================================
# 5. HELPER FUNCTION TO TEST QUESTIONS
# ============================================================

def test_question(
    test_case,
    question,
    expected_result,
    validation=None
):
    """
    Send a question to the /ask endpoint and check its response.

    validation is an optional function that checks the response body.
    """

    start_time = time.perf_counter()

    try:
        response = requests.post(
            ASK_URL,
            json={"question": question},
            timeout=180
        )

        elapsed = (
            time.perf_counter() - start_time
        ) * 1000

        # Save the observed response time.
        response_times.append({
            "Test case": test_case,
            "Response time (ms)": round(elapsed, 2)
        })

        # Check the result.
        if validation is None:
            passed = response.status_code == 200
        else:
            passed = validation(response)

        # Keep the report readable and avoid storing full responses.
        if response.status_code == 200:
            body = response.json()

            actual = (
                f"HTTP 200; status={body.get('status', 'unknown')}"
            )

            # Prefer the latency measured by the API itself.
            api_latency = body.get("response_time_ms")

            if isinstance(api_latency, (int, float)):
                response_time_ms = api_latency
            else:
                response_time_ms = elapsed

        else:
            actual = f"HTTP {response.status_code}"
            response_time_ms = elapsed

        record_result(
            test_case=test_case,
            expected_result=expected_result,
            actual_result=actual,
            passed=passed,
            response_time_ms=response_time_ms
        )

    except requests.RequestException as error:
        # Avoid printing request bodies or private configuration.
        record_result(
            test_case=test_case,
            expected_result=expected_result,
            actual_result=f"Request failed: {type(error).__name__}",
            passed=False
        )


# ============================================================
# 6. TEST SUPPORTED QUESTION
# ============================================================

def test_supported_question():
    """Verify that the knowledge base can answer a known question."""

    test_question(
        test_case="Supported question",
        question="How long does shipping take?",
        expected_result="Grounded shipping answer is returned",
        validation=lambda r: (
            r.status_code == 200
            and r.json().get("status") == "success"
            and "3 to 5 business days"
            in r.json().get("answer", "").lower()
        )
    )


# ============================================================
# 7. TEST UNSUPPORTED QUESTION
# ============================================================

def test_unsupported_question():
    """Verify that unsupported questions receive a fallback response."""

    test_question(
        test_case="Unsupported question",
        question="What is the warranty period for laptops?",
        expected_result="Insufficient-information fallback is returned",
        validation=lambda r: (
            r.status_code == 200
            and r.json().get("status") == "fallback"
            and "insufficient information"
            in r.json().get("answer", "").lower()
        )
    )


# ============================================================
# 8. TEST EMPTY QUESTION
# ============================================================

def test_empty_question():
    """An empty question should be rejected by FastAPI validation."""

    test_question(
        test_case="Empty question",
        question="",
        expected_result="HTTP 422 validation error",
        validation=lambda r: r.status_code == 422
    )


# ============================================================
# 9. TEST VERY LONG QUESTION
# ============================================================

def test_long_question():
    """Questions longer than 1,000 characters should be rejected."""

    long_question = "shipping " * 150

    test_question(
        test_case="Long question",
        question=long_question,
        expected_result="HTTP 422 validation error",
        validation=lambda r: r.status_code == 422
    )


# ============================================================
# 10. TEST BACKEND FAILURE
# ============================================================

def test_backend_failure():
    """
    Test the application's controlled error handling.

    IMPORTANT:
    This test temporarily stops Ollama on Windows only.
    For other operating systems, use the manual procedure below.
    """

    import platform
    import subprocess

    if platform.system() != "Windows":
        record_result(
            test_case="Backend unavailable",
            expected_result="Controlled HTTP 500 error",
            actual_result=(
                "Automatic shutdown is Windows-only; "
                "perform the manual backend test"
            ),
            passed=False
        )
        return

    # Ask before stopping the user's local Ollama process.
    print("\nBackend failure test:")
    print("This test temporarily stops Ollama.")
    print("Save any other work that depends on Ollama.")

    choice = input("Continue with the failure test? (y/n): ")

    if choice.strip().lower() != "y":
        record_result(
            test_case="Backend unavailable",
            expected_result="Controlled HTTP 500 error",
            actual_result="Test skipped by user",
            passed=False
        )
        return

    # Confirm that Ollama is reachable before stopping it.
    try:
        requests.get(
            "http://127.0.0.1:11434/api/tags",
            timeout=3
        )
    except requests.RequestException:
        record_result(
            test_case="Backend unavailable",
            expected_result="Controlled HTTP 500 error",
            actual_result="Ollama was already unavailable",
            passed=False
        )
        return

    try:
        # Stop Ollama for the duration of this test.
        subprocess.run(
            ["taskkill", "/F", "/IM", "ollama.exe"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Give the service a moment to stop.
        time.sleep(2)

        # Submit a valid question.
        start_time = time.perf_counter()

        response = requests.post(
            ASK_URL,
            json={"question": "How long does shipping take?"},
            timeout=10
        )

        elapsed = (
            time.perf_counter() - start_time
        ) * 1000

        # A controlled 500 response means the test passed.
        passed = (
            response.status_code == 500
            and "could not be processed"
            in response.text.lower()
        )

        record_result(
            test_case="Backend unavailable",
            expected_result=(
                "Controlled HTTP 500 error without "
                "exposing internal details"
            ),
            actual_result=f"HTTP {response.status_code}",
            passed=passed,
            response_time_ms=elapsed
        )

    except requests.RequestException as error:
        record_result(
            test_case="Backend unavailable",
            expected_result="Controlled HTTP 500 error",
            actual_result=f"Request failed: {type(error).__name__}",
            passed=False
        )

    finally:
        print(
            "\nIf Ollama was stopped, restart it using "
            "your normal Ollama startup method."
        )


# ============================================================
# 11. CREATE OBSERVATION TABLE
# ============================================================

def generate_observation_table():
    """Create and save the observation table."""

    df = pd.DataFrame(results)

    # Save the table as CSV.
    df.to_csv(
        REPORT_DIR / "observation_table.csv",
        index=False
    )

    # Save a formatted HTML version.
    html = df.to_html(
        index=False,
        border=1
    )

    with open(
        REPORT_DIR / "observation_table.html",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "<h1>Experiment 7 Observation Table</h1>\n"
            + html
        )

    print("\n" + "=" * 70)
    print("EXPERIMENT 7 OBSERVATION TABLE")
    print("=" * 70)

    print(df.to_string(index=False))

    return df


# ============================================================
# 12. GENERATE TEST RESULTS GRAPH
# ============================================================

def generate_test_graph(df):
    """Generate a graph showing passed and failed tests."""

    counts = (
        df["Pass/Fail"]
        .value_counts()
        .reindex(["PASS", "FAIL"], fill_value=0)
    )

    plt.figure(figsize=(7, 5))

    counts.plot(
        kind="bar",
        color=["seagreen", "tomato"]
    )

    plt.title("Experiment 7: Test Results")
    plt.xlabel("Test outcome")
    plt.ylabel("Number of tests")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "test_results_graph.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 13. GENERATE RESPONSE TIME GRAPH
# ============================================================

def generate_response_graph():
    """Plot response times for requests that were measured."""

    if not response_times:
        print("No response-time data available for the graph.")
        return

    df_times = pd.DataFrame(response_times)

    plt.figure(figsize=(10, 5))

    plt.bar(
        df_times["Test case"],
        df_times["Response time (ms)"],
        color="steelblue"
    )

    plt.title("Experiment 7: API Response Times")
    plt.xlabel("Test case")
    plt.ylabel("Response time (milliseconds)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "response_time_graph.png",
        dpi=150
    )

    plt.close()

    # Save response-time measurements.
    df_times.to_csv(
        REPORT_DIR / "response_times.csv",
        index=False
    )


# ============================================================
# 14. MAIN TEST RUNNER
# ============================================================

def main():
    """Run tests and generate the final report."""

    print("\nEXPERIMENT 7: AUTOMATED TESTING")
    print("=" * 50)

    # Check whether the API is running before sending test requests.
    try:
        response = requests.get(
            HEALTH_URL,
            timeout=3
        )

        if response.status_code != 200:
            print("API is not healthy. Start FastAPI first.")
            return

    except requests.RequestException:
        print("Cannot connect to FastAPI.")
        print("Start the application using:")
        print("uvicorn app:app --reload")
        return

    # Run functional and validation tests.
    test_health()
    test_supported_question()
    test_unsupported_question()
    test_empty_question()
    test_long_question()

    # Run the backend failure test last because it stops Ollama.
    test_backend_failure()

    # Generate the observation table and graphs.
    df = generate_observation_table()

    generate_test_graph(df)
    generate_response_graph()

    # Display a summary.
    passed = (df["Pass/Fail"] == "PASS").sum()
    failed = (df["Pass/Fail"] == "FAIL").sum()

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)

    print("Total tests:", len(df))
    print("Passed:", passed)
    print("Failed:", failed)

    if len(df) > 0:
        print(
            "Pass percentage:",
            round(passed / len(df) * 100, 2),
            "%"
        )

    print("\nReports saved in:", REPORT_DIR.resolve())
    print("- observation_table.csv")
    print("- observation_table.html")
    print("- test_results_graph.png")
    print("- response_time_graph.png")
    print("- response_times.csv")
    print("- test_log.txt")


# Run the complete testing process.
if __name__ == "__main__":
    main()
