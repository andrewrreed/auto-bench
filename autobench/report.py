import os
import json
from dataclasses import asdict
import pandas as pd
import matplotlib.pyplot as plt
from autobench.benchmark import BenchmarkResult


def gather_results(benchmark_result: BenchmarkResult):
    """
    Gather and process benchmark results from a scheduler run.

    This function collects performance metrics from successful deployments
    across different scenarios. It processes data from deployment statuses,
    scenario summaries, and scenario details to compile a comprehensive
    set of results for analysis.

    Args:
        scheduler_results_dir (str): Path to the directory containing
            the scheduler results.

    Returns:
        list: A list of dictionaries, where each dictionary contains
        processed metrics and details for a specific deployment scenario.

    Note:
        This function only considers deployments with a 'success' status.
    """

    metrics_to_keep = {
        "inter_token_latency": {"y": "Time (ms)"},
        "end_to_end_latency": {"y": "Time (ms)"},
        "time_to_first_token": {"y": "Time (ms)"},
        "tokens_throughput": {"y": "Tokens/s"},
        "tokens_received": {"y": "Count"},
    }

    results = []

    for sgr in benchmark_result.scenario_group_results:

        if sgr.deployment_status.get("status") != "success":
            continue  # only consider successful deployments

        for sr in sgr.scenario_results:

            sr = asdict(sr)

            entry = {
                "instance_id": sgr.deployment_details.get("instance_config").get("id"),
                "instance_type": sgr.deployment_details.get("instance_config").get(
                    "instance_type"
                ),
                "scenario_id": sr.get("scenario_id"),
                "executor_type": sr.get("executor_type"),
                "pre_allocated_vus": sr.get("executor_variables").get(
                    "pre_allocated_vus"
                ),
                "rate": sr.get("executor_variables").get("rate"),
                "duration": sr.get("executor_variables").get("duration"),
                "test_duration": sr.get("metrics").get("state").get("testRunDurationMs")
                / 1000.0,
                "requests_ok": sr.get("metrics")
                .get("root_group")
                .get("checks")[0]
                .get("passes"),
                "requests_fail": sr.get("metrics")
                .get("root_group")
                .get("checks")[0]
                .get("fails"),
                "dropped_iterations": (
                    sr.get("metrics")
                    .get("dropped_iterations")
                    .get("values")
                    .get("count")
                    if sr.get("metrics").get("dropped_iterations")
                    else 0
                ),
            }

            # add up requests_fail and dropped_iterations to get total dropped requests
            entry["dropped_requests"] = (
                entry["requests_fail"] + entry["dropped_iterations"]
            )
            entry["error_rate"] = (
                entry["dropped_requests"]
                / (entry["requests_ok"] + entry["dropped_requests"])
                * 100.0
            )

            # get p(90) and count values for metrics_to_keep
            for metric, values in sorted(sr.get("metrics").get("metrics").items()):
                if metric in metrics_to_keep:
                    for value_key, value in values["values"].items():
                        if (
                            value_key == "p(90)" or value_key == "count"
                        ):  # Only keep p(90) values if trend
                            entry[metric] = value

            if "tokens_throughput" in entry and "test_duration" in entry:
                entry["tokens_throughput"] = entry["tokens_throughput"] / (
                    entry["test_duration"]
                )

            if "inter_token_latency" in entry:
                entry["inter_token_latency"] = entry["inter_token_latency"] / 1000.0

            results.append(entry)

    return (
        pd.DataFrame(results)
        .sort_values(by=["instance_id", "rate"])
        .reset_index(drop=True)
    )


def plot_metrics(df: pd.DataFrame):
    """
    Plot performance metrics for different compute instance configurations.

    This function creates a 3x2 grid of subplots, each representing a different
    performance metric. The metrics are plotted against the request rate for
    each instance configuration.

    Args:
        df (pd.DataFrame): A DataFrame containing the benchmark results.
            Expected columns include 'instance_id', 'rate', and various metric columns.

    The function plots the following metrics:
    1. Inter Token Latency (P90)
    2. Time to First Token (P90)
    3. End to End Latency (P90)
    4. Request Output Throughput (P90)
    5. Successful Requests Count
    6. Error Rate

    The resulting plot is saved as a PNG file.
    """
    vus_param = "rate"
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    fig.tight_layout(pad=6.0)
    fig.subplots_adjust(hspace=0.4, wspace=0.2, bottom=0.15)

    names = sorted(df["instance_id"].unique())
    metrics = {
        "inter_token_latency": {"y": "Time (ms)"},
        "time_to_first_token": {"y": "Time (ms)"},
        "end_to_end_latency": {"y": "Time (ms)"},
        "tokens_throughput": {"y": "Tokens/s"},
        "requests_ok": {"y": "Count"},
        "error_rate": {"y": "Count"},
    }
    titles = [
        "Inter Token Latency P90 (lower is better)",
        "TTFT P90 (lower is better)",
        "End to End Latency P90 (lower is better)",
        "Request Output Throughput P90 (higher is better)",
        "Successful requests (higher is better)",
        "Error rate (lower is better)",
    ]
    labels = ["Time (ms)", "Time (ms)", "Time (ms)", "Tokens/s", "Count", "%"]

    # Plot each metric in its respective subplot
    for ax, metric, title, label in zip(axs.flatten(), metrics, titles, labels):
        for i, name in enumerate(names):
            df_sorted = df[df["instance_id"] == name].sort_values(by=vus_param)
            ax.plot(
                df_sorted[vus_param],
                df_sorted[metric],
                marker="o",
                label=f"{name}",
                # color=colors[i],
            )
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=0)
            ax.set_ylabel(label)
            ax.set_xlabel("Requests/s")

            # Add grid lines for better readability
            ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)  # Ensure grid lines are below the bars
            ax.legend(title="Instance", loc="upper right")

    # show title on top of the figure
    plt.suptitle("Constant Arrival Rate Load Test", fontsize=16)
