import os
import json

import pandas as pd
import matplotlib.pyplot as plt


def gather_results(scheduler_results_dir: str):
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

    with open(
        os.path.join(scheduler_results_dir, "deployment_statuses.json"), "r"
    ) as f:
        deployment_statuses = json.load(f)

    deployment_statuses = [
        status for status in deployment_statuses if status["status"] == "success"
    ]  # only consider successful deployments

    results = []

    for status in deployment_statuses:
        deployment_name = status["deployment_name"]
        deployment_dir = os.path.join(scheduler_results_dir, deployment_name)

        for scenario_dir in os.listdir(deployment_dir):
            scenario_path = os.path.join(deployment_dir, scenario_dir)

            if not os.path.isdir(scenario_path):
                continue

            with open(os.path.join(scenario_path, "summary.json"), "r") as f:
                result_summary = json.load(f)

            with open(os.path.join(scenario_path, "scenario_details.json"), "r") as f:
                scenario_details = json.load(f)

            entry = {
                "instance_id": status["instance_id"],
                "instance_type": status["instance_type"],
                "scenario_id": scenario_details["scenario_id"],
                "executor_type": scenario_details["executor_type"],
                "pre_allocated_vus": scenario_details["pre_allocated_vus"],
                "rate": scenario_details["rate"],
                "duration": scenario_details["duration"],
            }

            entry["test_duration"] = (
                result_summary["state"]["testRunDurationMs"] / 1000.0
            )
            entry["requests_ok"] = result_summary["root_group"]["checks"][0]["passes"]
            entry["requests_fail"] = result_summary["root_group"]["checks"][0]["fails"]
            entry["dropped_iterations"] = (
                result_summary["metrics"]["dropped_iterations"]["values"]["count"]
                if "dropped_iterations" in result_summary["metrics"]
                else 0
            )

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
            for metric, values in sorted(result_summary["metrics"].items()):
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


def plot_metrics(df: pd.DataFrame, file_name: str):
    """
    Plot performance metrics for different compute instance configurations.

    This function creates a 3x2 grid of subplots, each representing a different
    performance metric. The metrics are plotted against the request rate for
    each instance configuration.

    Args:
        df (pd.DataFrame): A DataFrame containing the benchmark results.
            Expected columns include 'instance_id', 'rate', and various metric columns.
        file_name (str): The base name for the output file (without extension).

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
    plt.savefig(f"{file_name}.png")
