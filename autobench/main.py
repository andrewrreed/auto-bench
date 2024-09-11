import os
from dotenv import load_dotenv

from autobench.compute_manager import ComputeManager
from autobench.scheduler import run_scheduler
from autobench.logging_config import setup_logging
from autobench.report import gather_results, plot_metrics

setup_logging()
load_dotenv(override=True)

VENDOR = "aws"
REGION = "us-east-1"
GPU_TYPES = ["nvidia-a10g", "nvidia-l4"]
# GPU_TYPES = ["nvidia-a10g"]
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
NAMESPACE = "andrewrreed"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")

compute_manager = ComputeManager()
possible_instances = compute_manager.get_instance_details(
    vendor=VENDOR, region=REGION, gpu_types=GPU_TYPES
)

viable_instances = compute_manager.get_viable_instance_configs(
    model_id=MODEL_ID, instances=possible_instances
)

# viable_instances = viable_instances[:1]


scheduler = run_scheduler(
    viable_instances=viable_instances,
    namespace=NAMESPACE,
    output_dir=RESULTS_DIR,
)


results_df = gather_results(scheduler.output_dir)

plot_metrics(
    df=results_df,
    file_name=os.path.join(scheduler.output_dir, "benchmark_report"),
)
