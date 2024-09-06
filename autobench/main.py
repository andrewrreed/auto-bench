from dotenv import load_dotenv

from autobench.compute_manager import ComputeManager
from autobench.scheduler import run_scheduler

load_dotenv(override=True)

VENDOR = "aws"
REGION = "us-east-1"
GPU_TYPES = ["nvidia-a10g", "nvidia-l4"]
GPU_TYPES = ["nvidia-a10g"]
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
NAMESPACE = "andrewrreed"

compute_manager = ComputeManager()
possible_instances = compute_manager.get_instance_details(
    vendor=VENDOR, region=REGION, gpu_types=GPU_TYPES
)

viable_instances = compute_manager.get_viable_instance_configs(
    model_id=MODEL_ID, instances=possible_instances
)

viable_instances = viable_instances[:1]


run_scheduler(viable_instances, NAMESPACE)
