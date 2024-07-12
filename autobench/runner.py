from dataclasses import asdict

from autobench.utils import TGIConfig, ComputeInstanceConfig, ComputeOptionUtility
from autobench.deploy import IEDeployment
from autobench.k6 import K6Config, K6ConstantArrivalRateExecutor, K6Benchmark

from recommender.main import get_tgi_config


class BenchmarkRunner:
    def __init__(self, model_id, vendor, region, gpu_types):
        self.model_id = model_id
        self.vendor = vendor
        self.region = region
        self.gpu_types = gpu_types
        self.compute_option_util = ComputeOptionUtility()

    def run_benchmark(self):

        # Lookup IE instances with specified vendor, region and gpu_type
        instances = self.compute_option_util.get_instance_details(
            vendor=self.vendor, region=self.region, gpu_types=self.gpu_types
        )

        # Filter instances that can run the model + gather TGI config
        viable_instances = self.compute_option_util.get_viable_instance_configs(
            model_id=self.model_id, instances=instances
        )

        # Deploy and run benchmark on each viable instance
        for tgi_config, instance_config in viable_instances:

            tgi_config = TGIConfig(**tgi_config)
            instance_config = ComputeInstanceConfig(**instance_config)

            try:
                deployment = IEDeployment(
                    tgi_config=tgi_config, instance_config=instance_config
                )
                deployment.deploy_endpoint()

                ## TO-DO: Run benchmark, spin down endpoint

            except Exception as e:
                print(e)
