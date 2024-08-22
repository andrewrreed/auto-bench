from autobench.k6 import K6Config, K6ConstantArrivalRateExecutor, K6Benchmark


def main():
    executor = K6ConstantArrivalRateExecutor(
        pre_allocated_vus=10, rate_per_second=1, duration="20s"
    )
    config = K6Config(
        host="https://dpv7afomnocq8b4l.us-east-1.aws.endpoints.huggingface.cloud",
        executor=executor,
        data_file="small.json",
    )
    benchmark = K6Benchmark(config=config, output_dir="results")
    benchmark.run()


if __name__ == "__main__":
    main()
