# auto-bench

`auto-bench` is a flexible tool for benchmarking LLMs on [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index). It provides an automated way to deploy models, run load tests, and analyze performance across different hardware configurations.

## Features
- Automated deployment of models to Hugging Face Inference Endpoints
- Configurable load testing scenarios using K6
- Support for various GPU instances
- Detailed performance metrics collection and analysis
- Easy-to-use Python API for creating and running benchmarks

## Metrics
`auto-bench` relies on [Grafana K6](https://grafana.com/docs/k6/latest/) to run load tests and collect metrics. The following metrics are collected:

- __Inter token latency:__ Time to generate a new output token for each user that is querying the system. It translates as the “speed” perceived by the end-user.
- __Time to First Token:__ Time the user has to wait before seeing the first token of its answer. Lower waiting time are essential for real-time interactions.
- __End to End latency:__ The overall time the system took to generate the full response to the user.
- __Throughput:__ The number of tokens per second the system can generate across all requests.
- __Successful requests:__ The number of requests the system was able to honor in the benchmark timeframe.
- __Error rate:__ The percentage of requests that ended up in error, as the system could not process them in time or failed to process them.

## Setup
To get started with `auto-bench`, follow these steps:

1. Clone the repository:

```
git clone https://github.com/andrewrreed/auto-bench.git
```

2. Set up a virtual environment and activate it:

```
python -m venv .venv
source .venv/bin/activate
```

3. Build the custom K6 binary with SSE support:
```
make build-k6
```

4. Install the required Python packages:

```
poetry install
```

## Getting Started
Check out the [Getting Started Notebook](./notebooks/getting_started.ipynb) to get familiar with basic usage.

## Contact
For questions or suggestions, please open an issue on the GitHub repository.