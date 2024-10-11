import os
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from autobench.compute_manager import ComputeManager
from autobench.scheduler import run_scheduler
from autobench.logging_config import setup_logging
from autobench.report import gather_results, plot_metrics
from huggingface_hub import whoami
from app.setup import setup_k6


if gr.NO_RELOAD:
    setup_k6()
    setup_logging()
    load_dotenv(override=True)

    cm = ComputeManager()


def format_viable_instances(viable_instances):

    data_points = []
    for _, v in viable_instances.items():
        entry = {
            # instance configs
            "gpu_type": v["instance_config"].architecture,
            "num_gpus": v["instance_config"].num_gpus,
            "vendor": v["instance_config"].vendor,
            "region": v["instance_config"].region,
            "gpu_memory_in_gb": v["instance_config"].gpu_memory_in_gb,
            "price_per_hour": v["instance_config"].price_per_hour,
            # tgi configs
            "max_input_tokens": v["tgi_config"].max_input_tokens,
            "max_total_tokens": v["tgi_config"].max_total_tokens,
            "max_batch_prefill_tokens": v["tgi_config"].max_batch_prefill_tokens,
            "estimated_memory_in_gigabytes": v[
                "tgi_config"
            ].estimated_memory_in_gigabytes,
        }
        data_points.append(entry)
    return pd.DataFrame(data_points).sort_values(by=["num_gpus", "gpu_memory_in_gb"])


with gr.Blocks() as demo:
    gr.HTML("<h1>IE AutoBench</h1>")
    gr.HTML(
        "<p>IE AutoBench is a tool for benchmarking the performance of large language models (LLMs) on various compute providers. This tool is currently in development and not all features are available.</p>"
    )

    session_state = gr.State()

    with gr.Row(variant="panel"):
        login_button = gr.LoginButton()
        namespace_selector = gr.Dropdown(label="Namespace", visible=False)
        model_selector = gr.Textbox(
            label="Model ID",
            info="The ID of the model to benchmark. Must be a model supported by TGI.",
            value="meta-llama/Meta-Llama-3-8B-Instruct",
        )

    with gr.Row():
        with gr.Column():
            preferred_vendor_selector = gr.Dropdown(
                label="Preferred Vendor",
                choices=cm.options.vendor.unique().tolist(),
                value="aws",
            )
        with gr.Column():
            preferred_region_selector = gr.Dropdown(
                label="Preferred Region",
                choices=[region[:2] for region in cm.options.region.unique().tolist()],
                value="us",
            )

    gpu_option_selector = gr.CheckboxGroup(
        label="GPU Type",
        choices=cm.options[["architecture", "instance_type"]]
        .apply(tuple, axis=1)
        .unique()
        .tolist(),
    )
    validate_compute_instances_button = gr.Button("Validate Compute Options")

    with gr.Row():
        viable_compute_instances = gr.Dataframe(
            label="Viable Compute Instances", visible=False
        )

    with gr.Row():
        run_benchmark_button = gr.Button("Run Benchmark", visible=False)

    with gr.Row():
        state_display = gr.JSON(label="State")

    with gr.Row():
        session_test = gr.Image(label="Session Test")

    @gr.on(
        triggers=demo.load,
        inputs=[],
        outputs=[namespace_selector],
    )
    def load_demo(oauth_token: gr.OAuthToken | None):
        if oauth_token:
            user_details = whoami(oauth_token.token)
            namespace_options = []
            namespace_options.extend(
                [
                    org["name"]
                    for org in user_details["orgs"]
                    if org.get("canPay", False)
                ]
            )  # add all orgs that can pay
            if user_details["canPay"]:
                namespace_options.insert(
                    0, user_details["name"]
                )  # add user's personal namespace

            if len(namespace_options) == 0:
                gr.Error(
                    "You do not have access to any namespaces that can pay for compute. Please add billing to your account or org."
                )

            return gr.Dropdown(
                choices=namespace_options,
                value=namespace_options[0],
                visible=True,
                interactive=True,
            )

    @gr.on(
        triggers=validate_compute_instances_button.click,
        inputs=[
            model_selector,
            preferred_vendor_selector,
            preferred_region_selector,
            gpu_option_selector,
        ],
        outputs=[
            session_state,
            state_display,
            viable_compute_instances,
            viable_compute_instances,
            preferred_vendor_selector,
            preferred_region_selector,
            gpu_option_selector,
            model_selector,
            validate_compute_instances_button,
            run_benchmark_button,
        ],
    )
    def get_viable_instances(
        model_id, preferred_vendor, preferred_region_prefix, gpu_types
    ):
        possible_instances = cm.get_instance_details(
            gpu_types=gpu_types,
            preferred_vendor=preferred_vendor,
            preferred_region_prefix=preferred_region_prefix,
        )

        viable_instances = cm.get_viable_instance_configs(
            model_id=model_id, instances=possible_instances
        )

        viable_instances = {
            instance["instance_config"].id: instance for instance in viable_instances
        }

        display_viable_instances = format_viable_instances(viable_instances)

        return (
            viable_instances,
            viable_instances,
            display_viable_instances,
            gr.update(visible=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=True),
        )

    @gr.on(
        triggers=run_benchmark_button.click,
        inputs=[
            session_state,
            namespace_selector,
        ],
        outputs=[session_test],
    )
    def run_benchmark(session_state, namespace):

        scheduler = run_scheduler(
            viable_instances=[list(session_state.values())[0]],
            namespace=namespace,
            output_dir=os.path.join(os.path.dirname(__file__), "benchmark_results"),
        )

        results_df = gather_results(scheduler.output_dir)

        plot_path = os.path.join(scheduler.output_dir, "benchmark_report")
        plot_metrics(
            df=results_df,
            file_name=plot_path,
        )

        return gr.Image(plot_path + ".png")


if __name__ == "__main__":
    demo.launch()
