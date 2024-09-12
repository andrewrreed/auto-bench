import gradio as gr


from autobench.compute_manager import ComputeManager
from app.setup import setup_k6


# install go and k6-sse
setup_k6()


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()
