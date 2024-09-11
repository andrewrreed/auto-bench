import gradio as gr
import subprocess
import os
import shutil


def run_command(command, shell=False, env=None):
    result = subprocess.run(
        command, shell=shell, check=True, text=True, capture_output=True, env=env
    )
    print(f"Command output:\n{result.stdout}")
    if result.stderr:
        print(f"Command error:\n{result.stderr}")


def build_k6_sse():
    # Clean Go module cache
    run_command(["go", "clean", "-modcache"])

    # Create temporary directory
    temp_dir = "/tmp/xk6"
    os.makedirs(temp_dir, exist_ok=True)

    # Change to temporary directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Install xk6
        run_command(["go", "install", "go.k6.io/xk6/cmd/xk6@latest"])

        # Build custom k6 binary
        env = os.environ.copy()
        env["GOFLAGS"] = "-mod=mod"
        run_command(
            [
                "xk6",
                "build",
                "master",
                "--with",
                "github.com/andrewrreed/xk6-sse@a24fd84",
            ],
            env=env,
        )

        # Create local bin directory
        local_bin = os.path.expanduser("~/.local/bin")
        os.makedirs(local_bin, exist_ok=True)

        # Move and rename the built binary
        shutil.move("k6", os.path.join(local_bin, "k6-sse"))

    finally:
        # Return to original directory
        os.chdir(original_dir)

    print("k6 with SSE support has been built and installed to ~/.local/bin/k6-sse")


def check_k6_sse():
    k6_sse_path = os.path.expanduser("~/.local/bin/k6-sse")

    if not os.path.exists(k6_sse_path):
        print("k6-sse executable not found at ~/.local/bin/k6-sse")
        return False

    try:
        result = subprocess.run(
            [k6_sse_path, "version"], capture_output=True, text=True, check=True
        )
        print(f"k6-sse version: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running k6-sse: {e}")
        return False


# Build k6-sse
build_k6_sse()

# Check if k6-sse exists and works
if check_k6_sse():
    print("k6-sse is installed and working correctly")
else:
    print(
        "k6-sse is not installed or not working. Please run build_k6_sse() to install it."
    )


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()
