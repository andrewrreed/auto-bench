import gradio as gr
import subprocess
import os
import sys
import shutil
import socket


def run_command(command):
    print(f"Running command: {command}")
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        print(f"Command succeeded: {command}")
        print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    # Run the commands
    commands = [
        "go install go.k6.io/xk6/cmd/xk6@latest",
        "which xk6",  # Check xk6 installation after it's installed
        "xk6 build --with github.com/phymbert/xk6-sse@0abbe3e94fe104a13021524b1b98d26447a7d182",
        "mkdir -p .bin/",
        "mv k6 .bin/k6",
        ".bin/k6 --version",
    ]

    for command in commands:
        success = run_command(command)
        if not success:
            if "go install" in command:
                print("Failed to install xk6. Checking Go installation...")
                run_command("go version")
                print("Checking GOPATH...")
                run_command("echo $GOPATH")
                break
            elif "xk6 build" in command:
                print("xk6 build command failed. Checking xk6 installation...")
                run_command("which xk6")
                print("Checking Go installation...")
                run_command("go version")
                print("Checking GOPATH...")
                run_command("echo $GOPATH")
                break

    print("Script execution completed.")


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
main()

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
