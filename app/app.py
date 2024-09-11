import gradio as gr
import subprocess
import os
import sys
import shutil
import socket


# def run_command(command, cwd=None):
#     try:
#         process = subprocess.Popen(
#             command,
#             shell=True,
#             cwd=cwd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#         )
#         stdout, stderr = process.communicate()
#         if process.returncode != 0:
#             print(f"Command '{command}' failed with return code {process.returncode}")
#             print(f"STDOUT: {stdout}")
#             print(f"STDERR: {stderr}")
#             raise subprocess.CalledProcessError(
#                 process.returncode, command, stdout, stderr
#             )
#         return stdout
#     except subprocess.CalledProcessError as e:
#         print(f"Error executing command: {e}")
#         print(f"STDOUT: {e.stdout}")
#         print(f"STDERR: {e.stderr}")
#         raise


# def check_git_config():
#     try:
#         # user_name = run_command("git config --get user.name").strip()
#         # user_email = run_command("git config --get user.email").strip()
#         print(f"Git user.name: {user_name}")
#         print(f"Git user.email: {user_email}")
#     except subprocess.CalledProcessError:
#         print("Git configuration not set.")


# def main():
#     temp_dir = "/tmp/xk6"
#     xk6_sse_dir = os.path.join(temp_dir, "xk6-sse")
#     local_bin_dir = os.path.expanduser("~/.local/bin/")

#     print(f"Python version: {sys.version}")
#     print(f"Current working directory: {os.getcwd()}")
#     print(f"Contents of current directory: {os.listdir()}")

#     # check_git_config()

#     # Create temporary directory
#     os.makedirs(temp_dir, exist_ok=True)
#     print(f"Created directory: {temp_dir}")

#     # Change to temporary directory
#     original_dir = os.getcwd()
#     os.chdir(temp_dir)
#     print(f"Changed to directory: {temp_dir}")

#     try:
#         # Clone repository
#         print("Attempting to clone repository...")
#         # run_command("git clone https://github.com/mstoykov/xk6-sse.git")
#         run_command("git clone git://github.com/mstoykov/xk6-sse.git")
#         print("Repository cloned successfully")

#         # Change to xk6-sse directory
#         os.chdir(xk6_sse_dir)
#         print(f"Changed to directory: {xk6_sse_dir}")

#         # Checkout specific branch
#         run_command("git checkout useSobek")

#         # Install xk6
#         run_command("go install go.k6.io/xk6/cmd/xk6@latest")

#         # Build xk6 with SSE plugin
#         run_command("xk6 build --with github.com/phymbert/xk6-sse=.")

#         # Create local bin directory
#         os.makedirs(local_bin_dir, exist_ok=True)

#         # Move k6 to local bin directory
#         shutil.move("k6", os.path.join(local_bin_dir, "k6-sse"))

#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         # Change back to original directory
#         os.chdir(original_dir)
#         print(f"Changed back to directory: {original_dir}")


# def run_command(command, shell=False, env=None):
#     result = subprocess.run(
#         command, shell=shell, check=True, text=True, capture_output=True, env=env
#     )
#     print(f"Command output:\n{result.stdout}")
#     if result.stderr:
#         print(f"Command error:\n{result.stderr}")


# def build_k6_sse():
#     # Clean Go module cache
#     run_command(["go", "clean", "-modcache"])

#     # Create temporary directory
#     temp_dir = "/tmp/xk6"
#     os.makedirs(temp_dir, exist_ok=True)

#     # Change to temporary directory
#     original_dir = os.getcwd()
#     os.chdir(temp_dir)

#     try:
#         # Install xk6
#         run_command(["go", "install", "go.k6.io/xk6/cmd/xk6@latest"])

#         # Build custom k6 binary
#         env = os.environ.copy()
#         env["GOFLAGS"] = "-mod=mod"
#         run_command(
#             [
#                 "xk6",
#                 "build",
#                 "master",
#                 "--with",
#                 "github.com/andrewrreed/xk6-sse@a24fd84",
#             ],
#             env=env,
#         )

#         # Create local bin directory
#         local_bin = os.path.expanduser("~/.local/bin")
#         os.makedirs(local_bin, exist_ok=True)

#         # Move and rename the built binary
#         shutil.move("k6", os.path.join(local_bin, "k6-sse"))

#     finally:
#         # Return to original directory
#         os.chdir(original_dir)

#     print("k6 with SSE support has been built and installed to ~/.local/bin/k6-sse")


def run_command(command):
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        print(f"Command succeeded: {command}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"Error: {e}")
        print(f"Output: {e.output}")


def main():
    # Run the commands
    commands = [
        "go install go.k6.io/xk6/cmd/xk6@latest",
        "xk6 build --with github.com/phymbert/xk6-sse@0abbe3e94fe104a13021524b1b98d26447a7d182",
        "mkdir -p .bin/",
        "mv k6 .bin/k6",
        ".bin/k6 --version",
    ]

    for command in commands:
        print(f"Running command: {command}")
        run_command(command)
    print("Command completed")


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
