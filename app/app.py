import gradio as gr
import subprocess
import os
import sys
import shutil
import socket


def run_command(command, env=None, verbose=False):
    print(f"Running command: {command}")
    try:
        if verbose:
            result = subprocess.run(command, shell=True, check=True, text=True, env=env)
            print(f"Command succeeded: {command}")
            return True, ""
        else:
            result = subprocess.run(
                command, shell=True, check=True, text=True, capture_output=True, env=env
            )
            print(f"Command succeeded: {command}")
            print(f"Output: {result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"Error: {e}")
        if not verbose:
            print(f"Output: {e.stdout}")
            print(f"Error output: {e.stderr}")
        return False, e.stderr


def install_go():
    latest_version = "go1.21.0"

    print(f"Installing Go version: {latest_version}")

    commands = [
        f"wget https://go.dev/dl/{latest_version}.linux-amd64.tar.gz",
        f"tar -C /usr/local -xzf {latest_version}.linux-amd64.tar.gz",
        f"echo 'export PATH=$PATH:/usr/local/go/bin' >> $HOME/.bashrc",
    ]

    for command in commands:
        success, _ = run_command(command)
        if not success:
            print(f"Failed to execute command: {command}")
            return False

    # Update PATH in the current environment
    os.environ["PATH"] = f"/usr/local/go/bin:{os.environ.get('PATH', '')}"

    # Verify installation
    success, output = run_command("go version")
    if success:
        print(f"Go installed successfully: {output.strip()}")
        return True
    else:
        print("Failed to verify Go installation")
        return False


def get_go_bin():
    _, output = run_command("go env GOPATH")
    gopath = output.strip()
    if gopath:
        return os.path.join(gopath, "bin")
    return None


# Install Go
if not install_go():
    sys.exit(1)

# Get the Go bin directory
go_bin = get_go_bin()
if go_bin:
    print(f"Found Go bin directory: {go_bin}")
    # Add Go bin to PATH
    os.environ["PATH"] = f"{go_bin}:{os.environ['PATH']}"
else:
    print("Could not find Go bin directory. Using default PATH.")

# Create a directory for k6 build
build_dir = "k6-build"
os.makedirs(build_dir, exist_ok=True)
os.chdir(build_dir)

# Initialize Go module
run_command("go mod init k6-build")

# Run the commands
commands = [
    "go install go.k6.io/xk6/cmd/xk6@latest",
    "which xk6",
    "xk6 build --with github.com/phymbert/xk6-sse@0abbe3e94fe104a13021524b1b98d26447a7d182",
    "mkdir -p ../.bin/",
    "mv k6 ../.bin/k6",
    "../.bin/k6 --version",
]

for command in commands:
    verbose = "xk6 build" in command
    success, output = run_command(command, verbose=verbose)
    if not success:
        if "go install" in command:
            print("Failed to install xk6. Checking Go installation...")
            run_command("go version")
            print("Checking GOPATH...")
            run_command("go env GOPATH")
        elif "xk6" in command:
            print("xk6 command failed. Checking xk6 installation...")
            run_command("which xk6")
            print("Checking Go installation...")
            run_command("go version")
            print("Checking GOPATH...")
            run_command("go env GOPATH")
            print("Listing Go bin directory:")
            if go_bin:
                run_command(f"ls -l {go_bin}")
            print("Checking xk6 version:")
            run_command("xk6 version")
        if "mv k6" in command or "../.bin/k6" in command:
            print("Checking if k6 file exists:")
            run_command("ls -l k6")
            print("Checking .bin directory:")
            run_command("ls -l ../.bin")

print("Script execution completed.")


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()
