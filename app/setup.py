import subprocess
import os
import sys
from loguru import logger


from autobench.logging_config import setup_logging

setup_logging()


def run_command(command, env=None, verbose=False):
    logger.info(f"Running command: {command}")
    try:
        if verbose:
            result = subprocess.run(command, shell=True, check=True, text=True, env=env)
            logger.success(f"Command succeeded: {command}")
            return True, ""
        else:
            result = subprocess.run(
                command, shell=True, check=True, text=True, capture_output=True, env=env
            )
            logger.success(f"Command succeeded: {command}")
            logger.info(f"Output: {result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {e}")
        if not verbose:
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error output: {e.stderr}")
        return False, e.stderr


def install_go():
    VERSION = "go1.21.0"

    logger.info(f"Installing Go version: {VERSION}")

    commands = [
        f"wget https://go.dev/dl/{VERSION}.linux-amd64.tar.gz",
        f"tar -C /usr/local -xzf {VERSION}.linux-amd64.tar.gz",
        "echo 'export PATH=$PATH:/usr/local/go/bin' >> $HOME/.bashrc",
        "rm -rf {VERSION}.linux-amd64.tar.gz",
    ]

    for command in commands:
        success, _ = run_command(command)
        if not success:
            logger.error(f"Failed to execute command: {command}")
            return False

    # Update PATH in the current environment
    os.environ["PATH"] = f"/usr/local/go/bin:{os.environ.get('PATH', '')}"

    # Verify installation
    success, output = run_command("go version")
    if success:
        logger.success(f"Go installed successfully: {output.strip()}")
        return True
    else:
        logger.error("Failed to verify Go installation")
        return False


def get_go_bin():
    _, output = run_command("go env GOPATH")
    gopath = output.strip()
    if gopath:
        return os.path.join(gopath, "bin")
    return None


# Install Go
def setup_k6():

    # check if k6 is already installed
    success, _ = run_command("k6 --version")
    if not success:
        logger.error("k6 is not installed. Installing k6...")
    else:
        logger.success("k6 is already installed")
        return True

    # check if go is installed
    success, _ = run_command("go version")
    if not success:
        logger.error("Go is not installed. Installing Go...")
        if not install_go():
            sys.exit(1)
    else:
        logger.success("Go is already installed")

    # Get the Go bin directory
    go_bin = get_go_bin()
    if go_bin:
        logger.info(f"Found Go bin directory: {go_bin}")
        # Add Go bin to PATH
        os.environ["PATH"] = f"{go_bin}:{os.environ['PATH']}"
    else:
        logger.warning("Could not find Go bin directory. Using default PATH.")

    # Create a directory for k6 build
    build_dir = "k6-build"
    os.makedirs(build_dir, exist_ok=True)
    os.chdir(build_dir)

    # Initialize Go module
    run_command("go mod init k6-build")

    # Run the commands to install k6-sse
    commands = [
        "go install go.k6.io/xk6/cmd/xk6@latest",
        "which xk6",
        "xk6 build --with github.com/phymbert/xk6-sse@0abbe3e94fe104a13021524b1b98d26447a7d182",
        # move executable to .bin
        "mkdir -p ../.bin/",
        "mv k6 ../.bin/k6",
        "../.bin/k6 --version",
        # clean up
        "rm -rf k6-build",
    ]

    for command in commands:
        verbose = "xk6 build" in command
        success, output = run_command(command, verbose=verbose)
        if not success:
            if "go install" in command:
                logger.error("Failed to install xk6. Checking Go installation...")
                run_command("go version")
                logger.info("Checking GOPATH...")
                run_command("go env GOPATH")
            elif "xk6" in command:
                logger.error("xk6 command failed. Checking xk6 installation...")
                run_command("which xk6")
                logger.info("Checking Go installation...")
                run_command("go version")
                logger.info("Checking GOPATH...")
                run_command("go env GOPATH")
                logger.info("Listing Go bin directory:")
                if go_bin:
                    run_command(f"ls -l {go_bin}")
                logger.info("Checking xk6 version:")
                run_command("xk6 version")
            if "mv k6" in command or "../.bin/k6" in command:
                logger.info("Checking if k6 file exists:")
                run_command("ls -l k6")
                logger.info("Checking .bin directory:")
                run_command("ls -l ../.bin")

    # check if k6-sse is installed
    success, _ = run_command("../.bin/k6 --version")
    if not success:
        logger.error("k6-sse is not installed. Exiting...")
        return False
    else:
        logger.success("k6-sse installed successfully")

        # set k6 executable path to env
        os.environ["K6_EXE"] = os.path.abspath("../.bin/k6")
        logger.info(f"K6_EXE directory set as env var: {os.environ['K6_EXE']}")

        return True
