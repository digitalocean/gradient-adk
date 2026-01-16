#!/usr/bin/env python3
import subprocess
import uuid
import time
import sys

NUM_DEPLOYS = 15
ENTRYPOINT_FILE = "main.py"
DEPLOYMENT_NAME = "test"


def run(cmd):
    """Run a shell command synchronously and stream output."""
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    start = time.time()
    for line in process.stdout:
        print(line, end="")
        # Once we detect the "Building" phase, we can move on
        if "Building..." in line:
            print(
                f"Detected build start after {time.time() - start:.1f}s — moving on\n"
            )
            break

    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def main():
    for i in range(NUM_DEPLOYS):
        workspace = f"test-{uuid.uuid4().hex[:8]}"
        print(f"\n=== [{i+1}/{NUM_DEPLOYS}] Deploying workspace {workspace} ===")

        # 1️⃣ Configure
        configure_cmd = (
            f"gradient agent configure "
            f"--agent-workspace-name={workspace} "
            f"--deployment-name={DEPLOYMENT_NAME} "
            f"--entrypoint-file={ENTRYPOINT_FILE}"
        )
        print(f"Running: {configure_cmd}")
        subprocess.run(configure_cmd, shell=True, check=True)

        # 2️⃣ Deploy (stop after build starts)
        deploy_cmd = "gradient agent deploy --skip-validation"
        print(f"Running: {deploy_cmd}")
        run(deploy_cmd)

    print("\n✅ All deploys launched.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
