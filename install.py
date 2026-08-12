"""
install.py

Installs the right PyTorch build for this machine, then everything in
requirements.txt.

PyTorch ships a separate wheel per CUDA version. Which one works depends
on the NVIDIA driver installed and the GPU's compute capability — neither
of which pip can detect from a requirements file. This script reads the
driver via nvidia-smi, picks the matching wheel index, and installs.

    python install.py              # detect and install
    python install.py --dry-run    # show what it would do
    python install.py --cpu        # force CPU-only build
    python install.py --cuda 12.4  # force a specific CUDA variant

Run this on every machine that will participate in inference.
"""

import re
import sys
import shutil
import argparse
import platform
import subprocess

# Highest-to-lowest. Each entry: (minimum driver-reported CUDA, wheel tag)
CUDA_WHEELS = [
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
]

INDEX = "https://download.pytorch.org/whl/{tag}"

# Blackwell (RTX 50-series) needs sm_120 kernels, which only exist in
# cu128 builds. Detecting by name is cruder than compute capability, but
# we cannot query capability before torch is installed.
BLACKWELL_HINTS = ("RTX 50", "RTX PRO 6000", "B100", "B200", "GB200")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def find_nvidia_smi():
    """nvidia-smi is often absent from PATH on Windows."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if platform.system() == "Windows":
        fallback = r"C:\Windows\System32\nvidia-smi.exe"
        try:
            if run([fallback]).returncode == 0:
                return fallback
        except Exception:
            pass
    return None


def detect_gpu():
    """
    Returns (driver_cuda, gpu_names) where driver_cuda is a (major, minor)
    tuple of the highest CUDA runtime this driver supports, or None if
    there is no usable NVIDIA GPU.
    """
    smi = find_nvidia_smi()
    if not smi:
        return None, []

    try:
        header = run([smi])
    except Exception:
        return None, []

    if header.returncode != 0:
        return None, []

    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", header.stdout)
    driver_cuda = (int(match.group(1)), int(match.group(2))) if match else None

    names = []
    try:
        listing = run([smi, "--query-gpu=name", "--format=csv,noheader"])
        if listing.returncode == 0:
            names = [n.strip() for n in listing.stdout.splitlines() if n.strip()]
    except Exception:
        pass

    return driver_cuda, names


def choose_wheel(driver_cuda, gpu_names):
    """Pick a wheel tag: a cuXXX string, or 'cpu'."""
    system = platform.system()

    if system == "Darwin":
        # macOS wheels on PyPI already include MPS support.
        return "default"

    if driver_cuda is None:
        return "cpu"

    if any(h.lower() in n.lower() for n in gpu_names for h in BLACKWELL_HINTS):
        if driver_cuda >= (12, 8):
            return "cu128"
        print("  ! Blackwell GPU detected but the driver does not support "
              "CUDA 12.8.\n    Update your NVIDIA driver, or this GPU will "
              "fall back to CPU.")

    for minimum, tag in CUDA_WHEELS:
        if driver_cuda >= minimum:
            return tag

    return "cpu"


def pip(args, dry_run):
    cmd = [sys.executable, "-m", "pip"] + args
    print("  $ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPU-only torch")
    ap.add_argument("--cuda", metavar="VERSION",
                    help="force a CUDA variant, e.g. 12.4")
    args = ap.parse_args()

    print(f"Python {platform.python_version()} on {platform.system()} "
          f"{platform.machine()}")

    driver_cuda, gpu_names = detect_gpu()

    if gpu_names:
        for n in gpu_names:
            print(f"  GPU: {n}")
        if driver_cuda:
            print(f"  Driver supports CUDA up to {driver_cuda[0]}.{driver_cuda[1]}")
    else:
        print("  No NVIDIA GPU detected — this machine will run on CPU.")
        print("  That is fine: it can still hold pipeline layers, just fewer.")

    # Resolve the wheel tag
    if args.cpu:
        tag = "cpu"
    elif args.cuda:
        try:
            major, minor = (int(x) for x in args.cuda.split(".")[:2])
        except ValueError:
            print(f"Could not read --cuda {args.cuda}. Use a form like 12.4.")
            return 1
        tag = choose_wheel((major, minor), gpu_names)
    else:
        tag = choose_wheel(driver_cuda, gpu_names)

    print(f"\nInstalling PyTorch build: {tag}")

    torch_spec = ["torch>=2.6"]
    if tag == "default":
        code = pip(["install"] + torch_spec, args.dry_run)
    else:
        code = pip(["install"] + torch_spec +
                   ["--index-url", INDEX.format(tag=tag)], args.dry_run)

    if code != 0:
        print("\nPyTorch install failed. Pick a build manually at "
              "https://pytorch.org/get-started/locally/")
        return code

    print("\nInstalling remaining dependencies")
    code = pip(["install", "-r", "requirements.txt"], args.dry_run)
    if code != 0:
        return code

    if args.dry_run:
        print("\nDry run — nothing was installed.")
        return 0

    return verify()


def verify():
    """Confirm torch imports and can actually run a kernel on this GPU."""
    print("\nVerifying")
    check = subprocess.run(
        [sys.executable, "-c",
         "import torch;"
         "print('torch', torch.__version__);"
         "print('cuda_available', torch.cuda.is_available());"
         "print('device', torch.cuda.get_device_name(0) "
         "if torch.cuda.is_available() else 'cpu');"
         "torch.zeros(8).to('cuda' if torch.cuda.is_available() else 'cpu').sum()"],
        capture_output=True, text=True,
    )

    if check.returncode != 0:
        print(check.stderr.strip())
        if "no kernel image" in check.stderr:
            print("\nThis torch build has no kernels for your GPU. "
                  "Reinstall with a newer CUDA variant:\n"
                  "    python install.py --cuda 12.8")
        return 1

    for line in check.stdout.strip().splitlines():
        print("  " + line)

    print("\nReady. Next steps on this machine:")
    print("  1. python benchmark.py <model_name>")
    print("  2. python launch.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
