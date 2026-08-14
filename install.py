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

import os
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
    """
    Locate nvidia-smi. It is frequently missing from PATH on Windows, and
    the install location has moved between driver versions.
    """
    found = shutil.which("nvidia-smi")
    if found:
        return found

    candidates = []
    if platform.system() == "Windows":
        candidates = [
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
    else:
        candidates = ["/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"]

    for path in candidates:
        try:
            if os.path.exists(path) and run([path]).returncode == 0:
                return path
        except Exception:
            continue
    return None


# Minimum NVIDIA driver version for each CUDA runtime, used when
# nvidia-smi's header can't be parsed but --query-gpu still works.
_DRIVER_TO_CUDA = [
    (570, (12, 8)),
    (560, (12, 6)),
    (550, (12, 4)),
    (530, (12, 1)),
    (520, (11, 8)),
]


def detect_gpu(verbose=False):
    """
    Returns (driver_cuda, gpu_names).

    driver_cuda is the highest CUDA runtime this driver supports, as a
    (major, minor) tuple, or None if it could not be determined.
    gpu_names is empty when no NVIDIA GPU was found.
    """
    smi = find_nvidia_smi()
    if not smi:
        if verbose:
            print("  nvidia-smi not found on PATH or in the usual locations")
        return None, []
    if verbose:
        print(f"  nvidia-smi: {smi}")

    # GPU names first — this tells us whether a GPU exists at all,
    # independently of whether we can read the CUDA version.
    names = []
    try:
        listing = run([smi, "--query-gpu=name", "--format=csv,noheader"])
        if listing.returncode == 0:
            names = [n.strip() for n in listing.stdout.splitlines() if n.strip()]
        elif verbose:
            print(f"  --query-gpu=name failed: {listing.stderr.strip()[:120]}")
    except Exception as e:
        if verbose:
            print(f"  --query-gpu=name raised: {e}")

    # CUDA version, attempt 1: the banner nvidia-smi prints with no args.
    driver_cuda = None
    try:
        header = run([smi])
        if header.returncode == 0:
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", header.stdout)
            if match:
                driver_cuda = (int(match.group(1)), int(match.group(2)))
            elif verbose:
                print("  could not find 'CUDA Version' in the nvidia-smi banner")
        elif verbose:
            print(f"  nvidia-smi returned {header.returncode}")
    except Exception as e:
        if verbose:
            print(f"  nvidia-smi banner raised: {e}")

    # Attempt 2: derive it from the driver version, which is machine
    # readable and far more stable across nvidia-smi releases.
    if driver_cuda is None:
        try:
            drv = run([smi, "--query-gpu=driver_version",
                       "--format=csv,noheader"])
            if drv.returncode == 0 and drv.stdout.strip():
                major = int(float(drv.stdout.strip().splitlines()[0]))
                for minimum, cuda in _DRIVER_TO_CUDA:
                    if major >= minimum:
                        driver_cuda = cuda
                        break
                if verbose:
                    print(f"  driver {major} -> CUDA {driver_cuda}")
        except Exception as e:
            if verbose:
                print(f"  driver_version query raised: {e}")

    return driver_cuda, names


class UnsupportedDriver(RuntimeError):
    """An NVIDIA GPU is present but no usable CUDA wheel matches its driver."""


def choose_wheel(driver_cuda, gpu_names):
    """
    Pick a wheel tag: 'cuXXX', 'cpu', or 'default' (macOS).

    A machine with an NVIDIA GPU never silently receives the CPU build.
    If the driver is too old or undetectable, this raises so the user is
    told to fix the driver rather than quietly ending up with an install
    that ignores their GPU.
    """
    system = platform.system()

    if system == "Darwin":
        # macOS wheels on PyPI already include MPS support.
        return "default"

    if not gpu_names:
        # Genuinely no NVIDIA GPU — CPU is the correct answer, not a fallback.
        return "cpu"

    is_blackwell = any(
        hint.lower() in name.lower()
        for name in gpu_names
        for hint in BLACKWELL_HINTS
    )

    if driver_cuda is None:
        raise UnsupportedDriver(
            f"Found {gpu_names[0]}, but could not determine which CUDA "
            f"version its driver supports.\n"
            f"  nvidia-smi is present but its output could not be read.\n"
            f"  Run 'python install.py --diagnose' to see the raw output.\n"
            f"  To proceed anyway, pass the CUDA version explicitly:\n"
            f"      python install.py --cuda 12.8\n"
            f"  or force a CPU-only install with --cpu."
        )

    if is_blackwell:
        # Blackwell needs sm_120 kernels, which only exist in cu128 builds.
        # No older wheel will run on this card at all.
        if driver_cuda < (12, 8):
            raise UnsupportedDriver(
                f"{gpu_names[0]} is a Blackwell (RTX 50-series) GPU and "
                f"requires CUDA 12.8 or newer.\n"
                f"  This driver supports up to CUDA "
                f"{driver_cuda[0]}.{driver_cuda[1]}.\n"
                f"  Update your NVIDIA driver, then re-run this installer.\n"
                f"  Download: https://www.nvidia.com/download/index.aspx\n"
                f"  (Older CUDA builds have no kernels for this card, so "
                f"there is no working fallback.)"
            )
        return "cu128"

    for minimum, tag in CUDA_WHEELS:
        if driver_cuda >= minimum:
            return tag

    raise UnsupportedDriver(
        f"Found {gpu_names[0]}, but its driver only supports CUDA "
        f"{driver_cuda[0]}.{driver_cuda[1]}.\n"
        f"  The oldest PyTorch build available needs CUDA 11.8.\n"
        f"  Update your NVIDIA driver, or install the CPU build with:\n"
        f"      python install.py --cpu"
    )


def pip(args, dry_run):
    cmd = [sys.executable, "-m", "pip"] + args
    print("  $ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def diagnose():
    """Show exactly what the hardware probe sees, for when detection fails."""
    print(f"Python {platform.python_version()} on {platform.system()} "
          f"{platform.machine()}\n")

    print("Hardware probe")
    driver_cuda, names = detect_gpu(verbose=True)
    print(f"  GPUs found        : {names or 'none'}")
    print(f"  Driver CUDA level : "
          f"{f'{driver_cuda[0]}.{driver_cuda[1]}' if driver_cuda else 'undetermined'}")

    smi = find_nvidia_smi()
    if smi:
        print("\nRaw nvidia-smi output")
        try:
            out = run([smi]).stdout
            for line in out.splitlines()[:12]:
                print("  " + line)
        except Exception as e:
            print(f"  failed: {e}")

    print("\nWheel selection")
    try:
        print(f"  would install: {choose_wheel(driver_cuda, names)}")
    except UnsupportedDriver as e:
        print(f"  would fail:\n    " + str(e).replace("\n", "\n    "))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPU-only torch")
    ap.add_argument("--cuda", metavar="VERSION",
                    help="force a CUDA variant, e.g. 12.8")
    ap.add_argument("--diagnose", action="store_true",
                    help="show what the hardware probe sees, then exit")
    args = ap.parse_args()

    if args.diagnose:
        return diagnose()

    print(f"Python {platform.python_version()} on {platform.system()} "
          f"{platform.machine()}")

    driver_cuda, gpu_names = detect_gpu()

    if gpu_names:
        for n in gpu_names:
            print(f"  GPU: {n}")
        if driver_cuda:
            print(f"  Driver supports CUDA up to "
                  f"{driver_cuda[0]}.{driver_cuda[1]}")
    else:
        print("  No NVIDIA GPU detected — this machine will run on CPU.")
        print("  That is fine: it can still hold pipeline layers, just fewer.")

    # Resolve the wheel tag
    try:
        if args.cpu:
            tag = "cpu"
            print("  (--cpu given: installing the CPU-only build)")
        elif args.cuda:
            try:
                major, minor = (int(x) for x in args.cuda.split(".")[:2])
            except ValueError:
                print(f"Could not read --cuda {args.cuda}. Use a form like 12.8.")
                return 1
            tag = choose_wheel((major, minor), gpu_names)
            print(f"  (--cuda {args.cuda} given)")
        else:
            tag = choose_wheel(driver_cuda, gpu_names)
    except UnsupportedDriver as e:
        print("\n" + "=" * 68)
        print("Cannot install a working PyTorch build")
        print("=" * 68)
        print(str(e))
        print("=" * 68)
        return 1

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

    return verify(expect_cuda=(tag not in ("cpu", "default")))


def verify(expect_cuda=False):
    """
    Confirm the installed build actually works on this machine.

    Checks more than torch.cuda.is_available(), which returns True even
    when the wheel contains no kernels for the installed card — the
    failure only appears later, mid-inference.
    """
    print("\nVerifying")

    script = """
import torch
print("torch", torch.__version__)
print("built_for_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("device", name)
    print("capability", f"sm_{cap[0]}{cap[1]}")
    # A real matmul: this is what fails with 'no kernel image is
    # available' when the wheel lacks kernels for this architecture.
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    (a @ b).sum().item()
    torch.cuda.synchronize()
    print("kernel_test PASSED (cuda)")
else:
    a = torch.randn(512, 512)
    (a @ a).sum().item()
    print("kernel_test PASSED (cpu)")
"""

    check = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True)

    if check.returncode != 0:
        print(check.stderr.strip())
        if "no kernel image" in check.stderr:
            print("\nThis PyTorch build has no kernels for your GPU.")
            print("Reinstall with a newer CUDA variant:")
            print("    python install.py --cuda 12.8")
        return 1

    out = check.stdout.strip()
    for line in out.splitlines():
        print("  " + line)

    if expect_cuda and "cuda_available True" not in out:
        print("\nA CUDA build was installed but torch cannot see your GPU.")
        print("Run 'python install.py --diagnose' to inspect the hardware probe.")
        return 1

    print("\nReady. Next steps on this machine:")
    print("  1. python benchmark.py <model_name>")
    print("  2. python launch.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
