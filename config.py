import torch
import os

from dataclasses import dataclass, field
from dotenv import load_dotenv


load_dotenv()

@dataclass
class SharedConfig:

    ### Model Config ###
    model_name: str = field(init=False)
    dtype: torch.dtype = torch.float16
    pipeline: list = field(default_factory=list)

    ### Generation Config ###
    tokens_to_generate: int = 30
    prompt: str = "hello world"
    debug: bool = False
    
    ### Networking Config ###
    tailscale_port = 65432

@dataclass
class LocalConfig:
    device: str

    ### Paths Config ###
    model_path: str
    layers_path: str
    handoff_dir: str
    received_dir: str





# ================================================================
# EVERYTHING BUT DEVICE, HANDOFF_DIR, LAYERS_DIR AND RECEIVED_DIR MUST BE SAME
# ACROSS MACHINE_A AND MACHINE_B
# ================================================================

# ================================================================
# MODEL CONFIG
# ================================================================
MODEL_PATH     = "./llama-3b"
STOPPING_LAYER = 15
DTYPE = torch.float16

# ================================================================
# GENERATION CONFIG
# ================================================================
PROMPT             = ("hello world")
TOKENS_TO_GENERATE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = True

# ================================================================
# NETWORK CONFIG 
# ================================================================
MACHINE_A_TAILSCALE_IP = os.getenv("MACHINE_A_TAILSCALE_IP")
TAILSCALE_PORT         = 65432
MSG_FIRST_PASS = 1
MSG_NEXT_PASS  = 2
MSG_TOKEN      = 3
MSG_EOS        = 4
MSG_LAYER      = 5
MSG_TTFT       = 6
MSG_STOP       = 7

ANIRUDH_MACHINE_A = "100.74.100.92"
PRANATHI_MACHINE_A = ""

# ================================================================
# PATHS
# ================================================================
HANDOFF_DIR  = "./handoff"
RECEIVED_DIR = "./received"
LAYERS_DIR   = f"./layers/{os.path.basename(MODEL_PATH)}"