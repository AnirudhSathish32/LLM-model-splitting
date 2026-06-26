import torch
import os
import json
from dataclasses import dataclass, field
from dotenv import load_dotenv


load_dotenv()

@dataclass
class SharedConfig:
    debug: bool
    port: int
    initiator_ip: str
    pipeline: list



@dataclass
class LocalConfig:
    device: str
    debug: bool
    tailscale_ip: str

    ### Paths Config ###
    model_path: str
    layers_path: str
    handoff_dir: str
    received_dir: str

    CONFIG_PATH = "./config/local_config.json"

    def save(self, path=None):
        """
        persist current settings to disk
        """
        path = path or self.CONFIG_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "device": self.device,
            "layers_path": self.layers_path,
            "model_path": self.model_path,
            "debug": self.debug,
            "tailscale_ip": self.tailscale_ip,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path=None):
        """
        Load priority:
          1. Saved file (user's UI choices)
          2. Environment variables (CLI override)
          3. Defaults (first run)
        
        Env vars override saved file when set, so a deployment
        can force a setting regardless of what the UI saved.
        """
        path = path or cls.CONFIG_PATH
        saved = {}
        if os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)

        return cls(
            device=os.getenv("DEVICE",
                            saved.get("device",
                                    "cuda" if torch.cuda.is_available() else "cpu")),
            layers_dir=os.getenv("LAYERS_DIR",
                                saved.get("layers_dir", "./layers")),
            model_dir=os.getenv("MODEL_DIR",
                                saved.get("model_dir", "./models")),
            debug=os.getenv("DEBUG", str(saved.get("debug", False))).lower() == "true",
            tailscale_ip=os.getenv("TAILSCALE_IP", saved.get("tailscale_ip", "")),
        )

                




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
MSG_PROFILE    = 8
MSG_PING       = 9
MSG_BENCHMARK_REQ = 10
MSG_PONG       = 11
MSG_BENCHMARK_MISS = 12
MSG_BENCHMARK_RESP = 13

ANIRUDH_MACHINE_A = "100.74.100.92"
PRANATHI_MACHINE_A = ""

# ================================================================
# PATHS
# ================================================================
HANDOFF_DIR  = "./handoff"
RECEIVED_DIR = "./received"
LAYERS_DIR   = f"./layers/{os.path.basename(MODEL_PATH)}"