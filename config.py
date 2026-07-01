import torch
import os
import json
from dataclasses import dataclass
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

    CONFIG_PATH = "./config/local_config.json"

    def save(self, path=None):
        """Persist current settings to disk."""
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
            layers_path=os.getenv("LAYERS_PATH",
                                saved.get("layers_path", "./layers")),
            model_path=os.getenv("MODEL_PATH",
                                saved.get("model_path", "./models")),
            debug=os.getenv("DEBUG", str(saved.get("debug", False))).lower() == "true",
            tailscale_ip=os.getenv("TAILSCALE_IP", saved.get("tailscale_ip", "")),
        )

# ================================================================
# PROTOCOL CONSTANTS
# ================================================================

# Inference data plane
MSG_FIRST_PASS = 1
MSG_NEXT_PASS  = 2
MSG_TOKEN      = 3
MSG_EOS        = 4
MSG_LAYER      = 5
MSG_TTFT       = 6
MSG_STOP       = 7
MSG_RESPONSE   = 8

# Daemon control plane
MSG_PING           = 20
MSG_PONG           = 21
MSG_BENCHMARK_REQ  = 22
MSG_BENCHMARK_RESP = 23
MSG_BENCHMARK_MISS = 24
MSG_CONFIG         = 25
MSG_READY          = 26
MSG_START          = 31

# Legacy convenience (remove when old scripts are deleted)
TAILSCALE_PORT = 65432
