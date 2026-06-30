from transformers import DynamicCache, DynamicLayer
import torch 
import time
import os
from query import Query
from model import Model
import socket
from networking.tailscale import (
    get_my_ip
)
from networking.serialization import (
    tensor_to_bytes,
    tensor_from_bytes,
)

from networking.protocol import (
    send_message,
    read_message
)
from config import (
    DEVICE,
    RECEIVED_DIR,
    HANDOFF_DIR,
    MSG_FIRST_PASS,
    MSG_NEXT_PASS,
    SharedConfig,
    LocalConfig
)

class InferencePeer:
    def __init__(self, shared: SharedConfig, local: LocalConfig):
        ### Shared Config Variable Init ###
        
        self.shared = shared
        self.local = local
        
        # find our entry in the pipeline
        my_assignment = None
        my_index = None
        my_ip = local.tailscale_ip
        for i, entry in enumerate(shared.pipeline):
            if entry["ip"] == my_ip:
                my_assignment = entry
                my_index = i
                break
        
        if my_assignment is None:
            raise ValueError(f"This machine ({my_ip}) is not in the pipeline")
        
         # identity — extracted for convenience, used frequently
        self.my_pipeline_entry = my_assignment
        self.role = my_assignment["role"]
        self.is_master = self.role == "master"
        self.is_tail = self.role == "tail"

        # chain neighbors
        if my_index > 0:
            self.upstream_ip = shared.pipeline[my_index - 1]["ip"]
        else:
            self.upstream_ip = None

        if my_index < len(shared.pipeline) - 1:
            self.downstream_ip = shared.pipeline[my_index + 1]["ip"]
        else:
            None
        self.initiator_ip = shared.initiator_ip

        #model used for this inference peer
        self.model = None
        self.loaded_model_name = None

        #caches 
        self.caches = {}   # Keyed by session ID 
        self.active_session = None

        #connections
        self.upstream_conn = None
        self.downstream_conn = None
        self.shared_port = 65432

    @property
    def cache(self):
        return self.caches.get(self.active_session)
    
    @cache.setter
    def cache(self, value):
        if self.active_session:
            self.caches[self.active_session] = value
    
    def switch_session(self, session_id):
        self.active_session = session_id
        if session_id not in self.caches:
            self.caches[session_id] = None
    
    def load_query_into_model(self, query: Query):
        if self.loaded_model_name != query.model_name:
            if self.model is not None:
                self.model.unload()

            self.model = Model(
                model_name=query.model_name,
                role=self.role,
                layer_start=self.my_pipeline_entry["layers"][0],
                layer_end=self.my_pipeline_entry["layers"][1],
                local_config=self.local,
                dtype=query.dtype,
            )
            if self.role == "master":
                self.model.setup_model_master(self.model)
            elif self.role =="tail":
                self.model.setup_model_tail(self.model)
            else:
                self.model.setup_model_middle(self.model)
            self.model.register_hooks(debug=self.shared.debug)
            self.loaded_model_name = query.model_name
            self.caches.clear()

        cache_key = (query.session_id, query.model_name)
        if self._has_stale_cache(query):
            self._purge_session_caches(query.session_id)
        if cache_key not in self.caches:
            self.caches[cache_key] = None
        self._active_cache_key = cache_key

        self.model.reset_turn_state()

    def connect(self):
        port = self.shared.port    # 65432

        # master listens first — others connect TO the master
        if self.is_master:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", port))
            server.listen(1)
            self.downstream_conn, addr = server.accept()
            server.close()
            print(f"[Peer] Downstream connected from {addr}")

        # tail connects upstream only
        elif self.is_tail:
            self.upstream_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.upstream_conn.connect((self.upstream_ip, port))
            print(f"[Peer] Connected upstream to {self.upstream_ip}")

        # worker connects upstream AND listens for downstream
        else:
            self.upstream_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.upstream_conn.connect((self.upstream_ip, port))
            print(f"[Peer] Connected upstream to {self.upstream_ip}")

            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", port))
            server.listen(1)
            self.downstream_conn, addr = server.accept()
            server.close()
            print(f"[Peer] Downstream connected from {addr}")

    def send_hidden(self, hidden):
        """Send hidden state to downstream neighbor on the INFERENCE connection."""
        payload = tensor_to_bytes(hidden)
        send_message(self.downstream_conn, MSG_FIRST_PASS, payload)

    def receive_hidden(self):
        """Receive hidden state from upstream neighbor on the INFERENCE connection."""
        _, payload = read_message(self.upstream_conn)
        return tensor_from_bytes(payload, device=self.model.device)
    
    def run_generation(self, query):
        if self.is_master:
            return self.run_master_generation(query)
        elif self.is_tail:
            return self.run_tail_generation()
        else:
            return self.run_worker_generation()
        
    def run_master_generation(self, query):
        #get active session

        # get cache for session
        cache = self.get_cache(query.session_id, query.loaded_model_name)
        full_sequence_ids = self.model.tokenize(query.session.messages).to(self.model.device)
        first_pass = True
        token_count = 0
        ttft_start = time.perf_counter()
        ttft = None

        while token_count < query.tokens_to_generate:
            self.model.pass_counter["i"] = token_count
            model_input = full_sequence_ids if first_pass else full_sequence_ids[:, -1:]

            hidden, cache = self.model.forward(model_input, cache)

            if first_pass:
                self.send_hidden(hidden, MSG_FIRST_PASS)
                first_pass = False
            else:
                self.send_hidden(hidden, MSG_NEXT_PASS)

            msg_string, token = self.receive_token()

            if ttft is None:
                ttft = time.perf_counter() - ttft_start

            if msg_string == "eos":
                break

            full_sequence_ids = torch.cat(
                [full_sequence_ids, token.unsqueeze(0).to(full_sequence_ids.device)],
                dim=-1,
            )
            token_count += 1

        self._set_cache(cache)
        response = self.model.decode(self.model.generated_ids)
        return response
    
    def run_worker_generation(self):
        cache = self._get_cache()
        first_pass = True

        while True:
            if first_pass:
                msg_type, hidden = self.receive_hidden(expect=MSG_FIRST_PASS)
                first_pass = False
            else:
                msg_type, hidden = self.receive_hidden(expect=MSG_NEXT_PASS)

            hidden = hidden.to(self.model.device)

            self.model.pass_counter["i"] += 1
            hidden, cache = self.model.forward(hidden, cache)

            self.send_hidden(hidden, msg_type)

            # worker doesn't know when generation ends — it follows
            # the upstream. When upstream stops sending, receive_hidden
            # either times out or gets a MSG_EOS/MSG_STOP.
            # For now, the loop breaks on connection close.

        self._set_cache(cache)

    def run_tail_generation(self, query):
        cache = self._get_cache()
        first_pass = True
        token_count = 0

        while True:
            if first_pass:
                msg_type, hidden = self.receive_hidden(expect=MSG_FIRST_PASS)
                first_pass = False
            else:
                msg_type, hidden = self.receive_hidden(expect=MSG_NEXT_PASS)

            hidden = hidden.to(self.model.device)

            self.model.pass_counter["i"] = token_count
            token, cache = self.model.forward(hidden, cache)

            # check EOS
            eos_ids = self.model.tokenizer.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]

            if token.item() in eos_ids:
                self.send_eos()
                break

            self.model.generated_ids.append(token.item())
            self.send_token(token)
            token_count += 1

            if token_count >= query.tokens_to_generate:
                break

        self._set_cache(cache)
    

def load_handoff_package(save_dir=RECEIVED_DIR, first_pass=True):
    if first_pass:
        hidden = torch.load(f"{save_dir}/hidden.pt", map_location=DEVICE)
        cos = torch.load(f"{save_dir}/cos.pt", map_location=DEVICE)
        sin = torch.load(f"{save_dir}/sin.pt", map_location=DEVICE)
        position_embeddings = (cos, sin)
        position_ids = torch.load(f"{save_dir}/position_ids.pt", map_location=DEVICE)
        return hidden, position_embeddings, position_ids
    else:
        hidden = torch.load(f"{save_dir}/hidden.pt", map_location=DEVICE)
        return hidden
    
def save_handoff_package(hidden, position_embeddings, position_ids, save_dir=HANDOFF_DIR):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden, f"{save_dir}/hidden.pt")
    torch.save(position_embeddings[0], f"{save_dir}/cos.pt")
    torch.save(position_embeddings[1], f"{save_dir}/sin.pt")
    torch.save(position_ids, f"{save_dir}/position_ids.pt")