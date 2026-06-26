# tailscale.py
import json
import subprocess

def get_status():
    result = subprocess.run(
        ["tailscale", "status", "--json"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tailscale status failed: {result.stderr}")
    return json.loads(result.stdout)

def get_my_ip():
    return get_status()["Self"]["TailscaleIPs"][0]

def get_online_peers():
    status = get_status()
    peers = []
    for peer in status.get("Peer", {}).values():
        if peer.get("Online", False):
            peers.append({
                "ip":       peer["TailscaleIPs"][0],
                "hostname": peer.get("HostName", ""),
                "os":       peer.get("OS", ""),
            })
    return peers