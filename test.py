from networking.tailscale import (
    get_my_ip,
    get_status,
    get_online_peers
)

#print(get_status())
print(get_my_ip())
print(get_online_peers())