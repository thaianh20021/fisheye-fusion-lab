#!/usr/bin/env bash
# Start a CARLA server offscreen without root.
#   tools/carla_server.sh [vulkan-adapter=0] [port=2000]     (CARLA_ROOT, default ~/carla)
# - CARLA needs the Vulkan loader libvulkan.so.1. If the system lacks it (Ubuntu package libvulkan1
#   not installed), it is fetched once with `apt-get download` (no root) into $CARLA_ROOT/vk.
# - Vulkan numbers GPUs differently from nvidia-smi; fflab (Web UI / `fflab carla-server`) finds the
#   mapping by starting the server and checking which GPU it landed on.
set -euo pipefail
ROOT="${CARLA_ROOT:-$HOME/carla}"
BIN="$ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
[ -x "$BIN" ] || { echo "CARLA not found at $ROOT (download CARLA 0.9.16 and extract it there, or set CARLA_ROOT)" >&2; exit 2; }
if ! ldconfig -p | grep -q 'libvulkan.so.1 '; then
  VK="$ROOT/vk/root/usr/lib/x86_64-linux-gnu"
  if [ ! -e "$VK/libvulkan.so.1" ]; then
    echo "libvulkan.so.1 missing: fetching Ubuntu package libvulkan1 into $ROOT/vk (no root needed)" >&2
    mkdir -p "$ROOT/vk/deb"
    (cd "$ROOT/vk/deb" && apt-get download libvulkan1 >/dev/null && dpkg-deb -x libvulkan1_*.deb "$ROOT/vk/root")
  fi
  export LD_LIBRARY_PATH="$VK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$BIN" CarlaUE4 -RenderOffScreen -nosound -graphicsadapter="${1:-0}" -carla-rpc-port="${2:-2000}" -quality-level=Epic
