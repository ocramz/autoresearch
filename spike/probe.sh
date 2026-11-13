#!/usr/bin/env bash
# Probe (pi-autoresearch-serverless, plan 17b, step 1): what a GPU task's container gives a sandbox
# uid. Runs as root, prints what the runtime injected (devices, driver mounts, sockets), then runs
# CUDA as uid 65534 with no groups, no capabilities and NO_NEW_PRIVS, as sbx would, with a fresh
# HOME and TMPDIR: a matmul, and a torch.compile so that Triton builds its launcher with gcc. It goes
# into the job by `spike/cloud.sh probe`, as an argument, so that any built image serves.
set -uo pipefail
cd /app 2>/dev/null || cd /

stage() { echo "=== $1 $(date -u +%FT%TZ)"; }

stage identity
id
cat /proc/self/status | grep -E '^(Cap|NoNewPrivs|Groups)'

stage devices
ls -lan /dev
ls -lan /dev/nvidia* 2>&1
ls -lan /dev/nvidia-caps 2>&1 || true
ls -lan /dev/dri 2>&1 || true

stage driver
ls -lan /usr/local/nvidia /usr/local/nvidia/* 2>&1 | head -60
stat -c '%n %U(%u):%G(%g) %a %F' /usr/local/nvidia /usr/local/nvidia/* 2>&1

stage mounts
cat /proc/self/mountinfo

stage sockets
find / -xdev \( -path /proc -o -path /sys \) -prune -o -type s -print 2>/dev/null
cat /proc/net/unix | awk 'NR == 1 || $8 ~ /^@/'

stage world-writable
find / \( -path /proc -o -path /sys \) -prune -o -type d -perm -0002 -print 2>/dev/null
find / \( -path /proc -o -path /sys \) -prune -o -user 65534 -print 2>/dev/null | head -50

stage limits
ulimit -a

stage cuda-as-65534
home=$(mktemp -d /tmp/probe-home.XXXXXX)
chown 65534:65534 "$home"
chmod 0700 "$home"
setpriv --reuid=65534 --regid=65534 --clear-groups --inh-caps=-all --bounding-set=-all --no-new-privs \
    env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" HOME="$home" TMPDIR="$home" \
    TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONUNBUFFERED=1 \
    python - <<'EOF'
import json, os, time, torch
print("uid", os.getuid(), "groups", os.getgroups(), flush=True)
for path in ["/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]:
    try:
        os.close(os.open(path, os.O_RDWR))
        print("open rw", path, "ok", flush=True)
    except OSError as e:
        print("open rw", path, "failed:", e, flush=True)
print("cuda available", torch.cuda.is_available(), flush=True)
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
t = time.time()
s = (a @ a).float().sum().item()
torch.cuda.synchronize()
print("matmul", s, round(time.time() - t, 3), "s", flush=True)
f = torch.compile(lambda x: torch.nn.functional.gelu(x @ x) + 1.0)
t = time.time()
s = f(a).float().sum().item()
torch.cuda.synchronize()
print("compiled", s, round(time.time() - t, 3), "s", flush=True)
caches = {}
for root, _, files in os.walk(os.environ["HOME"]):
    caches[root] = len(files)
print("PROBE_CACHES " + json.dumps(caches), flush=True)
EOF
echo "cuda-as-65534 exit: $?"

stage left-behind
for d in /proc/[0-9]*; do [ "$(stat -c %u "$d")" = 65534 ] && echo "${d#/proc/} $(cat "$d/comm")"; done
ipcs -a 2>/dev/null || true

stage done
