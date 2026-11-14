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

stage limits
ulimit -a

# The wrapper's invariant scan (wrapper/src/scan.rs), as `find` sees it: outside /proc, /sys and the
# sandbox uids' places, what each uid can write (access(2)) or owns, as a directory, file or fifo. An
# owned entry on a read-only mount is left out: the scan will count it as unwritable.
scan_view() {
    ro=$(awk '$6 ~ /(^|,)ro(,|$)/ { printf "%s ", $5 }' /proc/self/mountinfo)
    for uid in 65534 65533; do
        echo "--- writable by $uid"
        setpriv --reuid=$uid --regid=$uid --clear-groups --inh-caps=-all --bounding-set=-all --no-new-privs \
            find / \( -path /proc -o -path /sys \) -prune -o \( -type d -o -type f -o -type p \) -writable -print 2>/dev/null
        echo "--- owned by $uid, on a mount that isn't read-only"
        find / \( -path /proc -o -path /sys \) -prune -o \( -type d -o -type f -o -type p \) -user $uid -print 2>/dev/null |
            awk -v ro="$ro" 'BEGIN { n = split(ro, p, " ") } { for (i = 1; i <= n; i++) if (p[i] != "" && ($0 == p[i] || index($0, p[i] "/") == 1)) next; print }'
    done | grep -vE '^/(tmp|var/tmp|dev/shm|dev/mqueue|run/lock|var/log/journal|work)(/|$)'
}

stage scan-before
scan_view

# The fixes as the wrapper makes them: /dev, /dev/log and /var/log as before (plan 10d), then each
# read-write mount of the GPU runtime remounted read-only with the syscall and flags the wrapper uses,
# and /usr/local/nvidia, which chown is refused (the first probe), bound onto itself with what is
# mounted below it, and that bind made read-only.
stage driver-dir
lsattr -d /usr/local/nvidia 2>&1 || true
stat -c '%n %U(%u) %a' /usr/local/nvidia
touch /usr/local/nvidia/probe-file 2>&1 && { echo "root can create in /usr/local/nvidia"; rm -f /usr/local/nvidia/probe-file; }

stage fix
chmod 0755 /dev && echo "/dev 0755"
rm -f /dev/log && echo "/dev/log removed"
mount -t tmpfs -o mode=0755,uid=0,gid=0,size=1m,nosuid,nodev,noexec none /var/log && echo "/var/log covered"
python - <<'EOF'
import ctypes, os
FLAGS = {"nosuid": 2, "nodev": 4, "noexec": 8, "noatime": 1024, "nodiratime": 2048, "relatime": 1 << 21}
MS_RDONLY, MS_REMOUNT, MS_BIND, MS_REC = 1, 32, 4096, 16384
DRIVER = "/usr/local/nvidia"
libc = ctypes.CDLL(None, use_errno=True)

def mounts():
    return [(f[4], f[5].split(",")) for f in map(str.split, open("/proc/self/mountinfo"))]

def kept(options):
    return sum(v for k, v in FLAGS.items() if k in options)

def call(label, source, point, flags):
    r = libc.mount(source, point.encode(), None, flags, None)
    print(label, point, "->", "ok" if r == 0 else os.strerror(ctypes.get_errno()), flush=True)

for point, options in mounts():
    if point == "/dev/nvidia-caps" or point == DRIVER or point.startswith(DRIVER + "/"):
        if "rw" in options:
            call("remount read-only:", None, point, MS_REMOUNT | MS_BIND | MS_RDONLY | kept(options))
        else:
            print("already read-only:", point, options, flush=True)
if os.path.isdir(DRIVER) and not any(point == DRIVER for point, _ in mounts()):
    on = max((m for m in mounts() if DRIVER == m[0] or DRIVER.startswith(m[0].rstrip("/") + "/")), key=lambda m: len(m[0]))
    print("the driver directory is on", on, flush=True)
    call("bind onto itself:", DRIVER.encode(), DRIVER, MS_BIND | MS_REC)
    call("remount read-only:", None, DRIVER, MS_REMOUNT | MS_BIND | MS_RDONLY | kept(on[1]))
EOF
ls /usr/local/nvidia/lib64 | wc -l

stage scan-after
scan_view

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
