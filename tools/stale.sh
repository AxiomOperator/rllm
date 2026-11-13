#!/bin/bash
# stale.sh: list (default) or stop leftover LLM tasks on this machine.
#
#   bash stale.sh               list what would be stopped (nothing is touched)
#   bash stale.sh --kill        stop it: SIGTERM, then SIGKILL after 5 s
#   bash stale.sh --kill --services
#                               also stop long-running `rllm serve` processes
#
# Leftovers: rllm runs that aren't servers or the fleet agent (bench,
# perplexity, compare-ref, tune, run), rllm test servers on port 8097, GPU test binaries (gpu_tests,
# parity, llama_ref, *_dv builds), test scripts (check.sh and friends), model
# downloads from Hugging Face or a LAN peer and their retry loops, the
# temporary LAN file server (http.server), and the test clients.
# llama-server is stale too: --kill stops the llama-rocm user service (so
# systemd doesn't restart it) and any other llama-server process. Processes
# are matched by exact name, then by their command line: never `pkill -f`,
# which would also match this script.
#
# All machines from a workstation:
#   for h in 31 32 33 34 35; do ssh administrator@172.99.99.$h 'bash -s -- --kill' < tools/stale.sh; done

kill_it=0
services=0
for a in "$@"; do
    case $a in
        --kill) kill_it=1 ;;
        --services) services=1 ;;
        *) echo "usage: $0 [--kill] [--services]" >&2; exit 2 ;;
    esac
done

host=$(hostname)
self=$$
victims=()

cmdline() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null; }
show() { printf '%-8s %-8s %10s  %-8s %s\n' "$host" "$1" "$(ps -o etime= -p "$1" 2>/dev/null | tr -d ' ')" "$2" "${3:0:160}"; }
take() {  # pid kind args
    [ "$1" = "$self" ] || [ "$1" = "$PPID" ] && return
    show "$1" "$2" "$3"
    victims+=("$1")
}

printf '%-8s %-8s %10s  %-8s %s\n' HOST PID ELAPSED KIND COMMAND

# rllm itself: servers on other ports are services, everything else a leftover.
for p in $(pgrep -x rllm); do
    a=$(cmdline "$p")
    if [[ $a == *" agent "* ]]; then
        # The fleet agent (it may be what's running this script).
        show "$p" kept "$a"
    elif [[ $a == *" serve "* && $a != *"--port 8097"* ]]; then
        if [ $services = 1 ]; then take "$p" service "$a"; else show "$p" kept "$a"; fi
    else
        take "$p" rllm "$a"
    fi
done

# Test and dev binaries (exact process names; comm is cut to 15 characters).
for name in rllm-dv rllm-graphs gpu_tests gpu_tests_dv gpu_tests_graph parity parity_dv llama_ref; do
    for p in $(pgrep -x "$name"); do take "$p" test "$(cmdline "$p")"; done
done

# Downloads: curl fetching a model, and the bash loops that restart it.
for p in $(pgrep -x curl); do
    a=$(cmdline "$p")
    [[ $a == *huggingface.co* || $a == *:8099/* ]] && take "$p" download "$a"
done
for p in $(pgrep -x bash); do
    a=$(cmdline "$p")
    if [[ ($a == *"until curl"* && ($a == *huggingface.co* || $a == *:8099* )) || $a == *pdl.sh* ]]; then
        take "$p" download "$a"
    elif [[ $a =~ (check|longcheck|cyber_check|mtp_ab[a-z0-9_]*|graph_ab|draft_ab|plim)\.sh ]]; then
        # Test scripts: stopped too, or they would start their next step.
        take "$p" script "$a"
    fi
done

# Python: the temporary file server and test clients.
for p in $(pgrep -x python3); do
    a=$(cmdline "$p")
    if [[ $a == *http.server* ]]; then
        take "$p" fileserv "$a"
    elif [[ $a == *vsweep.py* || $a == *mtp_check* || $a == *draft_ab.py* ]]; then
        take "$p" client "$a"
    fi
done

# llama-server: never left running. Its user service first, or systemd
# restarts it (Restart=on-failure).
if systemctl --user is-active --quiet llama-rocm.service 2>/dev/null; then
    echo "$host: llama-rocm.service is running"
    if [ $kill_it = 1 ]; then
        systemctl --user stop llama-rocm.service && systemctl --user disable llama-rocm.service 2>/dev/null
        echo "$host: llama-rocm.service stopped and disabled"
    fi
fi
for p in $(pgrep -x llama-server); do take "$p" llama "$(cmdline "$p")"; done

if [ ${#victims[@]} -eq 0 ]; then
    echo "$host: nothing stale"
elif [ $kill_it = 1 ]; then
    kill "${victims[@]}" 2>/dev/null
    for _ in 1 2 3 4 5; do
        alive=()
        for p in "${victims[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
        [ ${#alive[@]} -eq 0 ] && break
        sleep 1
    done
    [ ${#alive[@]} -gt 0 ] && kill -9 "${alive[@]}" 2>/dev/null
    echo "$host: stopped ${#victims[@]} process(es)"
else
    echo "$host: ${#victims[@]} stale process(es); run with --kill to stop them"
fi

# GPU memory left (integrated GPUs share system RAM; GTT is what they use).
for d in /sys/class/drm/card[0-9]/device; do
    [ -r "$d/mem_info_gtt_used" ] || continue
    echo "$host: GPU memory $(( $(cat "$d/mem_info_gtt_used") / 1073741824 )) of $(( $(cat "$d/mem_info_gtt_total") / 1073741824 )) GiB in use"
done
