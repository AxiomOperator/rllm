# Fleet agent and dashboard (`rllm agent`)

`rllm agent` runs on every machine of a fleet. It manages that machine's
rllm servers, models, binaries, power limits and tests, and serves a
dashboard that controls every machine in its peer list. It is part of the
`rllm` binary, so a deploy updates it too.

## Setup

On every machine (same token everywhere; list every machine, this one
included):

```sh
rllm agent init --token <shared secret, 16+ chars> \
  --peer agx01=172.99.99.31:8090 --peer agx02=172.99.99.32:8090 \
  --peer agx03=172.99.99.33:8090 --peer agx04=172.99.99.34:8090 \
  --peer agx-05=172.99.99.35:8090 --serve-api-key <key for started servers>
rllm agent install        # systemd user service rllm-agent.service, started now
sudo loginctl enable-linger $USER   # once: keep it running without a login, start at boot
```

`init` writes `~/.config/rllm/agent.json` (mode 600); `rllm agent run`
runs the agent in the foreground. Open `http://<any machine>:8090/` and
enter the token (or open `http://<machine>:8090/#token=<token>` once; the
fragment never reaches the server and is removed from the address bar).

Config fields: `token`, `listen` (default `0.0.0.0:8090`), `name` (default:
hostname), `peers`, `bin_dir` (default `~/rllm-bin`: `rllm`, `parity`,
`gpu_tests`, and `wiki.test.raw` for perplexity), `model_dirs` (default
`<bin_dir>/models`; fetches go to the first), `serve_api_key`, `env`
(variables for started servers and tests, e.g. `LD_LIBRARY_PATH`),
`state_dir` (default `~/.local/share/rllm-agent`), and `schedules`:

```json
"schedules": [{"suite": "parity", "every_hours": 24},
              {"suite": "mtp", "model": "Qwen3.6-35B-A3B-Q4_K_M.gguf", "every_hours": 168}]
```

## What it does

- **Machines**: GPU memory, power, clock, temperature and busy (AMD APUs
  from the SMU metrics table, NVIDIA from nvidia-smi), RAM, disk, the
  deployed build, and each rllm server with health, generated tokens/s,
  requests and MTP draft acceptance.
- **Servers** run as transient systemd user units `rllm-serve-<port>`
  with restart off: they outlive the session that started them, a crash
  leaves them down (shown as failed), and stopping one stops exactly that
  unit. Before starting, the agent sizes the server (weights + KV cache +
  recurrent state + runtime against the memory a new server may use: on an
  integrated GPU the smaller of free GPU memory less 4 GiB and available
  RAM less 8 GiB) and refuses if it won't fit. Servers started by hand are
  listed and can be stopped too (by pid). Recurrent models get a KV pool
  of `ctx × parallel` tokens (they run without a prefix cache, so more is
  never used).
- **llama-server is never left running**: any llama-server process or an
  active `llama-rocm.service` shows as a red banner with a button that
  stops and disables the service and kills the processes; the agent
  refuses to start an rllm server next to one.
- **Models**: which file is on which machine. A copy goes over the LAN
  from a machine that has it (two ranged streams, checked against the
  source's SHA-256); a Hugging Face URL is downloaded once in 8 parallel
  ranges and checked against Hugging Face's LFS SHA-256. Each range
  attempt goes to its own scratch file and is kept only on an HTTP 206 for
  exactly the requested offset; nothing replaces the file until the hash
  matches.
- **Deploys** are rolling: one machine at a time pulls `rllm`, `parity`
  and `gpu_tests` from the source machine, checks their SHA-256, runs the
  new `parity` on its own GPU, and installs only if it passes (keeping the
  previous binaries for rollback); the first failure stops the rollout.
  The agent restarts onto the new binary; running servers keep theirs
  until restarted.
- **Power**: measured power and clocks, the boot limits from
  `ryzenadj.service`, and balanced (85/85/100 W) or performance
  (120/120/140 W) profiles through `sudo -n ryzenadj` (allow it with the
  sudoers line the dashboard shows).
- **Tests**: `parity` (GPU kernels against the CPU), `model` (the GPU
  model tests), `bench` (pp512/tg128), `perplexity` (wikitext-2, 32
  chunks), `mtp` (greedy output with and without MTP must match, on a
  temporary server), `full`; on one machine or many at once, and on a
  schedule. Results go to a history file that survives restarts.
- **Leftover processes**: the `tools/stale.sh` rules (test binaries,
  stuck downloads, test servers on port 8097, llama-server), with the
  agent and rllm servers on other ports kept.

## Security

Every `/api` and `/fleet` call needs `Authorization: Bearer <token>`
(compared in constant time); the dashboard page itself and `/healthz` are
open. The agent speaks plain HTTP and can start processes and replace
binaries, so run it on a trusted LAN only. It runs fixed actions (never
arbitrary commands); file names are checked to stay inside the model and
binary directories.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/status` | host, GPU, servers, llama-server, recent jobs, build |
| GET | `/api/models` | model files with size, architecture, GPU memory |
| GET | `/api/models/{name}/sha256` | hashes the file (cached) |
| POST | `/api/models/fetch` | `{name, peer}` or `{name, url}`; a job |
| POST | `/api/models/delete` | `{name}` (refused while served) |
| GET | `/api/files/models/{name}`, `/api/files/bin/{name}` | with HTTP ranges |
| GET | `/api/servers` | |
| POST | `/api/servers/plan`, `/api/servers/start` | `{model, port, ctx, parallel, kv_type, kv_tokens?, mtp, host, force}` |
| POST | `/api/servers/stop` | `{port}` |
| GET | `/api/servers/{port}/log` | the unit's journal |
| POST | `/api/llama/stop` | stop and disable llama-server |
| GET/POST | `/api/stale`, `/api/stale/kill` | `{services}` |
| GET/POST | `/api/power`, `/api/power/apply` | `{profile}` |
| GET | `/api/bin/manifest` | |
| POST | `/api/deploy`, `/api/deploy/rollback` | `{source}` |
| POST | `/api/tests/run` | `{suite, model}` |
| GET | `/api/tests/history`, `/api/jobs`, `/api/jobs/{id}` | |
| GET | `/api/fleet` | peers |
| POST | `/api/fleet/deploy`, `/api/fleet/distribute`, `/api/fleet/tests` | across machines |
| any | `/fleet/{name}/{path}` | forwarded to that machine's agent |
