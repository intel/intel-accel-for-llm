# AGENTS.md

## Build and test environment

- All builds and tests run **inside the dev container**, never on the host (the host `python3` has no `torch`).
- The container name is `CONTAINER_NAME` in [setvars.sh](setvars.sh) (default `iaxl.vllm`; a recipe selected with `IAXL_RECIPE` may change it, see [doc/design/recipes.md](doc/design/recipes.md)). It is started by [start.sh](start.sh), which also runs `pip install -e .` on entry.
- The repository is bind-mounted at the same absolute path inside the container, and `/_data` maps to `$PWD/_data`.

Run commands through the container from the repo root:

```bash
CONTAINER_NAME=$(sed -n 's/^CONTAINER_NAME=\${CONTAINER_NAME:-\(.*\)}$/\1/p' setvars.sh)
docker exec -w "$PWD" "$CONTAINER_NAME" bash -c '<command>'
```

Examples:

```bash
# rebuild the extension after editing iaxl/csrc
docker exec -w "$PWD" iaxl.vllm bash -c 'pip install -e . --no-build-isolation'

# benchmarks
docker exec -w "$PWD" iaxl.vllm bash benchmark/kvstore/kvstore_benchmark.sh --data-source mock
docker exec -w "$PWD" iaxl.vllm bash benchmark/kvstore/kvstore_benchmark.sh --ranks 2 --data-source mock
docker exec -w "$PWD" iaxl.vllm bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh --ranks 2
```

If `docker exec` reports the container is not running, ask the user to start it with `./start.sh`.
