# Remote (NIXL/RDMA) KV cache daemon -- operator scripts

Full design: [`doc/design/remote-nixl-kv-cache.md`](../../doc/design/remote-nixl-kv-cache.md)
(中文: [`remote-nixl-kv-cache.zh-CN.md`](../../doc/design/remote-nixl-kv-cache.zh-CN.md)).

Full usage walkthrough: [`doc/usage/remote-nixl-kv-cache.md`](../../doc/usage/remote-nixl-kv-cache.md)
(中文: [`remote-nixl-kv-cache.zh-CN.md`](../../doc/usage/remote-nixl-kv-cache.zh-CN.md)).

There is no separate *build* for the remote cache daemon (no CPU-only wheel
to cross-compile): it reuses the exact same `iaxl` package already compiled
on the GPU node (same `pip install -e .`, same `vllm/vllm-openai:v0.23.0`
-based image built by `start.sh`). Scripts here:

- `run-daemon.sh` -- one daemon process, env-var configured.
- `run-daemon-multi.sh` -- N daemon processes on one host, one per GPU rank
  (avoids many-to-one control-plane/codec contention when one process cannot
  keep up with `tp_size` GPU workers).
- `build_release.sh` -- packages a standalone, relocatable
  `dist/iaxl-remote-daemon-<version>.tar.gz` (compiled `iaxl` package +
  the 3 QAT/QPL runtime libraries it builds + these scripts) that can be
  copied to a remote storage node and run there without any build toolchain.
- `docker-run-daemon.sh` -- runs an extracted release bundle inside a
  container started from the same base image as the GPU node.

See the usage doc's "release package" section for the full copy-and-run
walkthrough.
