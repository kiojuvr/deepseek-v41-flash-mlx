# Layer 0 MoE replay

`dsv41-moe-trace` replays layer 0 MoE from the frozen native
`encoder.layer0.ffn_in` and saves `moe_out`. It uses the native MLX reference,
including on-demand routed expert loading and the shared expert. The output is
then compared with the existing native backbone `moe_out` using the boundary
comparator. This checks lifecycle and determinism of the expert path; it is not
an independent CPU oracle or full-model qualification.

The replay can load many FP4 expert slabs and may take several minutes. Run it
manually; do not leave a monitor process running:

```sh
bash tools/benchmark/run_moe_replay.sh
```

The optional first argument selects another native trace directory. Expected
resource use is GPU/Unified Memory for the on-demand expert cache and temporary
activation buffers; peak use depends on the unique experts selected by the
input. The script does not modify checkpoint files. Logs and results are written
to `artifacts/cpu-attention/moe-replay-<timestamp>-<pid>/`:
`identity.txt`, `test.log`, `trace-exit-code.txt`, `exit-code.txt`, `trace/`,
`comparison.log`, and `report.json`.

Exit zero means the replay and report completed. Review the report's boundary
bit differences, finite values, route tie count and identity hashes before
calling it a pass. A failure leaves logs and can be retried with the same
command, which creates a fresh directory. There is no token-level resume;
partial trace directories are not reused.

The comparator should show exact equality for a deterministic native replay.
If it does not, inspect the first differing boundary and expert cache lifecycle
before changing arithmetic or applying gate tolerances. Gate ID/score tolerance
does not excuse an expert output or MoE sum mismatch.

## Reviewed replay (2026-09-14)

Run `moe-replay-20260914-164602-24859` completed with exit code zero. The
single common boundary `encoder.layer0.moe_out` matched bit-for-bit for all 60
tokens (60 × 5120 BF16 elements); maximum and mean absolute differences were
zero, with no nonfinite values. Route tie counts were zero in both traces. The
reviewed summary is `artifacts/cpu-attention/moe-replay-20260914-reviewed.json`.

This proves deterministic native MLX replay and on-demand expert lifecycle for
this input. It is not a CPU oracle, official CUDA match, expert semantic
qualification, or full-model qualification. An independent expert-formula path
or second backend remains required before MoE numerical acceptance.
