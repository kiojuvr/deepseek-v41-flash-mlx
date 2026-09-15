# logits oracle比較の結果（2026-09-14）

native trace `native-20260914-121105-16915`（`1/sqrt`修正後）と固定oMLX trace `omlx-20260914-112949-15317`を、`tools/reference/compare_traces.py`で比較した。比較レポートは`artifacts/logits-trace/compare-20260914-121200.json`。

## 要約

| 境界 | exact | max_abs | mean_abs |
| --- | --- | --- | --- |
| `encoder.entry.*` | yes | 0 | 0 |
| `encoder.layer0.attn_in` / `attn_out` | yes | 0 | 0 |
| `encoder.layer0.post_attn` | no | 2.38e-07 | 4.4e-12 |
| `encoder.layer0.hidden` | no | 1.91e-06 | 2.7e-11 |
| `encoder.layer19.hidden` | no | 16 | 0.213 |
| `decoder.layer39.hidden` | no | 2048 | 27.4 |
| `final.norm` | no | 0.0586 | 0.0064 |
| `logits` | no | 0.927 | 0.088 |

- `logits` argmax は4tokenとも一致 `[5,6743,223,20]`。
- routing tie は layer36/token0 の1件のみで、native / oMLX とも6位ID=186で一致。
- `token_map_matches_native=True`。

## 解消済み: RMSNormのrsqrt差

修正前の最初の分岐は `encoder.layer0.attn_in`（`RMSNorm(hc_pre(hidden,pre_mix))`）だった。公式式CPU転記（[cut_layer0_attn_in.py](../tools/reference/cut_layer0_attn_in.py)）で **oMLXは0 mismatch、nativeは2要素1 ULP不一致**。MLX再現診断（[diagnose_rmsnorm.py](../tools/reference/diagnose_rmsnorm.py)）で `mx::rsqrt` が原因と特定し、`1/sqrt` に置換。修正後は `attn_in` / `attn_out` がbit一致になった。

## 残る最初の分岐: mHC FP32係数のreduction順

`encoder.layer0.post_attn`（`hc_post(attn_out, residual, attn_mix)`）が次の分岐。`attn_out` と `residual` はbit一致なので、差はmHC係数`attn_mix`にある。既存mHC probeも係数の1 ULP差を記録済み（attn pre 7/8、comb 22/32、max 1.788e-7）。

[diagnose_mhc_mixes.py](../tools/reference/diagnose_mhc_mixes.py)で分解:

| 段 | CPU転記との不一致 |
| --- | --- |
| mean square（20,480要素） | 1 / 4 |
| projection 24×20,480（row-wise） | 72 / 96 |
| projection（batched） | 78 / 96 |
| normalized（1/sqrt / rsqrt いずれも） | 68 / 96 |

**原因はmHC projectionのFP32 reduction順**であり、rsqrtではない。MLXとNumPy / PyTorch CPU転記で20,480要素のdot積の加算順が異なるため、現時点でbit一致を要求していない（再現不能とは断定しない）。`1/sqrt`変更でもprobe結果は不変。

## 含意と契約

[correctness契約](correctness.md)の数値許容差（2026-09-14）で扱いを分けた。

- 公式／外部reference → local reference: mHC FP32係数は暫定的に `max_abs ≤ 1e-6`、非有限値ゼロ。比較対象・dtype・入力・実装revisionを固定する。
- local reference → optimized path: 従来どおりbit一致を必須。
- M2判定: 係数許容差とargmax一致だけでは合格にしない。`logits`差の基準はより広いteacher-forced入力で別途固定する。
- NumPy / PyTorch CPU転記や固定oMLXとの差を「公式CUDAとの差」とは呼ばない。

現在の4 token・`logits` max_abs 0.927 / mean 0.088 では下流への影響を十分に評価できていない。sampling実装はこの契約と並行して進めてよいが、M2合格・production昇格とは分ける。

## 60 token teacher-forced入力（2026-09-14）

固定coding-agent風promptをcheckpoint tokenizerでtokenizeした60 token（`artifacts/logits-trace/prompt-tokens.txt`）で比較した（native `native-20260914-132304-19773`、oMLX `omlx-20260914-132355-19800`、レポート `compare-wide-20260914-132500.json`）。routing tieは両者0件。

| 境界 | exact | max_abs | mean_abs |
| --- | --- | --- | --- |
| `encoder.entry.*` | yes | 0 | 0 |
| `encoder.layer0.attn_in` | yes | 0 | 0 |
| `encoder.layer0.attn_out` | no | 0.0156 | 6.8e-05 |
| `encoder.layer2.attn_in` | no | 0.0742 | 3.8e-04 |
| `encoder.layer19.hidden` | no | 4.31 | 0.129 |
| `decoder.layer39.hidden` | no | 1353 | 1.64 |
| `final.norm` | no | 0.803 | 0.054 |
| `logits` | no | 13.28 | 0.779 |

- **最初の分岐は `encoder.layer0.attn_out`（SWA attention出力）**。`attn_in` は60 token全てbit一致するため、RMSNorm修正は保持されている。4 tokenでは activation量子化が差を吸収していたが、広い入力では現れる。
- `logits` argmax は **57/60 token一致**。不一致は token 1 / 8 / 12（max_abs 13.28 / 6.91 / 3.20）。
- 4 tokenの結果（argmax全一致）をM2基準に使えないことを示す。

次の切り分けは `attn_out` の内部（q/kv projection、RoPE、FP8 activation量子化、masked attention）で、sub-boundaryを追加して特定する。

### layer 0 attention の切り分け（2026-09-14）

sub-boundaryを追加して再比較した（native `native-20260914-145608-21879`、oMLX `omlx-20260914-145657-21905`、レポート `compare-wide-20260914-145700.json`）。

| 境界 | exact | max_abs |
| --- | --- | --- |
| `attn_qr`（q_norm出力） | yes | 0 |
| `attn_qb`（`wq_b`投影、RoPE前） | yes | 0 |
| `attn_kv_norm` | yes | 0 |
| `attn_kv`（RoPE後KV） | yes | 0 |
| `attn_q`（`wq_b`投影＋RoPE） | no | 0.0156（14要素、dim 464–480） |

- `attn_qb` がbit一致するため **`wq_b` 投影とactivation量子化は一致**。`mx.quantized_matmul` の2D per-row と 3D batched も0 mismatch（[diagnose_qmm.py](../tools/reference/diagnose_qmm.py)）。
- 不一致14要素はすべてRoPE tail（dim 448–511、特に464–480）。
- [diagnose_rope.py](../tools/reference/diagnose_rope.py) で native のuncompiled RoPE式を再現すると **nativeと0 mismatch**（batched / per-tokenも一致）。oMLXの`@mx.compile`'d `_rope` を同じ入力へ適用すると **ちょうど14 mismatch**。
- したがって **最初のnative/oMLX分岐は oMLX の compiled RoPE fusion** であり、nativeの式の誤りではない。nativeの `attn_out` 差（0.0156）もこのQ RoPE差に由来する。

この結果は「nativeがoMLXと違う」＝「nativeが誤り」ではないことを示す。公式式CPU転記とoMLXの差は区別して扱う。layer 1以降の差はこの14要素が伝播したものと、他の演算のcross-framework差が混在するため、層ごとに同じ切り分けを続ける。

## 再現

```sh
bash tools/benchmark/run_text_trace.sh
# oMLX traceは既存を再利用可
/System/Volumes/Data/Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3 tools/reference/compare_traces.py \
  --native <native-trace> --oracle <omlx-trace> --output <report.json>
/System/Volumes/Data/Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3 tools/reference/diagnose_mhc_mixes.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash --native <native-trace> --layer 0 --kind attn
```

CPU転記はNumPy/PyTorchであり、公式CUDA kernel実行のoracleではない。routing tieの最小ID方針、index tie停止、Engram layer 1/14の`rsqrt`未変更を引き継ぐ。

## 公式RoPE関数のCPU局所比較（引き継ぎ後）

`tools/reference/trace_cpu_rope.py`で公式model.pyからRoPE関数をAST抽出し、native trace `native-20260914-145608-21879`のRoPE前Q/KVを固定入力としてCPU実行した。公式sourceはM1 hashで照合済み。

| 境界 | bit不一致 | max_abs | 非有限値 |
| --- | ---: | ---: | ---: |
| encoder.layer0.attn_q | 7 | 0.015625 | 0 |
| encoder.layer0.attn_kv | 0 | 0 | 0 |

[比較レポート](../artifacts/logits-trace/compare-cpu-rope-takeover-20260914.json)。CPU traceのmanifestは`artifacts/logits-trace/cpu-rope-takeover-20260914/manifest.json`。full logits / CUDAのoracleではない。oMLX compiled fusionによる14要素差を再現した先の結果とは両立する。nativeの式が公式CPUとbit一致する、という意味には拡張しない。

比較器はsub-boundaryを演算順へ修正したため、新レポートではQ / KV / attention内部をattn_outより前に並べる。数値許容差による最初の分岐と、bit単位の最初の不一致を別に記録する。旧レポートは書き換えない。
