# Backbone trace / oracle比較 harness

2026-09-14。[logits oracle比較計画](logits-oracle-plan.md)に基づき、nativeの同一token列traceを保存するexporterと、任意の2 traceを境界単位で比較するharnessを追加した。**oracle traceはまだ生成していない。** 公式kernel実行 / 公式式CPU転記 / 固定oMLXの区別は計画文書のまま維持する。

## Native trace exporter

`dsv41-text-trace`が固定token列 `[0,42,1000,42]`（plain text、teacher forcing）をfull backboneへ通し、各境界を`<dir>/<name>.npy`（MLX `.npy`、bfloat16は生2byte）へ保存し、`manifest.json`にdtype / shape / `logits_argmax` / routing tie記録を書く。

保存する境界:

- `encoder.entry.hidden` / `encoder.entry.pre_mix`
- `encoder.layer0..19.hidden` / `.pre_mix`
- `encoder.out.hidden` / `.pre_mix`
- `decoder.layer20..39.hidden` / `.pre_mix`
- `decoder.out.hidden` / `.pre_mix`
- `final.collapsed` / `final.norm` / `logits`

`manifest.json`の`route_ties`はtop-6境界の6位 / 7位expert IDとscoreを記録する（最小ID breakの未qualified差）。KV bytes / scalesの境界保存は、最初の不一致layerを特定した後に追加する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_text_trace.sh
```

full backboneのため数分以上かかる。`artifacts/logits-trace/native-日時-PID/`へtrace、`trace.log`、`exit-code.txt`、identity SHA256を保存する。終了codeがないrunは未完了。

## 比較harness

`tools/reference/compare_traces.py`が2つのtrace directoryを境界単位で比較し、shape / logical dtype / bit一致 / 最大・平均絶対差 / 非有限値 / argmax一致と、**最初に分岐した境界**を`report.json`へ保存する。bfloat16はMLXの生2byte storageをmanifestのlogical dtypeでfloat32へ復元して差分を取る。

```sh
/System/Volumes/Data/Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3 \
  tools/reference/compare_traces.py --native <native-trace> --oracle <oracle-trace> \
  --output <report.json> [--tolerance 0]
```

## Oracle traceの作り方

計画の3分類を区別し、それぞれ別の`report.json`へ保存する。

### 固定oMLX対照（実装済み）

`tools/reference/trace_omlx.py`がpinned oMLX v0.7.0.dev2を**source変更なし**でhookする。`load(..., engram_ssd_offload=True, preserve_mtp=False)`で読み込み、`Block.__call__` / `Gate.__call__` / `hc_pre` / `project_logits`をmodule名前空間で差し替える。token列はnativeと同じく**1 tokenずつ**cacheへ投入し、token-serial scheduleを揃える。保存schemaはnativeと同じ。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_omlx_trace.sh
```

`artifacts/logits-trace/omlx-日時-PID/`へtrace、`trace.log`、`exit-code.txt`、identityを保存する。routing tieはlayer・token位置と、oMLXが選んだtop-6 ID（`olmx_ids`）を記録する。`manifest.json`の`token_map_matches_native`でEngram token mapの一致を確認する。これはApple Silicon実動referenceであり、公式kernel実行のoracleではない。

### 公式式CPU転記

`inference/model.py`の式・cast境界をPyTorch CPUで転記し、同じtoken列の境界を保存する。763Bの全展開は非現実的なため、まずentry / 1 layer / gate routingなど小さい境界から始め、`logits`へ広げる。CUDA reductionやtieのoracleとは呼ばない。

### 公式kernel実行

対応GPU上でのみ。現環境では未確認。

oracle traceも同じ`manifest.json` schema（`arrays`にname / logical dtype / shape）に従えば、このharnessをそのまま使える。

## より広いteacher-forced入力

`tools/reference/export_prompt_tokens.py`が固定のcoding-agent風promptをcheckpoint tokenizerでtokenizeし、`dsv41-text-trace` / `trace_omlx.py`共通の`--tokens-file`形式で保存する。manifestにtoken数・tokenizer SHA256を記録する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
/System/Volumes/Data/Users/kioju/.venvs/omlx-0.7.0.dev2/bin/python3 tools/reference/export_prompt_tokens.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --output artifacts/logits-trace/prompt-tokens.txt --max-tokens 128
TOKENS_FILE=artifacts/logits-trace/prompt-tokens.txt bash tools/benchmark/run_text_trace.sh
TOKENS_FILE=artifacts/logits-trace/prompt-tokens.txt bash tools/benchmark/run_omlx_trace.sh
```

比較レポートの`logits_summary`がtoken数、argmax一致token数、positionごとの`max_abs` / `mean_abs`を保存する。M2のlogits差基準はこの広い入力の結果を見て[correctness契約](correctness.md)で別途固定する。4 tokenの結果を基準にしない。

## 比較レポート

`report.json`には`first_diverging_boundary`（実行順）、境界ごとのbit不一致数・最大/平均絶対差・非有限値・argmax一致、`route_tie_comparison`（layer・tokenごとのnative / oracleの6位IDと一致）、`logits_summary`（positionごとのlogits差とargmax一致）を保存する。

## CPU layer 0 Attention trace（2026-09-14、実checkpoint実行待ち）

`trace_cpu_attention.py`と`cpu_reference.py`はnativeの`encoder.layer0.attn_in`を固定入力にし、その後のQ/KV projection、RMSNorm、RoPE、FP8 SWA KV、sink付きsparse attention、逆RoPE、output projectionをCPUで計算する。10境界を既存schemaで保存する。RoPEはhash検証済み公式ソースから関数を抽出し、他は公式式のCPU転記。FP32 reductionはCPU上の順序であり、公式CUDA実行やfull logits oracleではない。

```sh
bash tools/benchmark/run_cpu_attention_trace.sh
```

既定入力は`artifacts/logits-trace/native-20260914-145608-21879`（60 token）。別のnative trace directoryは第1引数で指定する。位置0からの1〜128 tokenに限定する。Pythonは`/Volumes/SDXC-512/deltafin/.venv/bin/python`のPyTorch CPU環境を使用する。

- CPU 4 threads、RAMは数GiBを見込む（実測前）。GPU、MoE、Engramのロードは行わず、checkpointは読み取り専用。実行は数分以上になる可能性がある。
- `artifacts/cpu-attention/run-日時-PID/`に`identity.txt`、`test.log`、`trace-exit-code.txt`、全体の`exit-code.txt`、`trace/manifest.json`と境界配列、`comparison.log`、`report.json`を保存する。
- 非有限値・source/header identity不一致・計算/比較エラーは非0終了。終了0はtrace生成と比較レポート作成の完了であり、数値一致のPASSではない。レポートとidentityのレビューが必要。
- 中断時のtoken途中再開は未実装。同じコマンドで新しいrun directoryへ再実行し、失敗ログを保持する。trace生成済みなら比較のみ再実行できる：`/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/compare_traces.py --native <入力directory> --oracle <run>/trace --output <run>/report.json`。

小さい算術検査はcheckpointをロードせずに行う。実checkpointの比較結果はまだない。

## 初回比較時点の未完了事項（後続結果はresults文書参照）

- `1/sqrt`変更後のnative trace再取得と再比較（効果未確認）。
- 分岐が残る場合、その境界のrouting IDs / score / weightとKV bytes / scalesを追加保存する。
- `logits`のargmax一致はexactness通過を意味しない。bit不一致数・最大差・非有限値を併記して判定する。
- Engram layer 1/14 の `mx::rsqrt` は未変更。既知のmHC / Engram差、routing tieの最小ID方針、index tie停止を引き継ぐ。

## 最初の比較結果（2026-09-14）

固定token列 `[0,42,1000,42]` で native と固定oMLXを比較した（native `native-20260914-114504-15930`、oMLX `omlx-20260914-112949-15317`）。

- `encoder.entry.*` はbit一致。`logits` argmax は4tokenとも一致 `[5,6743,223,20]`。
- routing tie は layer36/token0 の1件のみで、native / oMLX とも6位ID=186で一致。
- 最初の分岐は **`encoder.layer0.attn_in`**（= `RMSNorm(hc_pre(hidden,pre_mix))`）。`attn_out` / `moe_out` は activation量子化で差が消えbit一致。
- 公式式CPU転記（`tools/reference/cut_layer0_attn_in.py`）で切り分け: **oMLX はCPU転記と0 mismatch、native は2要素（1 ULP, max 1.22e-4）不一致**。
- MLX再現診断（`tools/reference/diagnose_rmsnorm.py`）: collapse は一致。native の `mx::rsqrt` が原因で、`1/sqrt` に置換すると **CPU転記と0 mismatch**。mean精度は無関係。

対応として `rms_norm_reference` と `hc_mixes` の正規化を `rsqrt` から `1/sqrt` へ変更した。**この変更後のnative trace再取得と再比較が未実施**であり、効果は未確認。
