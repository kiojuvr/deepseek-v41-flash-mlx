# Native text entryとRMSNorm reference

2026-09-13。`TextEntryReference`と`rms_norm_reference`を追加した。C++ / MLXのみで実行し、Pythonはfixture生成に限定する。**モデル全体のforwardは未接続、RMSNormの公式CPUとの差も未qualified。**

## 実装境界

- `embed.weight`の公式BF16 `[129280,5120]`をresident arrayへ読み、batch one / 1–128 tokensをgatherする。再量子化しない。
- 公式`Transformer.forward`に従い、embeddingを4 copiesへ展開し、FP32 pre-mix `[1,0,0,0]`を初期化する。hiddenは`[tokens,4,5120]`、pre-mixは`[tokens,4]`でbatch軸を省略する。
- copiesはbroadcast viewで共有する。返したarrayをimmutableな入力として使い、書き込みを伴うconsumerを接続する場合はaliasを考慮する。
- token範囲外、empty、129 tokens以上、未実装の画像入力を要求するtoken `129264`を拒否する。tokenizer / protocol / image処理は未接続。
- RMSNormは公式`RMSNorm.forward`と同じFP32 mean-square → rsqrt → input乗算 → weight乗算 → BF16 cast。融合kernelへ置換しない。BF16入力・weight、shape、epsilonを検査する。

RMSNormは独立したprimitiveとして検査している。モデル入口へRMSNormを挿入する変更ではない。実際の適用箇所はBlockのattention / FFN前と最終collapse後であり、今後の接続で公式順序を維持する。

embedding payloadは1,323,827,200 bytes（約1.233 GiB）。load時はCPU読み取りbufferとMLX所有copyが一時的に共存する。これはstatic payloadで、実runtime peakではない。probeは全backboneやEngram backingをresident化しない。

## 短い検査の結果

[結果](../artifacts/model-entry/native-verify.json)と[fixture identity](../artifacts/model-entry/manifest.json)を保存した。

| 検査 | 結果 |
| --- | --- |
| 実checkpointの7 token rows、先頭・末尾・重複を含む | BF16 bits一致 |
| 4 copiesと初期pre-mix | 全一致 |
| 一括 / tokenwise、model object破棄後のgraph評価 | 一致 |
| 不正token・画像token・入力長、RMSNorm dtype / shape / epsilon | 拒否 |
| 公式AST抽出RMSNorm CPU出力との比較 | 51,200要素中5要素でBF16 1 ULP差 |

RMSNorm入力は実embedding 7行とzero / `2**-60` / `2**60`の3行、weightは実`layers.0.attn_norm.weight`。最初の不一致はflat index 22270、native `0xbcf8` / CPU `0xbcf7`。既存Engramのcross-backend差と同様に、許容値を追加して合格へ変更しない。probe終了codeは2で保存し、model全体やCUDAとの一致は主張しない。

## 再現

workspaceのMLX 0.32.2 buildを使用する。fixture生成は小さな選択行だけを読み、native probeはembedding約1.32 GBをloadする。今回のnative probeは約0.44秒で終了した。load cache状態で所要時間は変わり、性能baselineには使わない。

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/export_model_entry_fixture.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --m1-summary artifacts/checkpoint/summary.json \
  --verification artifacts/checkpoint/verification.json --output artifacts/model-entry
cmake --build build-mlx -j 8
build-mlx/dsv41-model-entry-probe \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --m1-summary artifacts/checkpoint/summary.json --fixtures artifacts/model-entry \
  --output artifacts/model-entry/native-verify.json
```

終了code 0はこのfixtureの全比較bits一致、2は数値差、1は入力 / I/O / contract違反。結果を読まずに合格としない。[実行provenance](../artifacts/model-entry/run-provenance.json)を参照。

次は公式activation quantizationを伴う共通FP8 / FP4 linear、mHCのpre / post mixとBlockを接続する。入口だけをtoken→logits referenceとして扱わない。Engram gateとRMSNormのcross-backend差を未解決事項として保持し、M2終了前に公式oracleからの判定根拠を固定する。
