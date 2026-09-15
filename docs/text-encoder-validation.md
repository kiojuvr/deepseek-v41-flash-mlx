# Encoder 0..19 text referenceの検証

2026-09-13。`TextEncoderReference`を追加し、text入口から**encoder全20層**（layer 0..19）を接続した。producer（global KV所有）をlayer 2 / 8 / 14へ一般化し、reuse consumerをlayer 3–7 / 9–13 / 15–19へ拡張した。Engramはlayer 1とlayer 14で公式順序どおりBlock直前に入る。ユーザーrun `run-20260913-231203-36345`のログ、終了code 0、binary / M1 summary / Engram metadata / fixture provenanceの4件のSHA256を確認した。すべてレビュー時の現物と一致し、metadataはfixture-provenanceと一致した。以下のlocal整合性検査は通過した。[レビュー記録](../artifacts/text-encoder/reviewed-result.json)にログとidentityを保存した。

変更なしでの再実行は不要。以下は再現用手順。公式oracle比較、memory peak・経過時間・throughputは未測定であり、このPASSから推定しない。

## 実装境界

- `CompressorReference`はlayerと`compress_ratio`を受け取る。ratio>1はFP32 gate付きsoftmax pooling、ratio 1（decoder用）はcheckpoint BF16のままのplain projection + norm。
- `GlobalKVProducerReference`はlayer所有のcompressorとindex keyを持ち、`state.rows()==start/ratio`を検査する。
- `SharedAttentionReference`はsource layerとratioを保持し、consumerが自分のsource groupに属するかを`indices()`で検証する。
- `CompressedLayerReference` / `CompressedBlockReference`は任意のkv_source layer（2/8/14/20）を構築できる。
- `ReusedLayerReference` / `ReusedBlockReference`は任意の非source layer（3..39）を構築でき、sourceはconfigの`kv_source_layers`から決まる。
- `layer_owner.hpp`が`kv_source_layers=[2,8,14,20]`、`index_source_layers=[2,8,14,20,24,28,32,36]`、`compress_ratios`（0–1は0、2–19は2、20–39は1）を固定configとして符号化する。

## 範囲と資源

固定token IDs `[0,42,1000]`と継続token `42`で、chunk/tokenwise hidden・pre-mix・全encoder stateのbit一致、継続、fork/reset、不正token（image・語彙外）拒否を検査する。layer 8/14のproducer切り替えと、Engram layer 14のhash slice（24–47行）を含む。

encoder 18 Block分のrouted expertを扱う。expertは初回使用時にloadしてcacheするon-demand経路（全384 expert resident loadと数値同一）にしたため、実行したtokenで選択されたexpertだけが常駐する。attention・MLX staging・allocatorを含む実peakは未測定。Engramはmmapで必要rowのみ読み、checkpointはread-only。今回のユーザーrunは確認済みであり、変更なしで再実行する必要はない（このrunはeager load時点のbinaryで、現在のon-demand経路でも数値は同一）。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_text_encoder_reference.sh
```

## ログと再試行

`artifacts/text-encoder/run-日時-PID/`へ`test.log`、`exit-code.txt`、binary / M1 summary / Engram metadata / fixture provenanceのSHA256を含む`identity.txt`を保存する。検査またはログ保存の失敗は非ゼロ終了。

失敗・中断時は同じコマンドで新しいrunとして最初から再試行する。途中stateは保存しない。終了codeがないrunは未完了。ログとidentityのレビュー前に通過とは判定しない。

## 未接続・引き継ぐ差

- decoder layers 20..39、final collapse / norm / head、logits、samplingは未接続。
- decoderは`candidate_source_layer=20`によりcandidate block選択（`candidate_topk_blocks=2048`、`candidate_block_size=8`）を使う。この二段Top-Kはまだ実装していない。layer 24/28/32/36はindex sourceだがkv sourceではないため、shared index Kを読む点も未実装。
- 公式FP4 main compressed KVはE4M3 block-16 referenceのまま（既知差）。FP4 / index / attentionの独立oracle比較は未完了。
- 既知mHC / RMSNorm / Engram差を引き継ぐ。8層octetの過去PASSをencoder全体の証拠へ転用しない。
