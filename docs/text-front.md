# Token IDからlayer 0への接続

`TextFrontReference`は既存TextEntryReferenceとBlockReferenceを接続する。token IDをBF16 embeddingへ変換し、4 copiesへ展開、各tokenにFP32 `[1,0,0,0]` pre-mixを作ってlayer 0へ渡す。前tokenのBlock出力pre-mixを次tokenの入力へ回さない。Blockの出力pre-mixは、同じtokenの次layerが消費する値である。

request stateはlayer 0のSWA state。入力position / countを先に検査し、image tokenと語彙外IDはembedding前に拒否する。Block出力評価が完了するまでstateをcommitしない。tokenizer、prompt protocol、logits、sampling、layer 1以降はまだ含まない。

公式`inference/model.py:Transformer.forward`ではEngramを持つlayerへ入る直前にhiddenを更新し、その後Blockを呼ぶ。layer 1へ展開する際もこの境界を使う。EngramをattentionとFFNの間へ移動しない。

## 検証状況とユーザー実行

### Layer 1への接続追記

`SwaProjectionReference`、`SwaLayerReference`、Gate / Expert / MoE、Blockのweight ownerをlayer引数（既定0）で選択できるようにした。現実装はpure SWAのlayer 0・1だけを受理し、圧縮attentionが必要なlayer 2以降を拒否する。

2026-09-13の短いGPU検査で、layer 1の実Q/KV weightを使う位置127→128のchunk/tokenwise bit一致、実Gateが選んだFP4 expertの有限値・繰り返しbit一致を確認した。既存layer 0の短い検査も通過した。これはlayer 1全体のoracle検証ではない。

`TextPairReference`を追加し、token ID → layer 0 → Engram layer 1 → Block layer 1を接続した。hashは各tokenにつき一度更新し、token-major / layer-majorの先頭24行をEngram 1へ渡す。layer 0が返したpre-mixはEngram適用で変更せず、layer 1へ渡す。Engram hashと2層のKV stateは作業用copyで更新し、全出力の評価成功後にまとめてcommitする。metadataはruntimeと同一のshared ownerをstateへ渡す契約。

2層経路は[ユーザー実行の検証](text-pair-validation.md)をレビュー済み。chunk/tokenwise bits、継続、fork/reset、hash継続一致、不正token / 不整合position拒否が通過した。Engramの既知oracle差は引き継ぐ。入口での拒否時state保持を検査し、処理途中の障害注入は含まない。既存text-frontのレビュー結果は一般化前のbinaryに対する履歴として保持し、新binaryへ自動で引き継がない。

2026-09-13、ユーザーrun `run-20260913-144004-24882`のログ、終了code 0、binary / M1 summaryのSHA256を確認した。実token IDからのchunk/tokenwise hidden・pre-mix・KV state bit一致、継続、fork、reset、image / 語彙外token拒否は通過。記録されたSHA256はレビュー時の現物と一致した。[レビュー結果](../artifacts/text-front/reviewed-result.json)にログ本文とidentityを保存した。合成hiddenを使うlayer 0 Block検査とは別の証拠として保持する。

変更なしでの再実行は不要。以下は再現用の手順。今回の検査は公式oracle比較、生成品質、全layerの検証ではなく、memory peakや性能も測定していない。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_text_front_reference.sh
```

固定の合法token ID `[0,42,1000]`からembeddingとlayer 0を実行し、chunk/tokenwiseのhidden・pre-mix・KV stateをbit比較する。4 token目への継続、fork分離、reset、image / 語彙外token拒否時のstate保持も確認する。chat promptや生成品質の検証ではない。

layer 0全expertの約6.76 GiBに加え、embedding 1,323,827,200 bytesを読む。attention・allocator・stagingを含むpeakは未測定。約14 GiB以上の空きを目安とし、初回I/Oは数分以上になる可能性がある。checkpointはread-only、全モデルはloadしない。

ログは`artifacts/text-front/run-日時-PID/test.log`、終了codeは`exit-code.txt`、binary / M1 summary hashは`identity.txt`。失敗・中断時はログを保持し、同じコマンドで新しいrunとして最初から再試行する。途中stateの永続化は行わない。結果を読み取るまで通過とは判定しない。PASSでも公式oracle / 全層 / 256K qualificationとは区別する。
