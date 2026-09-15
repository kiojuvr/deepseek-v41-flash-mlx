# 3層text referenceの検証

2026-09-13。layer 2のmHC / norm / compressed attention / MoEを`CompressedBlockReference`へ接続し、`TextTripleReference`で既存2層経路の後ろへ追加した。ユーザーrun `run-20260913-164045-27655`のログ、終了code 0、binary / M1 summary / Engram metadata / fixture provenanceの4件のSHA256を確認した。すべてレビュー時の現物と一致し、以下のlocal整合性検査は通過した。[レビュー記録](../artifacts/text-triple/reviewed-result.json)にログとidentityを保存した。

変更なしでの再実行は不要。以下は再現用の手順。memory peak・経過時間・throughputは測定しておらず、今回のPASSから推定しない。

MoEのownerはlayer 0〜2を受け付ける。pure SWA attention / Blockは引き続き0・1のみで、layer 2を誤ってpure SWAとして扱わない。layer 2のFFNにはattention側pre-mixを渡し、FFN側pre-mixを次layerへの出力にする。

3層stateはEngram hash、layer 0/1のSWA、layer 2のwindow・global packed cache・pending compressorを含む。作業用copyで2層経路とlayer 2を実行し、両方の評価成功後にまとめてcommitする。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_text_triple_reference.sh
```

## 検証範囲

- token IDs `[0,42,1000]`のchunk/tokenwise hidden・pre-mix・3層stateのbit一致。
- 3 tokenでglobal KVが1 group、4 token目で2 groupになること。
- 4 token目への継続、fork分離、reset後の出力・state一致。
- image / 語彙外token、layer 2のposition不一致の拒否とstate保持。
- hashのcopyに4 tokenを追加した継続hash列の一致。

公式oracle比較、全expertを必ず選択させる網羅試験、Top-Kを絞る長さ、実行途中の障害注入は含まない。mHC / norm / Engram / FP4 / attentionの未完了oracle gateは継続する。

## 資源・ログ・再試行

3層分の全384 expertとembedding・attention・Engram projectionをresident loadする。空き約36 GiB以上を目安とするが、staging / allocatorを含む実peakは未測定。EngramはSSD mmapの必要rowだけを読む。checkpointはread-only。初回I/O等で数分以上になる可能性があるため、エージェント側では起動しない。

ログは`artifacts/text-triple/run-日時-PID/test.log`、終了codeは`exit-code.txt`。`identity.txt`へbinary、M1 summary、Engram metadata / fixture provenanceのSHA256を保存する。検査またはログ保存が失敗した場合は非ゼロ終了する。

失敗・中断後は同じコマンドで新しいrunとして最初から再試行する。途中stateは保存しない。終了codeがないrunは未完了。ログとidentityをレビューするまでPASSとは判定しない。既存2層の結果は過去binaryの証拠として維持し、新しい3層検査へ自動では引き継がない。
