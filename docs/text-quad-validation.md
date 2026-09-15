# 4層text referenceの検証

2026-09-13。`ReusedBlockReference`にlayer 3のmHC / norm / consumer attention / resident MoEを接続し、`TextQuadReference`を追加した。ユーザーrun `run-20260913-171330-28350`のログ、終了code 0、binary / M1 summary / Engram metadata / fixture provenanceの4件のSHA256を確認した。すべてレビュー時の現物と一致し、以下のlocal整合性検査は通過した。[レビュー記録](../artifacts/text-quad/reviewed-result.json)にログとidentityを保存した。

変更なしでの再実行は不要。以下は再現用手順。今回のログにはmemory peak・経過時間・throughputの測定がなく、性能や公式oracleとの一致は判定しない。

各tokenをlayer 0 → Engram 1 → layer 1 → layer 2 → layer 3まで通してから次tokenへ進む。layer 2が作ったそのtokenのpublicationをlayer 3へ渡す。chunk最後の候補を過去tokenへ使う経路は作らない。chunk全体の成功後にhash・4層stateをまとめてcommitする。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_text_quad_reference.sh
```

## 範囲と資源

固定token IDs `[0,42,1000]`と継続token `42`で、chunk/tokenwise hidden・pre-mix・4層stateのbit一致、global group境界、fork/reset、不正token・layer 3 position不一致の拒否を検査する。Engram hashはcopyへ4 tokenを追加した将来のhash列で比較する。

4層の全384 routed expertをresident loadし、選択されたexpertを計算する。全expertを実行する網羅試験ではない。embedding・attention・Engram projectionも含め、空き約48 GiB以上を目安とする。実peakは未測定。Engramはmmapで必要rowのみ読み、checkpointはread-only。初回I/O等で数分以上になる可能性があるため、エージェント側では起動していない。

## ログと再試行

`artifacts/text-quad/run-日時-PID/`へ`test.log`、`exit-code.txt`、binary / M1 summary / Engram metadata / fixture provenanceのSHA256を含む`identity.txt`を保存する。検査またはログ保存の失敗は非ゼロ終了。

失敗・中断時は同じコマンドで新しいrunとして最初から再試行する。途中stateは保存しない。終了codeがないrunは未完了。ログとidentityのレビュー前に通過とは判定しない。

公式oracle比較、処理途中の障害注入、512候補を超えるTop-K、長文性能は対象外。既知mHC / norm / Engram差、FP4 / index / attentionの独立oracle未検証を引き継ぐ。3層の過去PASSを4層の証拠へ転用しない。
