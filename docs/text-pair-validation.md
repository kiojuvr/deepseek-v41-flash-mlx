# 2層text referenceの検証手順

2026-09-13、ユーザーrun `run-20260913-144719-25596`のログ、終了code 0、binary / M1 summary / Engram metadata / fixture provenanceの4件のSHA256を確認した。すべてレビュー時の現物と一致し、以下のlocal整合性検査は通過した。[レビュー記録](../artifacts/text-pair/reviewed-result.json)にログ本文とidentityを保存した。

変更なしでの再実行は不要。以下は再現用の手順。公式oracle比較・全モデルqualification・性能測定は未完了。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_text_pair_reference.sh
```

## 範囲

token IDs `[0,42,1000]` → embedding → layer 0 → Engram 1 → layer 1を実行する。両層の全384 routed expertをresident loadし、選択されたexpertを計算する。全expertを実行する網羅検査ではない。

- 3 token chunkとtoken逐次のhidden / pre-mix / 両KV stateをbit比較する。
- Engram hash stateのcopyへ4 tokenを追加し、将来のhash列が一致することを確認する。hash内部メモリ全体のbit比較ではない。
- 4 token目への継続、fork分離、reset後の出力・hash・KV再現性を確認する。
- image / 語彙外token、および2層のposition不一致を拒否し、拒否前後のstateを比較する。
- metadata bytesを既存Engram fixture provenanceのSHA256と照合する。

拒否検査は入口のvalidationに対するもの。GPU実行途中・SSD read途中に障害を注入するrollback検査はまだ含めない。公式oracleとの数値一致、長文、APC、生成品質、全層や性能のqualificationも対象外。

## 資源と結果

2層のresident expert、embedding、attention、Engram projectionをloadする。約24 GiB以上の空きを目安とするが、allocator / stagingを含む実peakは未測定。EngramはSSD mmapから必要rowを読み、全backingをRAMへ読み込まない。checkpointはread-only。初回I/Oやreference projectionによって数分以上になる可能性があるため、エージェント側では起動しない。

`artifacts/text-pair/run-日時-PID/`へ以下を保存する。

- `test.log`: stdout / stderrとPASSまたは最初の失敗。
- `exit-code.txt`: 検査またはログ保存の終了code。0以外は失敗。
- `identity.txt`: binary、M1 summary、Engram metadata / fixture provenanceのSHA256。

失敗時は非ゼロ終了。Ctrl-C等で終了codeファイルが残らないrunも未完了として扱う。中断・失敗時はログを保持し、同じコマンドを再実行すると別runとして最初から再開する。途中のモデルstateは保存しない。ログとidentityをレビューするまで通過とは判定しない。
