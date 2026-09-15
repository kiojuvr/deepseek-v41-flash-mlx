# omlx baselineとnative計画の更新

2026-09-13。公式checkpoint `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash` がM3 Ultra / 512 GBでそのまま動作したというユーザー確認を採用する。omlx v0.7.0.dev2を現時点の最優先の実動・性能referenceとし、本projectは同じ公式精度を保持してfull-path性能を上回る専用native runtimeを目指す。動作可能性を新規に示すことだけを価値目標にしない。

## ユーザー報告の性能baseline

以下は提供された値をそのまま記録したもの。本projectによる再測定ではない。

| Context（報告表記） | Prefill tok/s | Decode tok/s | TTFT s | Peak Mem（報告GB） |
| ---: | ---: | ---: | ---: | ---: |
| 32K | 193.6 | 35.5 | 169 | 292.82 |
| 64K | 190.8 | 36.3 | 344 | 292.85 |
| 128K | 176.4 | 29.0 | 743 | 292.90 |
| 200K | 183.9 | 36.9 | 1088 | 292.96 |

比較目標として直ちに使う。正式なpaired comparisonには元のtoken IDs / corpus、正確なprompt・output token数（特に200Kの定義）、生成設定、DSpark、Engram RAM / SSD、MoE offload、APC、prefill chunk、warm-up、依存revision、custom kernelのbuild・dispatch状況、OSと測定経路を記録する。報告値に未知の設定を補わない。既存ログ・設定の回収を先行し、再測定を計画修正の前提にしない。

固定版の`omlx/admin/benchmark.py`はMLX peakを`mx.get_peak_memory()`で取得し、表示用`peak_memory_gb`を`1024**3`で割る。この経路の表示値ならGiBであり、OS file cacheを含む総physical memoryとは異なる。decode集約値の逆数をtail p95 / p99へ置換しない。同ファイルには内部generation durationを利用する経路があるため、API token arrival latencyも別に比較する。

## 調査identityと確認した実装

[omlx固定commit](https://github.com/jundot/omlx/tree/b390b31e0c6831225fed0f24d278eb1db7fcb68b)は`v0.7.0.dev2`をshallow cloneして確認した`b390b31e0c6831225fed0f24d278eb1db7fcb68b`。調査用checkoutは`/private/tmp/dsv41-omlx-0.7.0.dev2`。ソースを読んだ段階であり、インストール・モデルload・benchmarkは実行していない。以下のpathはすべてこのcommitに対する相対path。

2026-09-13追記: ユーザー指定の永続checkout `/Users/kioju/omlx-0.7.0.dev2` も同じcommitで、working treeに変更がないことを確認した。今後の参照はこの配置を使う。[source manifest](../artifacts/references/omlx-v0.7.0.dev2.json)で対象sourceをcommitの内容と照合し、hashとユーザー報告値を保存した。既存の部分診断は[Engram forward](engram-forward.md)に記録されており、full-model load / benchmark再実行とは区別する。

| Path | 静的確認した内容 / native開発への意味 |
| --- | --- |
| `pyproject.toml` | MLX `0.32.2`、mlx-lm `ab1806e8f5d6aa035973af194a1b9198ab4754dc`を指定。ユーザーの実行環境identityは別途記録する |
| `omlx/patches/deepseek_v41/loading.py` | original / convertedの両方を扱い、source checkpointは`iter_source_weights`へ接続。Engram SSD設定とMTP保持設定を持つ。直接loadの対照とする |
| `omlx/patches/deepseek_v41/language.py` | compile対象の演算、Engram前後のasync evaluation、条件付きgrouped expert native dispatchが存在。batch=1専用化の削減余地を実測する |
| `omlx/patches/deepseek_v41/packed_attention.py`, `kernels.py` | packed CSA2、shapeによるMMA / fused / split dispatch、丸めを扱う専用Metal実装。attentionを汎用演算だけの比較相手と仮定しない |
| `omlx/custom_kernels/glm_moe_dsa/csrc/deepseek_v41_grouped_expert.cpp` | V4.1向けnative実装が既に存在。C++への言語置換だけを速度向上の根拠にしない |
| `omlx/patches/deepseek_v41/storage.py` | read-only mmap、resident packed table、CPU row gather、prefetch workerを持つ。Engram専用cacheの開発を先行させる根拠にはしない |
| `tests/test_deepseek_v41_*.py` | 数値・routing・attention・storage等の比較対象候補。テストの存在は本projectや当該実機での合格を意味しない |

実装研究の第一referenceをomlxへ変更する。モデル意味論のcanonical sourceは公式checkpoint / minimal inference、optimized pathのexactness基準は保持するlocal referenceのまま。omlxとの数値差は最初の境界まで局所化し、公式oracleとの比較で判断する。omlxへ一致させるためだけに既存許容値や演算を変更しない。

## 実装の優先順位

既存のEngram・モデル入口の進捗を踏まえた具体的な接続順序とexit境界は[native統合計画](native-integration-plan.md)に記載する。次の実装対象は共通packed FP8 / FP4 linearで、そのままmHC / Blockへ接続する。

1. **M2 full referenceを完成する。** 現在のEngram gate / residual差を局所化し、公式意味論を確認する。必要な短いfixture検証を続け、embedding→encoder→decoder→logitsと継続stateへ接続する。Engram単体の追加性能最適化はfull pathの成立に必要なものへ絞る。
2. **比較可能なnative generationを作る。** tokenized入力、teacher forcing / greedy、prefill / decode計測、state dump、feature identityを備える。固定omlxを差分診断に利用し、公式oracleの二段階exactnessを維持する。最初はtext plain ARを揃え、ユーザー報告がDSpark有効ならその性能比較は対応機能を揃えるまで保留する。
3. **M3で専用execution planを計測する。** batch=1と固定モデル構造を使ったCPU graph構築・dispatch・同期削減、buffer lifetime / scratch再利用、packed attention / MoEの中間materialization削減を候補とする。CED schedule / replayは意味論と継続stateの証明を先行させる。omlxが既に消している費用を削減余地として二重計上しない。
4. **M4で外部baselineを超える。** 32K→64K→128K→200K比較点→256Kを進める。full-path wall / tail TPTとmemoryを同条件で比較する。200Kの正確なtoken数は元測定から固定し、本projectのK=1,024定義へ無条件変換しない。
5. **M5 / M6は実測された必要性と最終用途に従う。** 独自Engram cache / prefetchはfull-path stallとmemory pressureで必要性を示してから実装。DSpark / Rust API / visionと最終256K qualificationは維持する。

想定する勝ち筋は測定前の仮説であり、native化だけで性能優位を保証しない。部分kernelの改善は内部開発の証拠に限定し、最終的には固定omlxに対する改善を示す。

## 契約への影響と再qualification

[architecture](architecture.md)、[correctness](correctness.md)、[memory](memory-layout.md)、[qualification](qualification.md)を同時更新した。既存のcheckpoint integrity、Engram部分oracleとstorage測定の証拠は保持する。本runtimeのM2未完了をomlxの成功で完了へ繰り上げない。

内部candidateはlocal exactnessと従来のpromotion gateを通し、外部性能優位はomlxとのpaired comparisonで別に判定する。測定の反復回数とregression禁止はqualification契約に従う。256Kのtail閾値はM3終了時に独立して事前固定し、200K集約値から推定してpassにしない。API / DSpark / vision / storage変更後は該当correctnessとfull-path / memoryを再qualificationする。

今の変更は計画・referenceの固定で、長時間検証を開始しない。後続の長時間比較は実行可能なharnessができた時点で、正確なcommand、負荷・scope、raw log / result path、失敗時の停止・保存とrun単位の再開を備えたユーザー実行スクリプトとして渡す。未完成runtime向けの架空のbenchmark commandは記載しない。
