# deepseek-v41-flash-mlx

DeepSeek-V4.1-Flash専用のApple Silicon inference runtime。Apple M3 Ultra / 512 GB Unified Memory、single-node / single-userを対象に、約256K tokensまで成長する長時間coding-agent sessionの安定decodeを優先する。OpenCode等から利用するOpenAI互換APIを最終成果物とする。

**M1完了。M2ではEngramのhash / row read / Metal BF16復元を公式部分oracleと照合済み。full推論・API・性能・runtime peak memoryは未検証。** 最初のコミットは5文書のみで作成し、その契約に従ってC++20の実装を進めている。

全48 shard・96,085 tensorを照合し、全88ファイルの公式digest検証が通過した。Engram backingは202.758 GB、Unified Memory対象weightsは307.528 GB。[M1 report](docs/checkpoint-atlas.md)と[build / 実行手順](tools/inspect_checkpoint/README.md)を参照。

## 2026-09-13: 目標の更新

omlx v0.7.0.dev2で、M3 Ultra / 512 GB上の公式checkpointの直接動作と200Kまでの良好な性能がユーザーにより確認された。これを現時点の最優先の実動・性能referenceとし、本projectの価値目標を**同じ公式精度でomlxを上回る専用native runtime**へ更新する。native化の優位は未実証。M2のfull path完成を優先し、omlxとの同条件比較で改善を確認する。[参考実装の調査・報告値・改訂計画](docs/omlx-baseline.md)を参照。

## Runtime contract

- runtime coreはC++20を基準とし、必要なC++23機能はApple Clangとの互換性を確認して採用する。MLX C++ APIとMetal Shading Languageを直接使う。
- API / protocol層はRust + 公式deepseek-recipeを採用する。M3段階から非RELEASEの[developer API server](docs/developer-api-server.md)を起動でき、protocol検証を先行する。公式Axum server exampleを参考にnative runtimeを接続し、prompt・tool・reasoning・streaming parserを独自に再実装しない。
- 公式checkpoint tensorをcanonical source、DeepSeek公式minimal inferenceをモデル意味論のreference、omlx v0.7.0.dev2をApple Silicon上の実動・性能・実装構造の第一reference、vLLMを補助referenceとする。
- optimized pathは保持し続けるlocal reference pathに対してexactnessを要求する。correctnessを速度より優先する。
- backbone、MoE、attention state、global KV、indexer state、DSpark、visionをUnified Memoryに置く。Engram full backing storeをSSDに置き、working setはOS file-backed page cacheを基準にUnified Memoryへ残す。独自row cacheは必要性が実測で示されてから導入する。
- 公式FP4 main KVを実装し、追加の近似・再量子化を行わない。SWAはboundedなlive stateとBounded Replayを使い、全履歴のSWA KVは永続保存しない。APC / prefix reuseは後段に追加する。
- Metalは最初からruntimeの構成要素とする。M2でfull reference pathを接続し、M3からfull-pathの性能最適化を始める。

## 禁止事項

1. partial optimizationをproduction candidateとしない。
2. Pythonをproduction runtime dependencyにしない。
3. approximate KVを導入しない。
4. unofficial quantizationをbaselineにしない。
5. kernel単体speedupを昇格理由にしない。
6. full-path wall time / decode TPTが改善しなければoptimizationをrejectする。
7. reference pathを削除しない。
8. 256K qualification前にRELEASEしない。
9. GLM runtimeからコードをコピーしない。M0では再利用を検討する知見のinventoryだけを作る。

Pythonは外部oracle、fixture生成、開発時の分析に限って使用できる。productionのprefill、decode、sampling、cache、request処理にPythonへの往復を含めない。

## 文書

full backbone接続後の優先作業は[logits oracle比較](docs/logits-oracle-plan.md)。公式kernel実行、CPUへの式転記、固定oMLX対照を区別する。

| 文書 | 契約 |
| --- | --- |
| [architecture](docs/architecture.md) | native dataflow、責務、referenceの固定、GLM inventory |
| [correctness](docs/correctness.md) | canonical source、exactness、state検証、昇格条件 |
| [memory-layout](docs/memory-layout.md) | residency、KV内訳、Engram分離、M1 atlas仕様 |
| [qualification](docs/qualification.md) | M0–M6、32K→256K、wall/TPT hard gate |
| [Engram page cache](docs/engram-page-cache.md) | M2のhash / row lookup / Metal exactnessとユーザー実行のstorage比較 |
| [Engram forward](docs/engram-forward.md) | native projection / gate / residual、演算別診断と固定omlx対照 |
| [Packed linear](docs/packed-linear.md) | 共通FP8 / FP4 linear、固定scheduleとoMLXとの差 |
| [mHC](docs/mhc.md) | mix・collapse・展開のnative境界、係数の未解決数値差 |
| [MoE / Block](docs/moe-block.md) | layer 0のresident expertとBlock接続、ユーザー実行検査 |
| [Text front](docs/text-front.md) | token ID → embedding → layer 0、継続stateと実行検査 |
| [3層text検証](docs/text-triple-validation.md) | Engramとcompressed attentionを含むlayer 0〜2の実行手順 |
| [4層text検証](docs/text-quad-validation.md) | layer 3のglobal KV再利用を含む実行手順 |
| [encoder 0..19検証](docs/text-encoder-validation.md) | producer 2/8/14・reuse 3–7/9–13/15–19・Engram 1/14の実行手順 |
| [decoder/backbone検証](docs/text-decoder-validation.md) | decoder 20..39・candidate二段Top-K・token→logitsの実行手順 |
| [logits oracle計画](docs/logits-oracle-plan.md) | oracleの3分類、最初の比較単位、Main KV記述の訂正 |
| [trace/比較harness](docs/logits-oracle-harness.md) | native境界trace、固定oMLX外部hook、2 traceの境界比較手順 |
| [logits oracle結果](docs/logits-oracle-results.md) | 最初の分岐、RMSNorm修正、mHC reduction順の残差 |
| [sampling/generation](docs/sampling-reference.md) | greedy / temperature、参照RNG、生成loopの実行手順 |
| [SWA reference](docs/swa-reference.md) | bounded state、online softmax / sink境界と未接続範囲 |
| [Compressed attention](docs/compressed-attention.md) | layer 2 compressorとglobal KV / indexerの接続境界 |
| [Model entry](docs/model-entry.md) | native embedding / 初期pre-mix / RMSNormと短い実checkpoint検査 |
| [checkpoint atlas](docs/checkpoint-atlas.md) | M1の全tensor集計、integrity証拠、実機mappingとmemory予算 |

契約変更は理由・影響・必要な再qualificationを同じ変更に記載する。失敗したoptimizationに合わせて基準を緩めない。

## 対象checkpointと未確定事項

開発時のcheckpoint pathは`/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`。2026-09-11にダウンロード完了を確認し、baselineを`dba1be0a40aa45a94ad051997016db3960a90277`へ固定した。全tensor payloadは510,286,023,000 bytes。M0当時の資料identityは[architecture](docs/architecture.md)、現baselineは[M1 report](docs/checkpoint-atlas.md)とverification manifestに記録した。

M1集計は約551.881B backbone + 196.614B Engram + 14.225B DSpark + 0.485B visionで、計**763,205,315,794 logical parameters**。固定revisionのHugging Face API totalと一致した。計画入力の485Bはこのsnapshotのbaseline値として使わない。公式I8保存のrouted expertsはpacked FP4であり、INT8量子化モデルではない。

公式資料の1M context推奨はモデル側の能力・設定であり、本runtimeが広告できる上限はqualification済みの総context長とする。初期release gateは256K＝262,144 tokens。生成上限とcontext上限は別に扱う。

M2は進行中。[Engram native forward](docs/engram-forward.md)を接続したが、公式gateとのFP32差があり、layer 14のresidualにBF16 1 ULP差が1要素残るため未qualified。2026-09-13にproducerをlayer 2/8/14、reuse consumerを3–7/9–13/15–19へ一般化し、[encoder 0..19のtext reference](docs/text-encoder-validation.md)を接続、ユーザーrunのlocal整合性検査を確認した。decoder 20..39（candidate二段Top-K、index source 24/28/32/36）と[full backbone token→logits](docs/text-decoder-validation.md)も接続し、ユーザーrunのlocal整合性検査を確認した。routed expertはon-demand load。公式logits oracle比較とsampling / generation / DSpark / vision / APIは未完了。[OS page cache比較の6 run](docs/engram-cache-results.md)を確認し、mmap baselineを維持した。SSD miss・backbone同居時の性能は未判定。atlasのintegrity成功や部分oracleとの一致を、full推論exactnessや256K qualificationの代わりにしない。

2026-09-13: Engramの固定projection診断でomlxとnativeのdot / gateがbitwise一致し、同じCPU oracle差を再現した。[モデル入口](docs/model-entry.md)も追加し、embedding / 初期pre-mixは一致。RMSNormはCPU oracleに対して5要素のBF16 1 ULP差があり未qualified。どちらもfull-path完成・性能昇格には数えない。
