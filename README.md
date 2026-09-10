# deepseek-v41-flash-mlx

DeepSeek-V4.1-Flash専用のApple Silicon inference runtime。Apple M3 Ultra / 512 GB Unified Memory、single-node / single-userを対象に、約256K tokensまで成長する長時間coding-agent sessionの安定decodeを優先する。OpenCode等から利用するOpenAI互換APIを最終成果物とする。

**現在はM0（repository contract）のみ。推論・API・性能・512 GBへの収容は未検証。** 最初のコミットは本書と`docs/`の4文書だけで構成する。実装、build設定、空のsource treeはM1以降に追加する。

## Runtime contract

- runtime coreはC++20を基準とし、必要なC++23機能はApple Clangとの互換性を確認して採用する。MLX C++ APIとMetal Shading Languageを直接使う。
- 公式checkpoint tensorをcanonical source、DeepSeek公式minimal inferenceをモデル意味論のreference、vLLMをproduction構造のreferenceとする。
- optimized pathは保持し続けるlocal reference pathに対してexactnessを要求する。correctnessを速度より優先する。
- backbone、MoE、attention state、global KV、indexer state、DSpark、visionをUnified Memoryに置く。Engram full backing storeをSSDに置き、working setをUnified Memoryにcacheする。
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

| 文書 | 契約 |
| --- | --- |
| [architecture](docs/architecture.md) | native dataflow、責務、referenceの固定、GLM inventory |
| [correctness](docs/correctness.md) | canonical source、exactness、state検証、昇格条件 |
| [memory-layout](docs/memory-layout.md) | residency、KV内訳、Engram分離、M1 atlas仕様 |
| [qualification](docs/qualification.md) | M0–M6、32K→256K、wall/TPT hard gate |

契約変更は理由・影響・必要な再qualificationを同じ変更に記載する。失敗したoptimizationに合わせて基準を緩めない。

## 対象checkpointと未確定事項

開発時のcheckpoint pathは`/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`。2026-09-10時点でダウンロード途中であり、進捗表示の転送量をcheckpoint総容量として使わない。取得済み公式資料のrevisionとSHA-256は[architecture](docs/architecture.md)に記録した。

取得済み公式model cardは**552B backbone + 196B Engram**と記載している。計画入力にある「485B params / BF16・F32・F8・I8」は未照合で、logical parameters、packed tensorの保存要素数、storage dtype、scale等の補助tensorを区別してM1で解消する。parameter数からresident bytesを断定しない。

公式資料の1M context推奨はモデル側の能力・設定であり、本runtimeが広告できる上限はqualification済みの総context長とする。初期release gateは256K＝262,144 tokens。生成上限とcontext上限は別に扱う。

次の実装作業はM1 checkpoint atlas。全tensorのname / dtype / shape / bytes / shard / semantic ownerを収集し、Engram backingとresident領域を分離して512 GBへのmappingを確定する。全checkpointが揃うまではpartial atlasと表示し、M1完了や推論可能とは扱わない。
