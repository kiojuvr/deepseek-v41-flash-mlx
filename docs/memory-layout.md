# Memory layout contract

状態: M0の契約を維持し、M1のstatic accountingと実機budgetを[checkpoint atlas report](checkpoint-atlas.md)で確定した。実機のphysical memoryは512 GiB、Metal推奨working setは464 GiB。runtimeが全量を予約できる意味ではない。GBは10^9 bytes、GiBは2^30 bytes、MiBは2^20 bytes。本projectの1K tokensは1,024 tokens。

## Residency

| 領域 | 配置 | 方針 |
| --- | --- | --- |
| backbone / MoE weights | Unified Memory | 公式precisionを保持。weightsとscaleを数え、全体のBF16複製を作らない |
| persistent global main KV | Unified Memory | 公式FP4形式。SSDへのtieringを初期設計へ入れない |
| persistent indexer K / scales | Unified Memory | KV source単位で保持し、Reindex layerでは共有する |
| SWA / replay state | Unified Memory | boundedなlive ring・replay入力・補助state。全履歴SWA KVを保存しない |
| indexer候補 / Top-K / scratch | Unified Memory | persistent Kと区別し、lifetimesとcapacityを明示 |
| DSpark / vision weights・state | Unified Memory | 未有効の段階でもresident予算へ計上 |
| Engram full backing store | SSD | canonical rowとscaleを保持するread-only backing |
| Engram working set | OS file-backed page cache / Unified Memory | mmap / buffered preadを基準とし、独自の永続row cacheは初期実装しない |
| Engramの小規模projection・hash等 | Unified Memory | atlasで巨大lookup tableと分離して分類 |

2026-09-12更新: [Engram page-cache baseline](engram-page-cache.md)を採用する。OS cacheの64 GiBはmemory planning上の目安であり、per-file hard capやpinを意味しない。anonymous stagingには独立した上限とownershipを持たせる。

checkpoint自体のSSD保存は通常のload元。推論時のstorage subsystemとしてSSDを使う対象はEngram backingとする。OS file cache、mmapのresident pages、runtime hot cacheの物理的重複をmemory計測で確認する。

## Global KVの890 bytes/token

以下は取得済み公式config、`inference/model.py`と`kernel.py`から導いたpacked payloadの見積り。allocatorや実装の実測ではない。[provenance](architecture.md)を参照。

| 1 compressed position / sourceあたり | 計算 | Bytes |
| --- | --- | ---: |
| main KV（head dim 512、FP4 E2M1） | 512 / 2 | 256 |
| main KV scales（16 channelsごとのE4M3） | 512 / 16 | 32 |
| indexer K（head dim 128、FP4） | 128 / 2 | 64 |
| indexer K scales（32 channelsごとのE8M0） | 128 / 32 | 4 |
| 合計 | 288 + 68 | 356 |

KV source 2 / 8 / 14はratio 2、source 20はratio 1。完了groupだけを保存する場合、T tokensのpayloadは`356 × (3 × floor(T / 2) + T)` bytes。偶数Tでは`356 × 2.5 = 890 bytes/token`となる。main KVが720 bytes/token、persistent indexer Kが170 bytes/tokenを占める。**890にはpersistent indexer Kとscaleを含む。** candidate / Top-K index、allocator、temporary stateは含まない。

| 総context | Tokens | Packed global payload |
| --- | ---: | ---: |
| 32K | 32,768 | 27.8125 MiB |
| 64K | 65,536 | 55.625 MiB |
| 128K | 131,072 | 111.25 MiB |
| 256K | 262,144 | 222.5 MiB |

256Kでは233,308,160 bytes（main KV 180 MiB + indexer K 42.5 MiB）。capacity rounding、alignment、未完了group、SWA、replayに必要なencoder状態、token履歴、indexer scratchは別途予算化する。奇数長や予約capacityでは単純な890×Tと実allocationが一致しない。

minimal inferenceは`fp4_act_quant(..., inplace=True)`で量子化後の値をdequantizeしてcacheへ書くため、そのarray allocationをpacked payloadサイズと同一視しない。productionのpacked表現は同じdecode値・cast境界を再現してexactnessを検証する。indexer Kの170 bytes/tokenをworking stateとして二重計上しない。

## M1 checkpoint atlas

最初の実装としてC++の[`tools/inspect_checkpoint/`](../tools/inspect_checkpoint/README.md)を追加した。巨大payloadを一括loadせず、safetensors index / headerからatlasを作り、integrityのpayload検証を別段階で行う。checkpointを変更しない。

tensorごとに最低限以下を出力する。

| Field | 意味 |
| --- | --- |
| `name`, `shard`, `data_offsets` | canonical tensor名、所在、payload範囲 |
| `storage_dtype`, `storage_shape`, `storage_bytes` | header上の表現とoffset差で確定する実bytes |
| `logical_dtype`, `logical_shape`, `logical_numel` | FP4 packing等を解釈したモデル上の値。根拠未確認ならunknown |
| `scale_tensor`, `scale_format`, `group_size` | quantizationの対応関係 |
| `semantic_owner`, `layer_id`, `expert_id`, `role` | encoder / decoder / MoE / Engram backing / Engram resident / DSpark / vision等 |
| `residency`, `alias_group`, `runtime_bytes` | 予定配置、共有関係、alignment / repack込みの予算 |
| `revision`, `digest`, `completeness` | snapshot、integrity、missing / partial / header-valid / verifiedの区別 |

unknown owner / dtype / shapeを黙ってresidentへ分類しない。indexとheaderの欠落・重複・shape・offset・範囲外・payload長を照合する。`.incomplete`はcompleted shardへ数えない。全期待shardが揃い、整合性が確認できるまではpartial atlasを返す。全payload digestはダウンロード完了後に公式metadataとの対応を確認し、単なるファイルサイズ一致と区別する。

集計はstorage dtype、logical dtype、owner、layer、shard、residency別に行う。scale等を含む保存要素数とmodel parameter数を別列にし、weight共有も区別する。公式model cardの552B backbone / 196B Engramと計画入力の485Bをこの集計で照合する。M0でどちらかをresident memoryの根拠にしない。

## 512 GBへのmappingの確定条件

`peak physical memory`を次の構成で見積り、M2以降に実測する。

```text
resident weights + scales + required repacking
+ packed global KV + indexer K
+ bounded SWA / replay / compression / n-gram state
+ DSpark / vision transient state
+ peak activations / indexer workspace / Metal scratch
+ Engram file-backed working set + bounded anonymous I/O staging
+ allocator / graph / runtime overhead
+ OS and operational headroom
```

shared buffersは一度だけ数え、参照countと物理allocationを分ける。load / conversion / warm-upのpeakもsteady stateとは別に計上する。Engram file-backed working setの予算目安、OS余裕、Metal working-set limit、実使用可能memoryはM1の数値根拠とM2の実測で固定する。OS eviction policyとruntimeが制御するallocation上限を区別する。未確定項目を0扱いしてfit判定しない。

256K分のglobal KVが小さいことはモデル全体のfitを保証しない。予算を超えた場合はM1不合格として内訳を提示し、KV offloadや非公式量子化へ黙って切り替えない。

## Engram subsystem

M2ではread-only mmapとbuffered positional readによるexact lookupを比較し、OS page cacheを基準にする。M5はworking-set / page amplification / I/O stallの計測とpressure qualificationを完成させる段階とする。独自page cache、row cache、async prefetchは計測で必要性が示された場合に限って導入する。

rowとscaleをboundedな所有bufferへgatherしてからGPU consumerへ渡す。lookup missをzeroや近似値で代用しない。OS cache hitとmissで同じBF16値を返し、mmapの仮想容量をresident bytesとして数えない。mapping lifetimeを保つことと物理pageをpinすることは別。full tableをmlockしたり巨大mappingをMetal bufferとしてGPUへ直接渡したりしない。

mmap中にbackingをtruncateする操作はOS signalの原因になり得るため、checkpointは実行中immutableを前提とする。通常の変更はfile identity / size / timestampsで検出するが、悪意ある並行変更を排除する仕組みやSIGBUS復旧を実装済みとは扱わない。I/O errorを呼出側で扱う必要が強い場合はpread pathを選べる。natural cache条件、unique row/page、process I/O countersとfull-path TPTを分けて報告する。
