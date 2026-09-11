# M1 checkpoint atlas — 2026-09-11

**全48 shard・96,085 tensorのheader / index照合、semantic owner分類、および全88ファイルの公式digest照合が完了した。** 欠落、分類不能、scale不一致、integrity failureは0。これはcheckpointのaccounting / integrity完了であり、推論exactnessやruntime peak memoryのqualificationではない。

対象snapshotは[`dba1be0a40aa45a94ad051997016db3960a90277`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277)。取得先は`/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`。M0時点の途中snapshotから更新されたREADMEとencoding 2ファイルも今回の固定revisionで検証した。configとM0に固定したinference 9ファイルは当時のSHA-256と一致する。

## 保存容量とsemantic owner

| Owner | Tensors | Storage bytes | GiB |
| --- | ---: | ---: | ---: |
| Encoder（Engramを除く） | 46,704 | 147,838,599,136 | 137.685 |
| Decoder | 46,699 | 147,823,271,392 | 137.671 |
| Token embedding | 1 | 1,323,827,200 | 1.233 |
| Output head / norm | 2 | 1,323,837,440 | 1.233 |
| Engram resident projections | 8 | 315,043,840 | 0.293 |
| DSpark | 2,401 | 7,932,874,632 | 7.388 |
| Vision / aligner / image delimiters | 266 | 970,536,960 | 0.904 |
| **Unified Memory対象weights合計** | **96,081** | **307,527,990,600** | **286.408** |
| **SSD Engram table + scales** | **4** | **202,758,032,400** | **188.833** |
| **全tensor payload** | **96,085** | **510,286,023,000** | **475.241** |

全shardのファイルサイズ合計は510,296,708,312 bytes。payloadとの差10,685,312 bytesはsafetensors headerとlength prefix。indexの`metadata.total_size`はpayload合計に一致した。

Engram backingはlayer 1 / 14の`engram.embed.weight`と`engram.embed.scale`だけを対象とする。row数は384,006,168 / 384,016,682、1 rowは256 FP8 bytes + 8 E8M0 scale bytes＝264 bytes。巨大tableはshard 47 / 48にあり、同じshard内の`engram.wkv`や`q_weight` / `k_weight`はresident対象。shard単位で配置を決めると誤る。

## Parametersとdtypeの照合

| Storage dtype | Storage bytes | 意味 |
| --- | ---: | --- |
| BF16 | 3,952,883,712 | weights / norm等 |
| F32 | 169,229,128 | mHC / bias / sink等 |
| F8_E4M3 | 204,015,223,296 | FP8 dense / shared expert / Engram weights |
| F8_E8M0 | 23,563,015,184 | quantization scales。parameter集計から分離 |
| I8 | 278,585,671,680 | 1 byteに2つのFP4 E2M1をpackingしたrouted experts |

保存要素数は508,182,659,298。FP4をlogicalに展開すると、scale込みのlogical numelは786,768,330,978、**scaleを除いたparametersは763,205,315,794**となり、固定revisionのHugging Face APIが返したtotalと完全一致した。I8保存をINT8モデルと解釈したり、scaleをparametersに含めたりしない。

| Logical parameter区分 | Parameters |
| --- | ---: |
| Backbone（encoder / decoder、embedding / head、Engram resident projections） | 551,880,835,184 |
| Engram lookup tables | 196,613,849,600 |
| DSpark | 14,225,362,530 |
| Vision / aligner / image delimiters | 485,268,480 |
| **Total** | **763,205,315,794** |

公式model cardの552B backbone / 196B Engramはこの分離と整合する。当初の「485B」は今回の保存要素数・logical parametersのいずれにも一致しない。以前のHub表示の由来は確定せず、baselineを上記の再現可能な集計へ置き換える。

## 実機mappingとM2 admission予算

[実機probe](../artifacts/checkpoint/hardware.json)の結果はApple M3 Ultra、physical memory **512 GiB**、Metal recommended maximum working set **464 GiB**、maximum buffer length **348 GiB**。OSはmacOS 26.5.2（25F84）。これはdeviceから読んだ上限・推奨値で、allocation成功やresident維持の実測ではない。

packed resident weightsは286.408 GiB。公式converterでは43個の`attn.wo_a.weight`をBF16へ展開するため、同じreference表現を使う場合の追加allocationは、元のpacked weights / scalesも保持する保守的見積りで**2.6875 GiB**。全weightsのBF16化は行わない。runtime repackingとalignmentは別途計測する。

M2の実装は以下の予算内に収める方針とする。2026-09-12からOS cache分はruntimeが強制できる上限ではなく目安とし、physical footprint / pressureを別に監視する。実測値がない項目の0扱いを避けるための予算であり、必要量の検証済み上限ではない。超過したsubsystemは設計を再評価し、黙ってKV offloadや非公式量子化へ変更しない。

| Runtime領域 | Budget GiB | 根拠 / 残る検証 |
| --- | ---: | --- |
| Resident weights / scales / repacking | 296 | 286.408 + wo_a 2.6875 + 残り約6.905をlayout余裕。DSpark / vision weightsも含む |
| Global KV / persistent indexer K | 1 | 256K payloadは0.2173 GiB。capacity / alignmentはM2で確認 |
| SWA / replay / compression / token / n-gram state | 8 | replay開始時に必要なstateとlifetimesは未確定 |
| Prefill / decode activations、attention / indexer / MoE scratch | 48 | chunkingと最大同時生存量の設計・実測が必要 |
| DSpark / vision temporary state | 8 | weight容量とは別。有効化時に再測定 |
| Engram file-backed working set | 64 | 2026-09-12更新: OS page cache用の予算目安。per-file hard capではない |
| Engram anonymous I/O staging | 8 | runtime側で制限する枠。OS file pagesは64 GiB側へ計上 |
| Runtime / graph / allocator overhead | 8 | MLX graph retentionとallocator slackを測定 |
| **Runtime合計予算** | **441** | Metal推奨464 GiBより23 GiB小さい。OS cacheを含むため強制上限ではない |
| OS / 他process用の余裕 | 64 | 実機512 GiBからruntime予算を差し引いて確保 |
| **未割当の余裕** | **7** | 441 + 64 + 7 = 512 GiB |

weightsの所在とstatic payloadのmappingは確定した。runtime peakは未qualifiedのまま残す。上記441 GiBにはstartup / repacking peakも含め、shardを順次読むload設計とする。Metal bufferは348 GiB上限を超えず、ownerとlifetimeごとに分割する。512 GiB実機に全checkpointを一度loadしてからEngramを追い出す手順は採らない。

## 証拠と次工程

- [Upstream manifest](../artifacts/checkpoint/upstream-manifest.json): 固定revisionのAPI metadata。各ファイルのexpected digestとサイズ。
- [Verification](../artifacts/checkpoint/verification.json): 48 shardのLFS SHA-256と、config / tokenizer / 公式資料を含む全88ファイルの照合結果。
- [Summary](../artifacts/checkpoint/summary.json): 全集計、header digest、問題一覧。
- [Run provenance](../artifacts/checkpoint/run-provenance.json): tool / dependency / buildと結果ファイルのidentity。
- 全tensorの`atlas.jsonl`は`artifacts/checkpoint/`に生成済み。約92 MBのためGit対象外とし、[実行手順](../tools/inspect_checkpoint/README.md)から再生成する。

C++ toolの検査対象はindex/header coverage、packing / scale、owner、integrityであり、モデルの数値正当性ではない。次はM2: canonical weightsのnative readerとlocal referenceを接続し、token → encoder → decoder → logitsを公式oracleと照合する。replay・作業領域・residencyのruntime実測はこの段階から開始する。

小型fixtureの23試験はRelease buildとAddressSanitizer / UndefinedBehaviorSanitizer buildの両方で通過した。全checkpointの最終検証はRelease buildで実行した。モデル推論やGPU kernelの数値試験はまだ行っていない。
