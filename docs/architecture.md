# Architecture contract

状態: M0 / 2026-09-10。以下は実装契約であり、完成済みruntimeの記述ではない。

## 対象とnative boundary

Apple M3 Ultra / 512 GB Unified Memoryの1台で、1ユーザーの長時間coding-agent sessionを実行する。最初のschedulerは1本のactive generationを直列に扱う。分散推論、多ユーザーthroughput、汎用モデル対応は初期scopeに含めない。

C++20 runtimeからMLX C++のarray / graph / streamを使い、必要な演算をMetal kernelへ接続する。MLXの[公式C++ example](https://github.com/ml-explore/mlx/blob/main/examples/cpp/tutorial.cpp)でもarrayとlazy graphの直接操作を確認できる。C++23機能、MLX revision、Apple Clang、Metal compilerの組合せはbuild導入時に固定する。

```text
OpenAI-compatible API / protocol / request lifecycle (C++)
                         |
              session / scheduler / generation (C++)
                         |
         model + explicit state ownership (C++ / MLX)
             |                       |
        Metal kernels        Engram I/O + hot cache (C++)
             |                       |
          Apple GPU             SSD backing store
```

HTTPやprotocolのためにmodel dataflowをPythonへ戻さない。将来Rust serverを分離する場合も、native runtimeを1つの境界として呼び出し、tensor操作を細切れにFFI越しに制御しない。初期実装はC++とMetalに限定する。

## モデルの責務

CEDはencoder 20層（0–19）、decoder 20層（20–39）。decoder global KVはencoder最終状態から投影して共有する。各decoder layerのhiddenから独立したglobal KVを作る構造へ変更しない。実際の投影入力は公式コードのmHC collapse、normalization、compressorまで追跡し、「encoder出力」の呼称だけで演算を省略しない。

CSA2のFull / Reindex / Reuseはconfigとsource layerの関係から明示的に構築する。取得済みconfigのKV sourceは`[2, 8, 14, 20]`、index sourceは`[2, 8, 14, 20, 24, 28, 32, 36]`、candidate sourceは20。main KVとindexer Kの共有、Top-K再利用、causal visibilityを別々の責務として扱う。index source数だけpersistent Kを複製しない。

最初のend-to-end pathはtoken → embedding / Engram → encoder → decoder → norm / logitsまで接続する。MoE、single-pass mHC、CSA2、Engramを省いたpathをモデルreferenceとして認めない。minimal inferenceの全面forwardをまずreferenceとして移植し、CED prefill短縮とSWA Bounded Replayは同じlogits・stateを再現する別execution scheduleとして検証する。

DSparkとvisionのtensor所有権・memory予算はM1から確保する。M2はtext targetのplain autoregressive pathを通し、DSpark forwardとvisionの意味論をfixture対象にする。DSpark verification / commitと画像APIのfull integrationはM6とする。未実装機能を黙って無視して成功扱いしない。

## Ownershipとexecution

- modelはimmutable weightsとsemantic tensor mappingを所有する。
- sessionはtoken positions、global KV、indexer K、compressionの未完了group、bounded SWA / replay state、Engram n-gram stateを所有する。共有sourceもsessionに帰属させる。
- execution planはprefill / decode / replayで必要なproducer、consumer、stream、評価・同期境界を明示する。
- temporary buffersは用途・上限・最終consumerを持つ。非同期GPU workが完了するまでarray、Metal buffer、mmap、Engram pageを解放・再利用しない。
- candidate / Top-Kは対応するtoken範囲とgenerationを持つ。前回forwardや破棄済みspeculationのstateを再利用しない。
- generationはtokenをcommitする境界を持つ。将来のDSpark rollbackはKVだけでなくindexer、SWA、compression、Engram、RNGを含む全stateを扱う。

MLX graph構築時間だけを実行時間として扱わない。GPU完了までのwall timeと、tokenを利用可能にする境界で評価する。長いsessionに比例して過去graphを保持する設計は許可しない。

## Repositoryの最終配置

以下は予定構造。最初のコミットでは5つの文書だけを作る。

```text
deepseek-v41-flash-mlx/
├── CMakeLists.txt / README.md / LICENSE
├── docs/
│   ├── architecture.md / correctness.md
│   └── memory-layout.md / qualification.md / references.md
├── include/dsv41/
├── src/
│   ├── model/       config.cpp weights.cpp encoder.cpp decoder.cpp model.cpp
│   ├── attention/   csa2.cpp indexer.cpp swa_replay.cpp
│   ├── moe/ mhc/ dspark/ vision/
│   ├── cache/       kv_cache.cpp prefix_cache.cpp
│   ├── engram/      index.cpp store.cpp page_cache.cpp prefetch.cpp
│   ├── runtime/     session.cpp scheduler.cpp generation.cpp
│   └── server/
├── metal/           attention/ moe/ mhc/ engram/ dspark/
├── tools/           inspect_checkpoint/ convert/ benchmark/
├── tests/           reference/ exactness/ kernels/ long_context/ qualification/
└── third_party/
```

`metal/`はCMake、ABI、buffer lifetime、reference比較を伴う一級の実装領域とする。C++完成後の追加物として扱わない。`prefix_cache.cpp`は後段の機能であり初期実装の依存にしない。project LICENSEとthird-party noticesはコード導入時に整理する。checkpointのlicenseを本projectへ自動適用しない。

## Referenceの優先順位と固定

| 対象 | Reference | M0の状態 |
| --- | --- | --- |
| weight / config / tokenizer | [DeepSeek公式checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 資料を取得済み、全shard未検証 |
| モデル意味論・数値演算 | 同checkpointの`inference/model.py`, `kernel.py`, `engram.py`, `vision.py`等 | 読み取り確認済み、oracle未実行 |
| prompt / tool / reasoning / image protocol | 同checkpointの`encoding/`、[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) | Python reference取得済み、Rust revision未固定 |
| production構造 | [vLLM upstream](https://github.com/vllm-project/vllm)のV4.1実装 | 対応commit / path未確認。M3のCED / replay実装前に固定する |
| native backend | [MLX](https://github.com/ml-explore/mlx) | C++ API確認済み、依存revision未固定 |

公式資料はローカルcheckpoint内を読んだ。WebではこのV4.1 snapshotの本文とvLLM対応実装を確認できなかったため、未確認のproduction挙動を事実として補完しない。公式minimal inferenceはplain autoregressive generationで、DSpark forwardを含むが完成したspeculative serving engineではない。SWA ringと量子化後のdequantized値を使うreference表現を、packed persistent layoutやBounded Replayの実装済み証拠にしない。

取得済みREADME、config、modelのHugging Face download metadataが示すrevisionは`fb2764a5cf321eaa5070ca8f9e892818f477c16d`。これは全checkpointのintegrity証明ではない。M0で読んだファイルのSHA-256を以下に固定する。パスはcheckpoint rootからの相対パス。

| Path | SHA-256 |
| --- | --- |
| `README.md` | `94a04133ea0a65490881780180a8e671d674eec1dfa91e609f906a38fdcbf55e` |
| `config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `inference/config.json` | `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809` |
| `inference/README.md` | `2834402823199ee24e9a42bdf36a0fc6daf94448f444cb062c042a057a798f1c` |
| `inference/model.py` | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `inference/kernel.py` | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| `inference/engram.py` | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `inference/vision.py` | `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c` |
| `inference/convert.py` | `035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76` |
| `inference/generate.py` | `8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0` |
| `inference/image_processor.py` | `482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272` |
| `encoding/encoding.py` | `f64a67e5680a5621b9320585a9684967cd5c75a9b82e19d914cbc02845d72cab` |
| `encoding/test_encoding.py` | `854aa257e861b93c4f9727eb8bad6d801d449c04d2d708735d9088eadce698f5` |

M1でtokenizer、index、全shard、付随資料のrevision / digest manifestへ拡張する。M2でoracle環境を固定する。将来`docs/references.md`へ移す際もこのprovenanceを保持する。資料間に不一致があれば記録して解消するまで該当pathを未qualifiedとする。

## GLMからの知見inventory

参照先は`/Volumes/SDXC-512/glm53-flash-mlx`、確認時HEADは`f516f6b882cb0f09631640f936b582c38ec56da7`。以下は実在するファイル名に基づく調査候補で、コードの採用・移植・内容の再検証は行っていない。

| 候補ファイル（`scripts/`以下） | このprojectで再検討する知見 |
| --- | --- |
| `probe_resident_tensor_ownership.py` | resident tensorと非同期consumerのlifetime検証 |
| `probe_metal_input_layout_abi.py` | Metal inputのshape / stride / dtype契約 |
| `probe_semantic_branch_isolation.py` | speculative stateの分離と破棄 |
| `attribute_steady_decode_gpu_idle.py` | full-path wall timeとGPU idleの帰属 |
| `soak_cumulative_hybrid_allocation_1m.py` | 長時間sessionのallocation増加検出 |
| `qualify_native_coding_agent_http.py` | coding-agent APIを含むqualification設計 |

GLM固有のtensor配置、attention、cache形式、kernel、Python orchestrationは持ち込まない。上記の観点をDeepSeekの意味論に基づいて新規設計する。
