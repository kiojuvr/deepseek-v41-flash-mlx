# oMLX対照で進めるnative統合順序

2026-09-13。[改訂方針](omlx-baseline.md)を具体的な接続順序に落とす。oMLXの動作・性能はユーザー確認済み。本runtimeのfull pathは未完成で、C++化による優位は今後実証する。

## 現在地

| 境界 | workspaceにある実装・証拠 | 残る条件 |
| --- | --- | --- |
| checkpoint | 全48 shard・88ファイルのM1 attestation、native catalog | load peak、runtime residencyの実測 |
| Engram row lookup | mmap / pread、hash、BF16復元、cache比較6 run | full-path pressure / SSD miss / 実session |
| Engram forward | activation、projection、gate、residual、tokenwise比較 | CUDA projectionとの独立比較、公式CPUとの差の判定根拠 |
| Engram gate対照 | 固定oMLXとnativeのdot / gateがbitwise一致し、CPUとの同じ差を再現 | full-model精度・性能の証明ではない |
| text入口 | token ID → BF16 embedding → layer 0、ユーザーrunでchunk/token bits・継続・fork/reset・不正token拒否が通過 | 公式oracle比較、layer 2以降への接続 |
| RMSNorm | native primitive、実weightを用いた比較 | CPU oracleとのBF16 1 ULP差が5要素。未qualified |
| 共通linear / mHC | fixed-schedule FP8/FP4 reference、mHC mix / collapse / 展開、layer 0 Block接続 | mHC係数のCPU差 |
| SWA | layer 0のQ/KV → bounded state → masked attention → inverse RoPE → wo_a / wo_b、Block接続 | 公式oracle / padding / reduction比較、Bounded Replay |
| layer 0 MoE / Block | 全384 expert resident、ユーザーrunでhidden / pre-mix / KVのchunk-token bit一致、reset / position拒否 | 公式oracle比較、実token入口・次layerへの接続 |
| encoder / decoder / logits | 未接続 | 以下の統合作業 |
| layer 0 → Engram 1 → layer 1 | ユーザーrunでchunk/token bits・hash継続・fork/reset・不正入力拒否を確認 | 公式oracle比較、layer 2以降への接続 |
| layer 2 global KV producer | layer 2 Attentionへcompressor・FP4 cache・index query・SWA/global計算を接続、短いlocal検査通過 | FP4 / scoreの独立oracle比較、shared cache接続 |
| layer 2 → layer 3〜7 publication | cache / candidates / query positionを固定し、consumer検査とsnapshot保持を短い検査で確認 | layer 3 consumer Attention接続済み。layer 4〜7とrequest ownershipは未完了 |
| token → layer 2 Block | ユーザーrunでhidden・pre-mix・3層stateのchunk bits、global group境界・継続・fork/resetを確認 | 公式oracle比較、layer 3以降へのshared cache接続 |
| encoder 0..19 text経路 | producer 2/8/14、reuse 3–7/9–13/15–19、Engram 1/14を接続し、ユーザーrunでchunk/token bits・継続・fork/reset・不正token拒否を確認 | 公式oracle比較、decoderへの接続 |
| decoder 20..39 / head | ratio-1 producer 20、candidate二段Top-K、index source 24/28/32/36、reuse 21..39、final collapse/norm/headを接続し、ユーザーrunでchunk/token bit・継続・fork/resetを確認 | 公式logits oracle比較 |
| token → logits full backbone | encoder 0..19 → decoder 20..39 → logitsを接続し、ユーザーrunでchunk/token bit・継続・fork/reset・不正token拒否を確認（routing tie 3件は最小ID break、未qualified） | 公式logits oracle比較、sampling / generation |
| logits oracle比較 | native / 固定oMLX traceと公式式CPU転記で最初の分岐を `encoder.layer0.attn_in` に特定。RMSNormの分散reduction差、Q projection差、Attention算術差を局所化。layer 0 gateはscore差≈2e-6でもID一致、MoE native replayはbit一致。独立CPU FP4 MoEは152 expertsを通し、差270,799/307,200要素を検出 | M2のlogits差基準を広いteacher-forced入力で固定、expert projection / SwiGLU / shared / sum component trace |
| sampling / generation | greedy / temperatureの参照samplingと生成loop、RNG再現性の高速検査 | 公式RNGとのtoken列一致、stop sequence、API接続 |

既存の数値差を未解決として記録しながら、独立したfull-path接続を進める。CPUとMLXの全bit一致を各primitiveの実装着手条件にせず、公式oracle → local referenceの判定をM2 exit条件として保持する。local reference → optimized pathのbitwise条件は維持する。

## 統合進捗と共通linearの契約

進捗追記: [共通packed linear](packed-linear.md)を実装し、実FP8 / FP4の3行列でfixed-schedule referenceの短い比較が通過した。一括QMMには1 ULP差があるため診断用に保持する。[mHC](mhc.md)と[layer 0 SWA attention](swa-reference.md)を追加した。[layer 0 MoE / Block](moe-block.md)も接続し、ユーザーrunのlocal整合性検査を確認した。実token入口からlayer 3まで、Engram、layer 2のcompressor / indexer / FP4 global KV、layer 3のreuseを接続し、4層のユーザーrunを確認した。layer 3〜7のconsumer ownerを一般化し、追加4層のattentionを短く検査した。さらにproducerをlayer 2 / 8 / 14、reuse consumerを3–7 / 9–13 / 15–19へ一般化し、[encoder 0..19のtext reference](text-encoder-validation.md)を接続、ユーザーrunのlocal整合性検査を確認した。decoder 20..39（ratio-1 producer、candidate二段Top-K、index source 24/28/32/36）とfull backbone token→logitsも接続し、ユーザーrunのlocal整合性検査を確認した（[decoder / backbone検証](text-decoder-validation.md)）。routed expertはon-demand loadにし、full backboneのmemoryを抑えた。次は公式logits oracle比較とsampling / generationへ進む。attention・MoE・routing tieの公式oracle比較も未完了gateとして保持する。以下の設計上の条件は引き続き適用する。

Engram専用の固定shape projectionから、attention / MoEにも使うFP8・FP4 linearの契約を切り出す。canonical storageとMLX QMMのlayoutを区別し、activation量子化、weight / scale変換、積和とcastをそれぞれ比較可能にする。

固定oMLXの `convert.py:repack_weight` はE4M3 / E2M1 bytesをuint32へlosslessにpackし、FP8の32×32 block scaleを出力rowに展開する。`quantization.py:QuantizedProjection` は公式activation量子化を経てMLX `quantized_matmul`へ渡す。共通linearはこの経路を第一対照とする。汎用fallbackの算術まで手書きで再開発することを前提にしない。

ただし、layout変換がlosslessでもGEMMのreduction順序まで同じとは限らない。現Engram serial projectionは比較用として保持し、QMM経路の差を計測せず置き換えない。新しい経路は数値比較が済むまで実験referenceとし、optimized candidateへの昇格に使わない。oMLXが対応するaffine / unofficial量子化形式は本baselineへ持ち込まない。

この作業の完了条件は、実checkpointのFP8 / FP4 weightを使い、packed bytes・scales・activationとprojection出力を再現できること。単体throughputは判定目標にせず、直後のBlock接続へ渡せるAPIとownershipを成果物にする。

## 接続順序と比較境界

以下のoMLX pathは `omlx/patches/deepseek_v41/` からの相対path。固定commitとhashは[source manifest](../artifacts/references/omlx-v0.7.0.dev2.json)に記録する。`QuantizedProjection`の継承先である `deepseek_v4/switch_layers.py` も対照範囲に含む。

| 順序 | native接続 | 主なoMLX対照 | 完了を示す境界 |
| --- | --- | --- | --- |
| 1 | 共通FP8 / FP4 linear | `convert.py`, `quantization.py`, `activation.py` | canonical bytes / scale、量子化activation、projection出力 |
| 2 | mHCとBlockの呼び出し順序 | `language.py:hc_mixes / hc_pre / hc_post / Block`, `hyper_connection.py` | incoming pre-mix、attention後hidden、FFN後hidden、次pre-mix |
| 3 | attention / indexer / compressorとrequest state | `language.py:Attention / Indexer / Compressor`, `cache.py`, `packed_attention.py`, `kernels.py` | producer / reuse source、position、packed KV、候補・Top-K、継続state |
| 4 | routed / shared MoEをBlockへ接続 | `language.py:Gate / MoE / Expert`, `routing.py`、V4.1 grouped-expert native code | expert IDs、routing weights、shared / routed合成、Block出力 |
| 5 | 全20 encoder + 20 decoder、final collapse / norm / head | `language.py:LanguageModel`, `head.py` | token→全layer境界→logits、短いteacher-forced continuation |
| 6 | native generationと比較harness | `language.py`, `loading.py`, `admin/benchmark.py` | 同一入力のprefill・decode・state・wall、最初の差のdump |

mHCでは新しく計算したattention pre-mixを、そのattention自身の入力に使わない。固定oMLXのBlockではincoming preでattention入力をcollapseし、attentionから得たpreでFFN入力をcollapseし、FFN側preを次へ返す。実装時は公式Blockとも再照合する。

順序2だけで偽のattention / MoE出力を返すBlockをモデルreferenceと呼ばない。接続境界を用意した後、3・4を同じBlockへ接続してから全層へ展開する。各段階の短い診断は許容するが、専用probeや単体kernelの最適化を次の独立目標として増殖させない。

## 比較へ進む条件

まずtext plain ARの機能・tokenized入力を揃える。次に報告済み32K / 64K / 128K / 200Kで、prefill / decode / TTFTの測定境界を一致させる。256Kは両runtimeの新規測定が必要。200K集約decodeの36.9 tok/sを、256K後半のp95 / p99 TPT上限へ変換しない。

報告された292.96 GBをruntime全体のphysical memory上限に採用しない。raw bytes、MLX allocation、process RSS / footprint、OS file cache、swapを分け、DSpark / vision、Engram residency、APC、custom kernel dispatch、正確な入出力token数とともに記録する。

その比較harnessが動作する段階で、長時間検証の再開・ログ保存を備えたスクリプトを渡す。今回の計画確認ではモデルloadや長時間benchmarkを起動していない。API / DSpark / visionと最終256K gateは従来どおり残す。

layer 3 consumer Attentionは実weightでsnapshot再生・reset/forkの短い検査が通過。layer 3 mHC / MoE / Blockとtoken逐次の4層text経路も接続し、[4層ユーザー検査](text-quad-validation.md)をレビューし、出力・stateのchunk bits、global group境界、継続・fork/reset・不正入力拒否の通過を確認した。公式oracle比較は未完了。
