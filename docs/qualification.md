# Qualification contract

状態: M1の[checkpoint atlas / integrity検証](checkpoint-atlas.md)が完了。本runtimeのnumerical / performance qualificationは未達。2026-09-13にユーザー報告のomlx v0.7.0.dev2を外部性能baselineとして採用した。同条件の比較runは未実行。[改訂計画](omlx-baseline.md)を適用する。

2026-09-14時点で、修正版native pathはencoder 0..19、decoder 20..39、Engram、final collapse/norm/headを接続し、実token IDからlogitsまでの継続・fork/reset・不正token契約を通過した。layer-0 MoEの独立CPU比較は、expert出力のBF16 cast境界とexpert ID昇順加算を合わせた後、59要素の局所差（最大 `0.0009765625`）まで縮小している。M2は公式CUDAとのbit一致ではなく、[MLX reference着地点](correctness.md)に定義したcanonical referenceとnative pathの同一backend exactnessを判定する。長文qualification、API接続、公式実装との差分資料はM3以降の独立gateとして保持する。

## Milestones

| Phase | 成果物とexit gate |
| --- | --- |
| M0 — Repository contract | 5文書でarchitecture / correctness / memory / reference / performance契約と禁止事項を固定。GLMは知見inventoryのみ |
| M1 — Checkpoint atlas | 全tensorの所在・形式・owner・bytesを確定。Engram backingを分離し、512 GB予算を作る。missing / unknownと総数の不一致を解消 |
| M2 — End-to-end MLX reference | C++ / MLX / Metalでtoken→encoder→decoder→logits。Engramを含むtext path、generation、state契約をcanonical MLX referenceとnative pathで一致させる。公式CPU/oMLX/CUDA比較は別証拠として保存し、M2のbackend exactnessへ混ぜない |
| M3 — Native execution graph | Python往復のないprefill / decode / replayを形成。固定omlxを第一reference、vLLMを補助として、CED scheduleとpacked KVのexactnessを検証。ここから性能最適化 |
| M4 — 32K→256K | 32K correctnessから64K / 128K / 200K比較点 / 256Kへ進み、長時間session後半のdecode TPTをhard gateとして測定 |
| M5 — Engram storage engine | OS page-cache基準のmmap / pread、working-set telemetryとpressure検証。独自cache / async prefetchは必要性が示された場合のみ。全state exactnessとM4再実行 |
| M6 — DSpark / API / Vision / RELEASE | native DSpark、Rust / Axum + deepseek-recipe API、C++ bridge、tool call、reasoning effort、streaming、image input。最終構成で256K qualificationを再実行してrelease |

M0のreference方針とrecipeの採用候補revisionは[architecture](architecture.md)に固定した。omlx実行環境の依存・設定の固定、補助vLLM / native MLXのrevision選定、recipeのbuild / 接続検証と依存lockは後続phaseのexit条件。未確認referenceを利用済みと表示しない。M4の一度のpassをM5 / M6変更後へ無条件に持ち越さない。

## Context ladderとworkload

K＝1,024 tokens。32K / 64K / 128K / 256Kはpromptと生成済みtokenを合わせた総context長で、generation budgetではない。API admissionはprompt tokens + requested output reserveがqualified上限以内か確認する。公式推奨の1M context / 256K以上のmax_tokensを未検証のserver設定として広告しない。

各段階で同一checkpoint / tokenizer、1 active session、固定tokenized corpusを用いる。coding-agent workloadには複数turn、code、長いtool output、追記prefillとdecodeの反復を含める。fixture hash、turn schedule、seed、prompt長、output長を事前固定する。

各context上限Lについて、少なくとも次の3種類を実行する。

1. L−4,096 tokensまでprefillし、teacher-forced continuation 4,096 tokensを実行してLへ到達する。後半2,048 tokensのlatencyとexactnessを記録する。
2. 同じ開始contextからgreedy生成を行い、実際のtoken / logitsを比較する。EOS等で測定長を満たさない場合は明記し、短いrunを長時間passの代わりにしない。
3. 小さいcontextからtool turnとgenerationを累積してLへ達するsessionを実行する。各段階のstate、追記prefill、first decode、tail TPT、reset後のmemory回収を確認する。

大きなpromptを一度だけloadする試験では累積sessionを代替できない。Engram cold / warm両条件で同じcorpusを使い、異なるlookup localityも含める。SWA / compression境界、candidate容量境界、cancel / resumeは[correctness](correctness.md)の対象とする。

## 測定境界

| Metric | 定義 |
| --- | --- |
| load / warm-up | checkpoint読込・検証・resident化・compileを別々に計測 |
| cold prefill wall | fresh session、Engram cache状態を記録し、GPU完了までを計測。OS page cacheがcoldか不明ならunknownとする |
| warm prefill wall | weights / kernels / Engramのwarming条件を固定したfresh sessionで計測。APCなし |
| TTFT | request受付から最初のoutput token利用可能まで。M6ではAPI経路も測定 |
| first decode latency | prefill→decode切替の最初のstepを独立に記録 |
| decode TPT | 各committed output tokenを利用可能にするまでのwall latency、ms/token。値が小さいほどよい |
| tail decode TPT | 上限付近の固定測定窓におけるmean / p50 / p95 / p99 / maxとraw時系列 |
| full session wall | 追記prefill、decode、replay、Engram stall、sampling、同期を含む総時間。外部tool実行待ちは別計上 |
| memory / I/O | physical footprint、peak allocation、graph / cache容量、swap増加、page faults、Engram hit / miss / bytes / latency |

TPTにCPU dispatch、GPU execution、必要な同期、Engram I/O待ち、sampling、state commitを含める。token/sやkernel GPU timeは補助指標。DSpark有効時はdraft token数で割らず、accepted / committed token数を使い、block emissionによる待ち時間も記録する。

M4ではnative generation boundary、M6ではそれに加えてloopback APIのtoken event境界を計測する。streamingのchunking / bufferingも報告し、全tokenを最後に返す方式でtail latencyを隠さない。

## Performance hard gates

correctnessの失敗は性能評価に優先する。全candidateは同じfull workloadでbaselineと比較する。測定環境にはhardware、OS、compiler、MLX / Metal revision、build flags、checkpoint、power / thermal条件、Engram cache予算を記録する。モデルdownload等の競合I/Oがあるrunはqualificationに使用しない。

- cold条件は最低3回、warmのbaseline / candidate比較は実行順を交互にして最低5組を記録する。除外runには理由を残す。
- promotion対象のfull-path wallまたはtail TPTの改善がrun間変動を超えることを要求し、他の主要指標にregressionがないことも確認する。microbenchmarkだけの改善、平均でtail悪化を隠す変更はrejectする。
- 256K session後半のTPTは必須gate。M3終了時にM4実行前の独立した契約変更で、絶対的なmean / p95 / p99上限、許容drift、memory / I/O上限、統計的判定法を数値で固定する。candidateの結果を見てpassになる値へ調整しない。
- 外部baselineの32K–200K報告値は[omlx baseline](omlx-baseline.md)に固定した。これは256Kのtail分布・絶対上限を与えないため、未測定の値を外挿しない。上記閾値が未設定のままM4やRELEASEをpassにすることは禁止する。
- bounded stateの容量増加、古いgraph保持、unexpected full-model copy、継続的なswap増加、I/O failure、NaN / Inf、state不一致は失敗とする。予算内の意図したcache成長は内訳と上限を示す。

late-session driftは同等のcontext長・cache条件のfresh runとも比較し、contextの伸長に伴う費用とsession経過に伴う劣化を区別する。稀なstallを集約値だけで消さず、raw traceに残す。

## omlxに対する性能昇格

内部optimizationは従来のlocal baseline gateを満たすこと。加えて、専用native runtimeとしての性能優位を主張するには、固定omlxとの同一入力・機能・cache・測定境界の比較でfull-path wallまたはtail TPTの改善がrun間変動を超え、他の主要指標にregressionがないことを必須とする。内部baselineへの改善だけでomlx超えとはしない。32K / 64K / 128K / 200Kの比較結果を保存し、256Kは両runtimeを新規測定する。omlxが256Kで未実行・失敗の場合もnativeの256K gateは維持し、比較不能を速度勝利に数えない。

## Release evidence

各runはruntime commit、全reference identity、build / environment、入力・schedule・seed、enabled features、raw timing、state比較、memory / I/O、判定閾値、結果を保存する。失敗・未実行・unsupportedはpassと別に表示する。

M6ではtext-onlyとimage入力、DSpark on / off、tool call、数値reasoning effort、streaming / non-streaming、cancel / resumeを検証する。protocolは公式encoding fixtureと照合し、OpenCodeから実際にAPIを通す。APIや画像処理を加えた最終構成で256K gateを満たすまでRELEASEと呼ばない。APCを後から追加する際もstate再利用のexactnessとfull-path性能を再qualificationする。

recipe採用に伴い、Rust toolchain / Cargo.lock / recipe revisionとnative bridge ABIをrun identityへ含める。mock serverの起動はAPI接続成功に数えない。実token IDsの増分decode、JSON / SSEの整合性、未対応option、stop sequence、切断・cancel・backpressure・backend failureを検証し、native committed token数とAPI usageを別計数する。requestからSSEまでのTTFT / tail latencyとnativeのみのTPTを併記する。

## 長時間検証の運用

SWA optimized reductionのbit-exact referenceからの分離条件、observable semantics、persistent state、
離散判断の事前固定gateは[swa non-bitwise gate](swa-nonbitwise-gate.md)に定義する。まずcomponent profileで
attentionが2063-token wallの最大項かつ40%以上かを確認し、その後に短いsemantic propagation probeを行う。
結果を見る前にgateを緩めず、いずれも単独ではM2 / performance / 32K qualificationに数えない。

component profile `context-ladder/32k-run-20260916-233434-29168`はexit 0、40層×17 chunkのcall数と
identityが一致した。prefill 225.927秒に対し、attention / MoE / post-MoEは71.066 / 106.119 /
38.743秒、未分類layer overheadは9.678秒。attentionはcomponent合計の32.9%で最大項でも40%以上でも
ないため、SWA Metal進行条件はfail。MoE pathを次の最適化対象とし、profile値を通常performanceや32Kへ
昇格しない。

### Layer-major chunk integration (2026-09-16)

単独layer-major経路の最初の性能診断は次で実行する。

```sh
bash tools/benchmark/run_layer_major_prefill_measurement.sh
```

1モデルのみで256-token prefill（2×128）と最初のsample 1 tokenを実行する。
各chunkのbank構築・attention・MoE・同期・出力finite検査を時間へ含め、chunkごとの時間と
MLX memoryをprogress JSONLへ保存する。decode forwardは実行しない。
数分、メモリ予算80 GB、約578 GBの論理checkpoint readを見込む。wall予算600秒はchunk境界で
検査するため厳密な強制終了時刻ではない。結果は既存runnerの
`artifacts/context-ladder/32k-run-日時-PID/`内のresult.json、result.json.progress.jsonl、resource.log。
directory名に32kとあるが実際のcontextは257。checkpointはread-only。途中resume不可で、
失敗ログを保持して同じscriptをfresh runする。
`32k-run-20260916-054037-12182`をreview済み。exit 0、記録identityすべて一致。
256-token prefillは129.679377秒、1.97410 token/s。最大RSS 22,721,495,040 bytes、
peak footprint 24,637,482,880 bytes、prefill終了MLX active 11,388,604,388 bytes。
OS compression増分0、decompression増分9、swap増分0。
毎128 tokenで全bankを再構築する方式は速度最適化として採用しない。
bank構築の時間内訳は未計測であり、以前の2K individual runとの同条件比較でもない。
比較検査のwallから得られなかった単独経路の費用を確認する段階で、32K qualificationではない。
この結果とoMLX実装の再確認を受け、M3は局所的なpacked-bank延長ではなく
[prefill dataflow redesign](prefill-dataflow.md)へ移行する。32K換算ではbank再構築だけで約73.9 TBの
論理checkpoint readになるため、同経路の追加長時間測定は停止する。次のperformance gateは
resident atlas、device batched routing、chunk-wide packed attentionを統合した後の短いpaired比較とする。

decoder 20..39とfull backboneにも明示的な`forward_packed_chunk`を追加した。
layer 20のtoken別publicationを21..39へ渡し、24/28/32/36のindex再publicationを後続層へ引き継ぐ。
各bankは成功・例外の両方で解放し、request stateは全層成功時にcommitする。

```sh
bash tools/benchmark/run_layer_major_backbone_check.sh
```

2×128 token・40層のhidden/pre-mix、全token logits、state、publication、hash継続、順序を揃えた
route ties、不正token拒否をindividual経路と比較する。decoderの最終index sourceは36を要求する。
数分、メモリ予算240 GB、80 bank構築で約578 GBの論理checkpoint readを見込む。
ログは`artifacts/expert-bank/layer-major-backbone-日時-PID/`。checkpointはread-only。
途中resumeはなく失敗ログを保持してfresh runする。
`layer-major-backbone-20260916-053417-11816`をreview済み。exit 0、記録されたidentityはすべて一致。
40層2×128 tokenのhidden/pre-mix・全token logits・state・publication・hash・route tiesと
不正token拒否はpass。wall 216.95秒、最大RSS 161,230,979,072 bytes、peak footprint
164,634,260,512 bytes、MLX active 150,968,218,264 / cache 10,439,160,666 /
peak 158,313,268,932 bytes。OS compression増分0、decompression増分10、swap増分0。
この入力範囲のfull-backbone chunk correctnessをpassとする。比較用両モデルとbank構築を
含むため、このwallやmemoryから単独runtime性能は算出しない。
256 tokenでは512行超Top-Kを通らず、速度・32K・APIのqualificationも含まない。

encoder 0..19の`forward_packed_chunk`を追加。pure SWA 0/1、Engram 1/14、producer 2/8/14と
各5 reuse層をlayer順で処理し、各bankは出力評価後に解放する。失敗時にも現在層のbankを解放し、
request stateは最終成功時にcommitする。2×128 tokenのindividual比較runnerは次。

```sh
bash tools/benchmark/run_encoder_packed_chunk_check.sh
```

hidden/pre-mix、SWA/global/index/compressor state、publication、hash継続、token/layer順に揃えた
route tie records、不正token拒否を検査する。数分、メモリ予算160 GB、40 bank構築で約289 GBの
論理checkpoint readを見込む。ログとresourceは`artifacts/expert-bank/encoder-chunk-日時-PID/`。
checkpointはread-only。途中resumeはなく失敗ログを保持してfresh runする。
`encoder-chunk-20260916-052648-11499`はreview済み。exit 0、記録されたidentityはすべて一致。
2×128 tokenのhidden/pre-mix・state・publication・hash継続・route ties・不正token拒否は一致。
wall 110.11秒、最大RSS 75,545,018,368 bytes、peak footprint 76,079,383,656 bytes、
MLX active 65,340,430,900 / cache 7,616,481,606 / peak 72,675,334,408 bytes。
OS compression増分0、decompression増分1、swap増分0。20層のこの入力範囲のcorrectnessをpassとする。
両モデルとbank構築・比較を含むため、単独runtime速度・memoryへの換算はしない。
decoder index再publication、512行超Top-K、
40層全体・性能・32Kのqualificationは含まない。

pure SWA / compressed producer / reuse Blockへpacked MoE chunk APIを追加した。
attentionはtoken順で更新し、MoEを最大128 tokenでまとめる。producerのpublicationはtoken別に保持し、
consumerへ同じ位置のsnapshotを渡す。publication出力は計算成功時にcommitする。
layers 0..7の4-token probeでは出力と検査対象のattention stateがbit一致、MLX peakは
16,516,688,260 bytesだった。この初版probeには全state比較と継続検査が不足していたため、
全Global KV、pending compressor、layer 3、hash継続、publication、2 chunk継続、invalid-token検査を追加した。

```sh
bash tools/benchmark/run_octet_packed_chunk_check.sh
```

2×128 token・8層を比較する。数分、メモリ予算80 GB、bank構築に約116 GBの論理checkpoint readを
見込む。結果は`artifacts/expert-bank/octet-chunk-日時-PID/`へ保存。途中stateのresumeはなく、
失敗ログを保持して同じcommandでfresh runする。
`octet-chunk-20260916-052149-11016`をreview済み。exit 0、記録されたbinary / source / fixture
identityはすべて一致。2×128 tokenの出力・検査対象全state・publication・hash継続・不正token拒否が
一致した。wall 44.18秒、最大RSS 41,834,348,544 bytes、peak footprint 42,306,678,216 bytes、
MLX active 31,849,248,324 / cache 7,447,436,985 / peak 39,171,073,528 bytes。
OS compressions / decompressions / swapの増分はいずれも0。既存の圧縮済みページは残っているため、
compressor総量0という意味ではない。8層・この入力範囲のcorrectnessをpassとする。
比較用両モデルとbank構築を含むwall / memoryから単独runtime性能は算出しない。
512 compressed rowを超えるTop-Kとdecoder index-source再publicationはこの検査に含まれない。
bank解放後のcache増加だけでは、次層が同一allocationを再利用した証明にはならない。
また128 tokenごとに全bankを再構築する費用は速度評価に含める必要がある。
40層統合・性能・32Kのqualificationは未達。

ユーザー指定に従い、数分以上かかる見込みの検証はユーザー実行用スクリプトとして渡す。実行手順、ログ / 結果path、想定負荷、再開方法を準備し、assistantが長時間監視し続ける形にしない。結果を回収するまでは未実行 / 未判定として保持し、その間は独立した実装・短いcorrectness検査を進める。

producer recurrent chunkとbatched index scoreの局所gateは次で実行する。

```sh
cmake --build build-mlx -j2 --target dsv41-producer-chunk-test
build-mlx/dsv41-producer-chunk-test \
  /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  artifacts/checkpoint/summary.json
```

2026-09-16にexit 0を確認。512-row継続stateから128-tokenを進め、ratio 2 / ratio 1 compressor、
全token cache prefix、main / index packed bytes、pending state、index key、GPU batched score / causal mask、
80 block中64 blockのlayer 20 candidate source、layer 24 candidate consumer、640-row中512-row Top-K、
全score同値時の最低ID tie、不正入力atomicityはtoken-serial CPU referenceとbit一致した。実行部
1.43秒、最大RSS 150,683,648 bytes、peak footprint 566,838,088 bytes。同じ変更を通る8-token・
layers 0..2 probeもexit 0でhidden / pre-mix / state / publicationがbit一致した。40層full path、
long-context、performance qualificationには数えない。

device index selection変更後の40層2×128 promotionは次の既存runnerで行う。数分、Unified Memory
予算100 GB、実route unionに応じたcheckpoint read（上限578 GB）。結果は
`artifacts/expert-bank/compact-layer-major-日時-PID/`へ保存する。checkpointはread-only、途中resumeは
なく、失敗directoryを保持してfresh rerunする。このfresh結果をreviewするまで40層gateは未判定。

```sh
bash tools/benchmark/run_compact_layer_major_check.sh
```

fresh run `expert-bank/compact-layer-major-20260916-222621-27115`をreview済み。exit 0、binary / source /
fixture identityは現worktreeと一致した。両128-token chunkのhidden / pre-mix / logits / 全state /
publication / hash / route tiesとinvalid-token atomicityはindividual referenceとbit一致。bank constructions
80、loaded experts 10,543。wall 121.74秒、MLX active 150,972,764,108、cache 156,216,429,644、
peak 155,303,213,832 bytes、最大RSS 304,053,288,960、peak footprint 308,094,288,840 bytesだった。
VM compression増分0、decompression 5 pages、swap 0。この入力範囲の40層correctness gateをpassとする。
比較用二重modelと80 bank構築を含むため、旧135.45秒との差やmemoryをproduction performance / footprintへ
換算しない。32K、外部parity、performanceは未qualified。

device index selection後のcompact-only再計測は次で行う。

```sh
bash tools/benchmark/run_device_index_prefill_measurement.sh
```

1 model、layer-major、2063-token prefill + 1 decode、4 I/O worker、assignment chunk 128、MLX cache
limit 8 GiBを固定する。約5分、Unified Memory 30 GB未満を目標（予算100 GB）とし、checkpointはread-only、SSD readを
伴う。結果は`artifacts/context-ladder/32k-run-日時-PID/`へ保存。resume不可、失敗directoryを保持して
fresh rerunする。`DSV41_DRY_RUN=1`によるpreflight
`context-ladder/32k-run-20260916-223018-27478`はexit 0で、identityにcompressor / global KV / index key /
index queryが含まれ、I/O worker 4、assignment 128、layer-major条件を確認した。model測定は未実行。

実測 `context-ladder/32k-run-20260916-223119-27624`をreview済み。exit 0、identity / 固定条件は一致。
prefill 2063 tokenは226.851410秒、9.094059 token/s。変更前の同条件I/O telemetry run
`32k-run-20260916-153052-20781`の243.880444秒、8.459063 token/sに対し、wall 6.98%短縮、throughput
7.51%上昇した。単発paired observationでありperformance acceptanceには使わない。bank constructions
680、loaded experts 70,736、route union統計、read calls 212,208、index ties 9,395、route ties 4は一致。
bank read / totalは40.263 / 42.239秒（旧41.018 / 42.998秒）なので、主要差はI/O量変更ではない。
active/cache/peakは11,394,303,192 / 8,345,273,386 / 14,906,983,306 bytes、最大RSS
96,693,927,936、peak footprint 97,823,275,608 bytes、compression / swap増分0。

ただし最初の128-token chunkはcache 85,801,361,337 bytesを記録し、8 GiB cache limitがchunk終了後に
初めて適用されていた。最終値だけでcache policyをpassにせず、runnerはmodel構築前にlimitを設定するよう
修正する。この測定はdevice index executionのbounded full-path観測として採用するが、cache lifecycle、
32K、外部performanceは未qualified。

修正後の短いfull-path check `context-ladder/32k-run-20260916-223749-27811`をreview済み。exit 0、
identity一致、128-token prefill + 1 decode、4 I/O worker、assignment 128。最初のchunk終了時cacheは
7,920,962,087 bytes、decode後8,561,084,853 bytesで8 GiB設定内、旧測定の最初のchunk
85,801,361,337 bytesのtransientを除去した。active 11,453,985,520、peak 14,906,999,690 bytes、
最大RSS 22,866,231,296、peak footprint 23,887,621,336 bytes。compression / decompression / swap増分0。
prefill 13.604407秒、9.408716 token/sだが、1 chunk値を2063-token performanceの代用にはしない。
cache-limit適用境界のbounded lifecycle gateをpassとする。

同じ修正を通るfull 2063-token再測 `context-ladder/32k-run-20260916-223941-28024`もreview済み。
exit 0、identity / 固定条件は一致し、全progress sampleのcacheは約7.38--7.99 GiB、prefill終了時
8,582,960,092 bytes、decode後8,432,915,464 bytesで8 GiB設定内だった。active / peakは
11,394,303,192 / 14,906,999,690 bytes、最大RSS 22,865,559,552、peak footprint
23,893,093,496 bytes、compression / decompression / swap増分0。修正前の最初のchunkだけ約79.9 GiBに
達したtransientはfull測定でも除去され、cache lifecycle gateをpassする。prefillは226.037111秒、
9.126820 token/sで、device-index初回測定226.851410秒からwall 0.36%短縮。同変更前の
243.880444秒、8.459063 token/sに対してwall 7.32%短縮、throughput 7.89%上昇した。bank
constructions 680、loaded experts 70,736、route union、read calls 212,208、tie countsは一致する。
これは単発のbounded full-path観測であり、反復paired performance、32K、外部parity、256Kを
qualifyしない。

### 32K first measurement

最初の32K実測runnerは総context 32768を、base prefill 28656、teacher-forced continuation
4096（後半2048をtailとして別計測）、greedy decode 16に分ける。128-token chunkでfull
backboneを一度だけ進め、各phase wall / tokens-per-second、first decode、全decode latency、
MLX active/cache/peak、OS max RSS / page fault、state position、finite logits、routing tieを保存する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_context_32k.sh
```

full 40-layer model、on-demand routed experts、32K persistent KV、Engram mmapを同居させるため、
数十分以上・数百GB規模のUnified Memoryと継続的なcheckpoint / Engram SSD I/Oを見込む。
他のmodel jobと同時実行しない。checkpointはread-only。結果は
`artifacts/context-ladder/32k-run-日時-PID/`へ、`result.json`、`resource.log`、前後`vm_stat`、
prompt / binary / source identity、command、diff、exit codeとともに保存する。model途中stateは
永続化しないためresume不可。失敗directoryを保持し、同じcommandでfresh runする。

このrunnerはcanonical referenceのcorrectness / scaling診断であり、M3 native execution graphの
performance測定ではない。各1024 tokenで区間token/s、phase平均、平均からのprefill予測時間、
MLX active / cache / peak bytes、tie数を`result.json.progress.jsonl`へflushする。既定では実wall
1800秒、予測prefill 3600秒を超えるrunを早期停止し、partial JSONLを保持する。制限を意図的に
解除する場合だけ次を指定する（解除してもperformance qualificationにはならない）。

```sh
DSV41_CONTEXT_WALL_BUDGET_SECONDS=0 \
DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0 \
bash tools/benchmark/run_context_32k.sh
```

MLX cacheを周期的にclearすることは測定条件を変え、active stateやper-token同期を解消しないため
このrunnerでは行わない。増加がcacheかactive allocationかをtelemetryで分離してから扱う。

最初のbounded optimizationとして32K runnerは`DSV41_RUNTIME_LAYER_FINITE_CHECKS=0`を使う。
これにより40 Blockと内部SWA / compressed / reused attention / compressorでtokenごとに行っていた
redundant finite reduction / host同期を省くが、
shape / state validation、quantization error、routing / index decision、128-token chunk境界のfinite検査は
維持する。通常のcanonical testは未指定（既定1）のため従来どおり全layer検査を行う。この変更は
同期削減候補であり、短いfull-model parityと区間telemetryをレビューするまで性能昇格しない。

layer finite check省略のfull-model bit parityは次で確認する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_layer_finite_policy_check.sh
```

固定3-tokenを同一processでcheck on / offの両方へ通し、hidden、pre-mix、全state、logits、route tie
recordsをbit比較する。full 40-layer checkpoint、on-demand experts、Engram mmapを使うため、数分と
高いUnified Memory / SSD負荷を見込む。結果は`artifacts/layer-finite-policy/run-日時-PID/`へ保存。
途中stateは永続化しないためresume不可で、失敗runを保持して同じcommandからfreshに再実行する。

`layer-finite-policy/run-20260916-001358-3544`をreview済み。exit 0、16.82秒、最大RSS
28188295168 bytes、peak footprint 31265664912 bytes、swap 0。check on / off間でhidden、pre-mix、
state、logits、route tie recordsがbit一致したため、このbounded optimizationのcorrectness gateは
passとする。外部oracle、performance、32K、256Kのqualificationには昇格しない。

MoE routingはcorrected scoreのCPU evalと別GPU finite reductionを直列同期していた。raw / corrected
scoreを1回のCPU evalへ統合し、同じhost passでfinite / nonnegativeを検査する。384件のfull sortも
同じscore降順・ID昇順を保つ固定top-7 insertionへ置換した。200 selection×交互9組のbounded
microbenchmarkでは、同期統合前49.8907 msから統合後4.26196 ms（median、約11.71倍）へ短縮。
単体のID / weight bits / strict / lowest-ID policyと、3-token full-backboneのhidden / pre-mix / state /
logits / tie records bit一致を確認した。1-thread Metal top-7は58.3517 msと遅かったためruntimeから
除外した。この値はrouting単体の候補判断であり、full-path performanceや32Kをqualifyしない。

index top-kも全履歴の完全sortを廃止し、選択512行と境界1行だけを`partial_sort`する。candidate
level-oneも全blockではなく採用block数だけを選択する。4,097行のindex selectionと32,768行・
4,096 blockのcandidate maskについて、従来のfull-sort定義とID / mask exact一致を短いtestで
確認した。これはhost計算量を抑える変更で、GPU→CPU score同期やfull-pathの費用はまだ残る。

同期削減後のbounded scaling確認は32Kを再開せず、prefill 2048 + decode 16だけを同じ測定器で
実行する。wall 900秒で停止し、結果のresumeは不可。次のexact commandを使う。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
CONTEXT_TOKENS=2064 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 \
DSV41_CONTEXT_WALL_BUDGET_SECONDS=900 \
DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0 \
bash tools/benchmark/run_context_32k.sh
```

full 40-layer、on-demand experts、Engram mmapを使い、最大15分、高いUnified Memory / SSD負荷を
見込む。結果は通常どおり`artifacts/context-ladder/32k-run-日時-PID/`へ保存する。これは旧runとの
概算速度・memory比較用であり、短いcontextによって32K / M3 performanceを代替しない。

`32k-run-20260916-003243-5447`をreview済み。repo canonicalのprefill 2048は355.8466秒、
5.7553 token/s。前半1024は5.4976 token/s、後半は6.0384 token/sで、この範囲では長文化による
低下はない。decodeはfirst 70.75 ms、以降を含むmean 172.43 ms、p50 176.52 ms、p95 / p99 /
max 206.36 ms。最大RSS 155772649472 bytes、peak footprint 159268265888 bytes、swap / compressor
0。prefill終了時のMLX active 153767494628 bytesに対してcacheは301734394 bytesだけなので、
高いfootprintはGC/cache反復でなく、routingで早期にresident化したmodel / expert dataが主体。
route tie 4、index tie 9395（先頭256以降は記録省略）で最低ID policyの外部oracle gapは残る。
報告済みoMLX 32K値との差が大きく、次はtoken逐次referenceの小修正でなく、packed attentionと
grouped expertをchunk実行するM3 execution boundaryを実装する。32Kは引き続き保留する。

M3 grouped expertの最初のprobeとしてlayer 0の6 expertだけを一時bankへstackし、MLX
`gather_qmm`でw1 / w3 / SwiGLU / w2をまとめた。最初のprobeはw2手前の2回目の公式activation
quantizationを欠いてdown bit mismatchとなり、同じ量子化を6行batchで追加後、gate / up /
activation / downとroute-weight適用後の全bitsが個別QMMと一致した。warm後の交互9組medianは
個別0.849666 ms、grouped 0.648000 ms（1.31121倍）。これは6 expert temporary bankだけのprobeで、384 expert bankの
構築・resident memory、command encoder fusion、shared expert、Block全体を含まない。次は1層分の
resident packed bankを作り、memoryとfull MoE出力をgateにする。

layer 0の全384 routed expertを`[expert, output, packed-input]`へ直接読み込む
`PackedExpertBank`を追加した。checkpoint tensorはread-onlyのまま順次copyし、個別
`ExpertReference`を同時に384個保持しない。選択6 expertのw1 / w3、SwiGLU、2回目の
activation quantization、w2、route weight適用を`gather_qmm`でbatch化し、公式と同じexpert
ID昇順でFP32加算する。full-modelへ接続する前のmemory / bit gateは次で実行する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_expert_bank_probe.sh
```

layer 0の7,219,445,760 bytes（約6.72 GiB）packed bankを構築し、選択した6 expertについて個別QMMとのgate / up /
activation / route-weighted down / routed sumのbit一致、交互9組のwarm timing、MLX active / cache /
peakとOS resourceを記録する。checkpoint SSD readと構築一時領域を伴い、数分と10 GiB超の
Unified Memoryを見込む。他のmodel jobと同時実行しない。結果は
`artifacts/expert-bank/run-日時-PID/`に保存する。途中bankは永続化しないためresume不可で、
失敗runを保持して同じcommandからfreshに再実行する。exit 0でも1層6-routeのprobeに限られ、
shared expert、gate、Block統合、40層memory、32K / 256K、performance qualificationを満たさない。

`expert-bank/run-20260916-005454-6366`をreview済み。exit 0、bank構築1.9446秒、packed
7,219,445,760 bytes、MLX active 7,332,744,728 bytes、cache 1,507,380 bytes、最大RSS
9,855,123,456 bytes、peak footprint 10,198,851,048 bytes、swap / compressor 0。選択6 expertの
intermediate / route-weighted output / routed sumはbit一致し、warm medianは個別1.069375 ms、
grouped 0.670375 ms（1.5952倍）。bounded bank gateはpassとするがfull MoE以降は未qualified。

この結果を受けて`DSV41_RUNTIME_PACKED_EXPERT_BANK=1`のopt-in lazy full-MoE経路へ接続した。
bankは各layerの初回MoE実行時に構築し、BF16 routed出力だけでなくshared expertとの最終加算に使う
round前FP32 accumulated valueも維持する。既定は0のままである。layer 0の短いintegration probe
`expert-bank/packed-moe-short-20260916-01`ではindividual / packed間のshared、routed、totalが
bit一致した。packed初回1.5307秒、MLX active 7,413,381,284 bytes、route tie 0。

40層full-backboneへの昇格gateは次で実行する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_packed_expert_backbone_check.sh
```

one-token individual / packed間のhidden、pre-mix、全state、logits、route tie recordsをbit比較し、
続けて既存3-token lifecycle検査を行う。40 bankだけで288,777,830,400 bytes、比較中はbackboneも
二重に保持するため、数分、概ね310--340 GBのUnified Memory、約289 GBのcheckpoint SSD readを
見込む。他のmodel jobと同時実行しない。結果は`artifacts/expert-bank/full-backbone-日時-PID/`へ
保存する。model / bank途中stateは永続化しないためresume不可。失敗runを保持し、同じcommandで
freshに再実行する。結果をreviewするまで既定化せず、performance / 32K / 256Kも未qualifiedとする。

`expert-bank/full-backbone-20260916-010306-7117`をreview済み。one-token 40-layerのhidden、
pre-mix、state、logits、route tie recordsはindividual / full-bank間でbit一致し、標準3-token
lifecycleもpassした。wall 110.39秒、最大RSS 275,597,082,624 bytes、peak footprint
333,794,721,120 bytes、swap 0。ただし前後`vm_stat`差分はcompressions 5,971,439 pages、
decompressions 6,220 pagesで、40 full bankによるmaterialなmemory pressureを示す。したがって
full-backbone correctness gateだけをpassとし、40層bank常駐の既定memory promotionはrejectする。

代替として、on-demandでresidentになった選択6 expertだけをtokenごとに`stack`する経路も
layer 0でbit一致したが、stack生成を含むwarm medianは個別1.551709 ms、grouped 1.757875 ms
（0.8827倍）で遅い。このtoken単位経路はperformance候補としてrejectし、opt-in診断に留める。

1層full bankを128 tokenの間だけ保持し、768 routeを単一batched `gather_qmm`へまとめるprobeでは、
token逐次packed bankとのround前FP32 accumulated / BF16 routedがbit一致した。medianは逐次
39.599458 ms、batch 23.878166 ms（1.6584倍）、MLX peak 7,311,597,120 bytes。これを
layer-major chunk executorへ進むbounded evidenceとする。実route、gate / shared expert、stateful
attention、40層bank lifecycleをまだ含まないため、full-path performanceや32Kはqualifyしない。

さらに実Gate route、shared expert、route-weight適用、sharedとの最終加算を含むlayer 0 full MoEの
128-token batch APIを追加した。token逐次individual経路とのshared / routed / totalはbit一致。
`expert-bank/moe-batch-128-warm-20260916-01`では初回individual 0.332882秒、full bank構築込みbatch
1.589774秒。一方bank構築後の交互5組medianはindividual 0.169750秒、batch 0.059011秒
（2.8766倍）。単純な差分では構築費の回収は約11.35 chunk、約1,453 tokenだが、layer 0単一runの
概算であり昇格閾値には使わない。`release_packed_bank()`を用意し、次はstateful attentionを保った
layer-major chunk境界でbankを1層ずつ構築・解放する。40層同時常駐へ戻さない。

2026-09-16追記: `GateReference::forward_batch`で全tokenのscore graphを先に構築し、route scoreの
GPU→CPU同期をtileあたり1回へ統合した。最初に試した2-D gate matmulは実checkpointのrouted bitsを
変えたため採用せず、one-row matmulのreduction scheduleを維持する。短いlayer 0 / 128-token probeは
shared / routed / total bit一致、serial warm median 0.169902458秒、batch 0.027609959秒
（6.1537倍）、MLX peak 8,667,009,980 bytesだった。route policy単体もID、weight bits、tie metadataが
一致した。これはhost同期を128回から1回へ減らしたbounded resultで、Top-Kとassignmentはまだhost、
full-backbone / performance / 32Kは未qualifiedである。

続いてTop-7、`gather_qmm`用token / expert assignment、expert ID昇順の6-route FP32 reductionを
parallel Metal kernelへ移した。hostへ戻すのはtileごとのselected / boundary ID、boundary score、
errorだけで、通常のexpert演算はdevice arrayを直接消費する。CPU route policyとのID、weight bits、
tie metadata、およびlayer 0 / 128-token serial individual経路とのshared / routed / total bitsは一致。
device dispatch後のprobeはserial warm median 0.170790208秒、batch 0.023039500秒（7.4129倍）、
MLX peak 8,659,125,956 bytesだった。単一layer / resident bank warm後のbounded値であり、resident atlas、
attention、full-backbone performance、32Kをqualifyしない。

expert bank loaderはMLX allocatorの最終所有bufferへcheckpoint payloadを直接`pread`するよう変更し、
host vectorとMLX allocationの同時保持を除去した。`expert-bank/run-20260916-083356-14519`をreview済み。
exit 0、identity一致、選択6 expertのintermediate / weighted output / FP32 accumulation / BF16 sumが一致。
7,219,445,760-byte bankの構築0.6914835秒、MLX active 7,332,777,496、peak 7,333,825,620 bytes、
最大RSS 7,438,286,848、peak footprint 7,780,800,272 bytes、swap 0だった。旧reviewed runの構築
1.9446秒、最大RSS 9,855,123,456 bytesに対する単一run差はqualification速度に使わないが、
一時的なfull-size source copyを持たないownership gateをpassとする。

40層resident atlas候補は`DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=1`で、model construction中に全bankを
transactionalに構築し、完成後だけ全blockへ共有する。packed bank無効時は設定を拒否する。layer 0 /
128-token probeでは明示的なrelease要求後もbankが保持され、serialとのfull MoE bitsが一致した。
40層2×128のmodel-lifetime load / warm reuse / memory検査は次をユーザーが実行する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_resident_atlas_prefill_check.sh
```

1 model、40 layer、2×128 prefillとsample 1 token。model initializationで40 bankを最終bufferへ直接構築し、
両chunkで再利用する。resultの`packed_expert_bank_constructions`がexactly 40、各chunk progressの
`bank_constructions`が0でなければ失敗する。
数分、Unified Memory予算340 GB、論理checkpoint read約289 GB。ログは通常の
`artifacts/context-ladder/32k-run-日時-PID/`へ保存する。checkpointはread-only。途中atlas / model stateの
resumeはなく、失敗ログを保持してfresh rerunする。これはresident ownership / bounded prefill測定で、
external parity、performance、32K、256Kのqualificationではない。

`context-ladder/32k-run-20260916-083752-15291`をreview済み。exit 0、全identity一致、bank construction
exactly 40。chunk 0は52.154976秒、chunk 1は20.366817秒（6.284705 token/s）、256-token prefillは
72.527666秒（3.529688 token/s）。最終MLX active 300,232,626,504、cache 3,547,535,361、peak
302,906,528,592 bytes。最大RSS 274,643,566,592、peak footprint 304,490,591,320 bytes。swap増分0だが、
16 KiB pageでcompressions 2,730,769、decompressions 2,695,009（それぞれ約44 GiB）、file-backed pageは
5,009,938減少した。resident reuse gateはpassするが、memory-pressure gateとperformanceはpassしない。
旧rebuild-every-chunk runの129.679秒との差は条件が異なる単一runなのでqualification speedupにしない。

resident runの後、pure SWA / compressed producer / reuse consumerのprojectionをchunk化し、tokenごとの
output評価をchunk末尾へ統合した。causal attention順、producer / index publication、window / global state
commit順は維持する。短い実checkpoint検査では各attention outputとstateがtoken-serial経路とbit一致し、
layer 0 Blockの128-token continuationは0.478701秒から0.203458秒だった。これは単一layerのbounded値。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_layer_major_backbone_check.sh
```

40層へのpromotion check `layer-major-backbone-20260916-080331-13926`をreview済み。exit 0、
binary / source / fixture identityはすべて一致し、2×128 tokenのhidden / pre-mix / logits / 全state /
publication / hash / route tiesとinvalid-token atomicityがtoken-serial経路と一致した。wall 209.59秒、
最大RSS 161,229,012,992 bytes、peak footprint 165,197,902,976 bytes、MLX active
150,968,218,536 / cache 11,013,681,063 / peak 158,321,252,416 bytes。OS compression / swap増分0、
decompression増分11。この入力範囲のdevice routing / assignment / reduction correctness gateをpassとする。
比較用二重model、80回のbank構築、token-serial attentionを含むため、前回216.95秒との差やwall自体を
performance改善に数えない。>512-row Top-K、resident atlas、32Kも未qualifiedである。

attention chunk化後のpromotion check `layer-major-backbone-20260916-084839-16111`もreview済み。
exit 0、記録されたbinary / source / fixture identityは一致し、2×128 tokenのhidden / pre-mix /
logits / 全state / publication / hash / route tiesとinvalid-token atomicityは両chunkともtoken-serial
経路とbit一致した。wall 205.03秒、最大RSS 158,829,903,872 bytes、peak footprint
162,887,872,008 bytes、MLX active 150,968,473,996 / cache 11,107,704,952 / peak
158,325,972,000 bytes。OS compression / swap増分0、decompression増分431。これによりattention
execution-boundary変更の40層correctness gateをpassとする。この比較用runは二重modelと
80回のbank構築を含むため、209.59秒からのwall差をspeedupにしない。次はresident atlas
の同一ハーネスをfresh rerunし、chunk 1のsteady prefillとmemory pressureを比較する。

最新のPhase 2 promotion `expert-bank/layer-major-backbone-20260917-003137-31497`をreview済み。
exit 0、identity一致。40層×2×128 tokenでhidden / pre-mix / logits / 全state / publication / hash /
route tieがbit一致し、invalid-token atomicityもpassした。active 150,972,764,108、cache
11,185,991,308、MLX peak 158,282,141,432 bytes、最大RSS 160,464,470,016、peak footprint
164,649,644,608 bytes、swap 0。比較用経路の80 bank構築を含むため184.43秒のwallはperformance
根拠にしない。この結果でchunk-wide mHC/stateを含むPhase 2の40層correctness gateを閉じる。

model-initialization atlasのfull-path run
`context-ladder/32k-run-20260917-003631-31787`もreview済み。exit 0、identityはcommit `4b35f7a`
の実装内容と一致する。model construction 26.154秒で40 bank / 15,360 expertsを一度だけ構築し、
17個のprefill chunkはすべてbank construction / expert load増分0だった。2,063-token prefillは
127.717秒、16.1529 tok/s、first decodeは0.086293秒、生成tokenはbaselineと同じ339。reviewed
baseline 225.927秒、9.1313 tok/sに対する単発観測差はwall -43.47%、throughput 1.769倍である。
route device batch 680、expert-major batch 680、assignment 495,120、diagnostic readback 0。

最終MLX active 300,171,527,368、cache 4,155,602,295、peak 302,822,327,564 bytes。最大RSS
273,030,742,016、peak footprint 305,189,124,288 bytes、swap 0。ただし16 KiB page counterの増分は
compression 2,136,698（約32.60 GiB）、decompression 2,129,881（約32.50 GiB）であり、340 GB予算内を
memory headroom十分とは解釈しない。Phase 1のmodel lifetime / transactional publication / warm-path
zero-construction / teardown条件とPhase 3初期scheduleのfull-path接続を確認したが、同一modelへの独立した
fresh request反復は未実施である。Phase 3のnon-empty expert tile dispatchは未昇格で、component再測定は
次項で別に記録する。

resident component profile `context-ladder/32k-run-20260917-005021-32331`をreview済み。exit 0、commit
`7567138`、tracked patch 0、identity全件一致。prefill 126.640645秒、16.29019 tok/s、生成token 339。
40層additive wall 126.130113秒に対しattention 69.651924秒（55.22%）、MoE 53.881925秒（42.72%）、
post-MoE 2.390266秒（1.90%）。40 initialization banks、17 warm chunksのconstruction/load 0、route
device batch 680、readback 0、expert-major batch 680。最大RSS 274,533,367,808、peak footprint
304,958,437,760 bytes、swap 0、compression / decompressionは約30.79 / 30.78 GiB。この同期profileは
component attributionであり通常scheduleのperformance qualificationではない。

この結果によりPhase 4のdevice index publicationを開始した。optimized resident runnerでは
`DSV41_RUNTIME_INDEX_DIAGNOSTICS=0`とし、top-k row/candidate maskをdevice authoritativeのまま
publication / republish / candidate consumer / attention gatherへ渡す。reference既定値1は従来host vectorと
tie診断を保持する。次の長時間確認は以下で、1 model、2,063 prefill + 1 decode、約289 GB checkpoint read、
340 GB予算、5--10分、ログ保持、resumeなし。route/index readbackとも0を要求する。

短時間fixtureはdiagnostics 1 / 0の両方で640-row candidate/top-kをtoken-serial oracleと一致確認し、
diagnostics 0のcompressed producer→device publication→reuse attentionもoutput/state bits、continuation、
fork/reset/rejectionを維持した。これはPhase 4の最初の境界変更だけをqualifyし、40層wallやchunk-atomic
frontier全体をqualifyしない。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_resident_atlas_prefill_measurement.sh
```

そのrun `context-ladder/32k-run-20260917-084044-36764`をreview済み。exit 0、revision `9bfeb71`、
tracked patch 0、identity全件一致。prefill 129.090453秒 / 15.9810 tok/s、decode 0.149775秒、生成token
339。40 model-init banks / 15,360 experts、warm construction 0、route device batch 680、route readback 0、
index readback 0、expert-major batch 680 / assignment 495,120。直前126.640645秒に対して1.93%遅く、
単発差はperformance改善に数えない。MLX active/cache/peakは300,171,601,115 / 4,028,291,832 /
302,822,401,315 bytes、最大RSS 272,842,407,936、peak footprint 305,053,726,576 bytes、swap 0。
compression/decompressionは約30.80/30.66 GiB。この証拠でdevice index publicationのfull-path
readback gateを閉じるが、Phase 4 wall-time gateは閉じない。

次のreuse chunk-attention候補はreferenceの64-key online-softmax順とBF16 probability castを維持しつつ、
token rowsを共通padded GPU graphへ移す。DwarfStarのbatch-attention fixtureに合わせ、実測を見る前に
hidden/pre-mix/logits relative RMS `<0.002`、全logits argmax一致、route tieと全persistent
state/publication/hash exactを固定した。3-token device-publication統合fixtureはpass済み。40層2×128の
長時間gateは以下。約578 GB logical checkpoint read、240 GB Unified Memory予算、所要最大5分、失敗ログ
保持、resumeなし。エージェント側では起動しない。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_chunk_attention_backbone_check.sh
```

run `attention/chunk-backbone-20260917-085608-37522`をreview済み。revision `62517e2`、tracked patch 0、
identity一致、swap 0だったが、最初の128-token chunk hiddenはrelative RMS `0.00554266`、max abs
`2048`、mean abs `1.02469`、BF16 mismatch 2,595,308で固定gate `<0.002`をfailした。そこで実行は
state/publication/route qualification前に停止した。rank-3 MLX chunk attentionはrejectし、既定offを維持する。
この失敗runからperformanceやPhase 4完了を主張しない。閾値は変更せず、次候補はDwarfStar型の
token/head batch attention kernelとfull-sweep後のatomic frontier commitを一体で評価する。

Metal候補の40層run `attention/chunk-backbone-20260917-092514-39493`もrejectした。clean `c746c4d`、
identity一致、tracked patch 0、swap 0だが、chunk 0 hidden relative RMS `0.0055142`、max `2048`、mean
`0.845199`、BF16 mismatch 2,587,396で固定上限を超えた。state/route gate前の停止であり、103.71秒wallは
performance qualificationに使わない。kernelは削除した。

次のshape-bucketed hybridはMLX scalarと同じSteel split-K QKおよびtoken-wise AVを保持し、raw offsetと
selected-row countが等しいtokenだけを同じsoftmax graphへ入れる。最大幅paddingを除いた公式layer 2→3の
2×128 fixtureはattention outputとstateがbit-exact。保持したscalar QK/AV callはtelemetryへ別計上する。
40層、route/index、wall timeは未qualificationで、長時間gateのcommandと固定閾値は変更しない。

40層run `attention/chunk-backbone-20260917-113353-41239`をreview済み。exit 0、commit `731c6aa`、
tracked patch 0、identity全件一致。両128-token chunkのhidden / pre-mix / logitsはbit-exactで、route tie、
全state/publication/hash、invalid-token atomicityもexact。最大RSS 160,473,219,072、peak footprint
165,164,675,696 bytes、swap 0。227.03秒はoracleとcandidateを同時に持つcorrectness harness全体なので
performance値ではない。shape-bucket candidateの40層correctness gateだけを閉じる。

次の長時間測定は1 resident model、2,063 prefill + 1 decode、component同期あり、340 GB予算、5--10分。
既存resident component profileと同じ測定semanticsでAttention wall、total prefill、chunk group、残存scalar
QK call、AV batchを記録する。失敗directory保持、resumeなし。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_shape_bucket_attention_profile.sh
```

40層gate後、同一shape bucket内のAVだけをrank-3 matmulへまとめた。公式layer 2→3のpositions 0--127 /
128--255 fixtureはoutput/state bit-exactで、scalar QKは変更していない。この短いfixtureは性能qualificationでも
40層qualificationでもない。component profileで効果を確認して候補を残す場合、同じ40層gateを再実行する。

component run `context-ladder/32k-run-20260917-120246-41847`をreview済み。exit 0、clean `cbca6c1`、
tracked patch 0、identity一致、swap 0。prefill 107.299秒 / 19.227 tok/s、Attention 46.708秒、MoE
57.206秒、post-MoE 2.656秒。比較対象の126.641秒 / 69.652秒 Attentionからprefill 15.27%、
Attention 32.94%短縮した。bank constructionはmodel-initの40だけ、route/index readback 0、生成token 339。
scalar QK call 613,439、scalar AV call 0、AV batch 117,579を確認した。このrunは性能候補の保持を決めるが、
AV拡張の40層correctness gateを代替しない。

AV拡張後のgate `attention/chunk-backbone-20260917-121232-42128`をreview済み。clean `130fcf6`、exit 0、
tracked patch 0、identity一致。両128-token chunkのhidden / pre-mix / logitsはbit-exact、route tie、全state /
publication / hash、invalid-token atomicityもexact。最大RSS 160,476,315,648、peak footprint
165,187,973,744 bytes、swap 0、compression増加0。194.38秒は二重model correctness harness wallであり、
performance値ではない。AV拡張の40層gateは閉じた。次は`run_resident_atlas_prefill_measurement.sh`で
chunk attentionを明示的に有効化し、通常lazy full-path wallを測る。

そのresident rerun `context-ladder/32k-run-20260916-100044-16544`をreview済み。exit 0、
全identity一致、bank construction exactly 40。chunk 0は49.217220秒、chunk 1は17.518658秒
（7.306462 token/s）、256-token prefillは66.739255秒（3.835823 token/s）。attention chunk化前の
同一resident harness比でchunk 1は13.98%短縮、steady throughputは16.26%上昇、prefill全体は
7.98%短縮した。単発paired observationでありacceptance thresholdには使わない。最終MLX active
300,232,980,456、cache 3,644,950,939、peak 302,906,882,544 bytes。最大RSS
275,314,180,096、peak footprint 304,587,486,296 bytes。swap増分0だが、16 KiB pageの
compressions 2,667,465、decompressions 2,638,747（約43 GiBずつ）が発生した。attentionの
execution-boundary変更は採用するが、40層full resident atlasはmemory-pressure gateをpassしない。

代替としてroute確定後のglobal expert ID集合だけを、MLX allocatorの最終所有bufferへ直接
`pread`するcompact bankを追加した。global-to-local ID変換後もexpert ID昇順のreductionは保つ。
192 expertのbounded probeでは3,609,722,880 bytes、構築0.318450秒（full 384 expertは
7,219,445,760 bytes、0.689934秒）で、accumulated / routed bitsはfull bankと一致した。
`expert-bank/device-route-20260916-100929-17026`もreview済みで、128-tokenの実routingは40 expertを
選択し、compact first path 0.095176秒、shared / routed / totalはfull bankとbit一致した。

one-token 40層promotion `expert-bank/compact-backbone-20260916-112530-17381`はexit 0、identity一致、
individual / compactのhidden / pre-mix / state / logits / route tiesがbit一致。標準3-token lifecycleもpass。
wall 26.23秒、最大RSS 43,973,935,104 bytes、peak footprint 47,121,603,728 bytes、compression /
decompression / swap増分0。これはone-token correctness gateであり、128-tokenの実route union、
prefill performance、I/O overlapをqualifyしない。次の2×128 promotionで全層の実際のunion数を記録する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_compact_layer_major_check.sh
```

individual referenceとroute-first compact bankを40層・2×128 tokenでbit比較する。数分、
Unified Memory予算100 GB。checkpointはread-only。論理readは実route unionに依存し、
full bank 80回の578 GBが上限。ログは`artifacts/expert-bank/compact-layer-major-日時-PID/`。
途中stateのresumeはなく、失敗ログを保持してfresh rerunする。結果review前にperformance /
32Kへ昇格しない。

compact 2×128 promotion `expert-bank/compact-layer-major-20260916-121151-17714`をreview済み。exit 0、
binary / source / fixture identityは一致し、2×128のhidden / pre-mix / logits / state / publication /
hash / route tiesとinvalid-token atomicityはindividual経路とbit一致した。compactのloaded expertは
10,543、80 bank constructions（両chunkのindividual / compact比較を合わせた値）。wall 135.45秒、
最大RSS 275,867,566,080 bytes、peak footprint 308,103,594,928 bytes、MLX active
150,968,473,996 / cache 156,248,259,671 / peak 155,311,236,876 bytes。OS compression 1,943,277 pages、
decompression 103,160 pages増加、swap 0。比較用individual modelのcacheが同一processに残るため、
このactive / cacheはcompact productionの上限として使わず、performance / 32Kは未qualified。

oMLXの汎用設計（cache block境界、prefill transientのadmission、allocation時のcache回収、SSD row prefetch）を
参照し、MLX allocator cacheの上限をchunk境界で適用するopt-inを追加した。
compact-onlyで比較用individual cacheを排除した短い測定は次のcommandで行う。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
CONTEXT_TOKENS=2064 TEACHER_TOKENS=0 TAIL_TEACHER_TOKENS=0 DECODE_TOKENS=1 \
DSV41_RUNTIME_PACKED_EXPERT_BANK=1 DSV41_RUNTIME_COMPACT_EXPERT_BANK=1 \
DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=0 DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=8589934592 \
DSV41_CONTEXT_EXECUTION=layer_major \
DSV41_RUNTIME_EXPERT_IO_THREADS=8 \
DSV41_CONTEXT_WALL_BUDGET_SECONDS=900 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0 \
bash tools/benchmark/run_context_32k.sh
```

1 model、40層、2048-token prefill + 1 decode、cache limit 8 GiB。数分、Unified Memory約100 GB、
route unionに応じたcheckpoint readを見込む。結果は`artifacts/context-ladder/32k-run-日時-PID/`の
result / progress / resourceへ保存し、cache limit、loaded expert counter、active/cache/peakとVM差分を確認する。
途中resumeはなく、失敗時はdirectoryを保持してfresh rerunする。このrunはcompact-only memory / I/Oの診断で、
performance、M3、32K、256K qualificationではない。

`DSV41_RUNTIME_EXPERT_IO_THREADS=8`は各bankのprojection readだけを最大8 workerへ分ける。tensor handleの取得、
buffer配置、Metal評価、route/reduction順序は単一threadで確定し、例外時は全workerをjoinしてから失敗する。
1 threadとのpaired測定前は既定化しない。

8-worker run `32k-run-20260916-131819-19052` をreview済み。exit 0、identity一致、prefill 2063 token
255.763933秒（8.066032 token/s）で、1-workerの321.949894秒（6.407829 token/s）よりwallが
20.56%短縮した。bank constructions 680、loaded experts 70,736は不変。MLX active
11,393,512,656、cache 8,377,779,966、peak 14,915,177,242 bytes、最大RSS 96,621,903,872、
peak footprint 97,804,712,440、swap 0。VM compression/decompressionの増加はそれぞれ0 / 2 pagesで、
cache limitとmemory headroomは維持された。sys time 478.47秒、involuntary context switchの増加があるため、
8 workerのwall改善はbounded observationとし、スレッド数の既定値をまだ変更しない。

`32k-run-20260916-123843-18457`は`DSV41_CONTEXT_EXECUTION=layer_major`が欠萷し、`execution=individual`で
走ったため失敗扱いとする。position 1024までの1.1457 token/sはcompact性能の
根拠に使わない。cache limit下のactive 15.91 GB、cache 0.40 GB、swap 0は観測として保存する。

fresh layer-major compact-only run `32k-run-20260916-130600-18634`はexit 0、identity一致、prefill 2063 token
321.949894秒（6.407829 token/s）、decode 0.080524秒で完了した。この結果はold canonical 2048 runの
5.7553 token/sより約11.3%高いが、contextとハーネス条件が異なるためspeedupとしてqualifyしない。
MLX active 11,393,512,656、cache 8,424,931,426、peak 14,915,222,298 bytes、最大RSS 96,611,958,784 bytes、
peak footprint 97,798,158,888 bytes、swap 0。cache limitは8 GiB近傍で推移し、activeは約10.6 GiBで安定した。
17 chunk × 40 layerの680 constructions、loaded expert 70,736（平均104.0 expert/bank）だった。index ties 9,395、route ties 4は存在するが、official oracle gapは残る。この測定はcompact-only memory / I/O診断として採用し、performance / 32Kは未qualified。

`CACHE_CONDITION`は観測ラベルだけであり、既定`unknown`をcold/warmへ読み替えない。この1 runは
動作・測定境界の初回観測で、反復、外部baseline、teacher-forced exactness比較、事前閾値を
満たさないため、exit 0でも32K performance / correctness qualificationとは判定しない。

初回`32k-run-20260915-230322-2124`はexit 1。165.94秒、最大RSS 140957204480 bytes、
peak footprint 141651408480 bytes、swap 0で、メモリ不足ではなく
`ambiguous index top-k boundary requires oracle`により停止した。compressed rowsが512を超える
実入力で初めて到達する離散境界である。公式も`torch.topk(sorted=False)`でtie順を規定せず、
oMLXは`argsort`を使うため、canonical full runtimeは最小row IDへ決定的に固定し、tie総数と
先頭256 record（layer / token / selected / excluded / exact score bits / candidate利用）を結果へ
保存する。strict単体APIの拒否は維持する。修正後のfresh 32K runのレビューまでは未判定。

次のrun `32k-run-20260915-232027-2685` は11,264 token時点、2348.47秒で中断されexit 130。
最大RSS 170474700800 bytes、peak footprint 174264406280 bytes、swap / compressorはいずれも0。
単純平均でも約4.80 token/sであり、32K prefillは約1.9時間相当になる。live sampleでは
decoder reused block内の`mx::eval`、GPU copy / concatenate、Metal allocator / resource bindingが
支配的だった。途中観測では約159 GiBのresident IOAccelerator領域と約5.5万regionが観測
され、GC反復ではなくcanonical referenceのtoken-by-token GPU同期とallocation churnである。
したがって、このrunのwall値をnative performanceとして扱わず、M3 execution graphの実装前に
32K performance qualificationへ昇格しない。
### Compact I/O worker sweep

worker数は独立process sweepで選定する。257-token context（256 prefill + 1 decode）、compact-only、
layer-major、cache limit 8 GiBを固定し、worker 1 / 2 / 4 / 8のwall、sys time、bank construction、
active / cache、VM差分を比較する。各run数分、Unified Memory 30 GB未満を見込む。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_compact_io_sweep.sh
```

logsは`artifacts/context-ladder/compact-io-sweep-日時-PID/workers-N/`に保存する。各sub-runは失敗しても
残りを継続し、失敗ディレクトリを保持してsweep全体をfresh rerunする。結果はperformance、M3、32K、256K
qualificationではない。

`compact-io-sweep-20260916-132600-19251`をreview済み。全4 sub-runはexit 0、identity一致、
loaded expert 9,748、bank constructions 80で揃った。1 workerは40.138459秒（6.377923 token/s、sys
25.70秒）、2 workerは32.661227秒（7.838040 token/s、sys 27.79秒）、4 workerは28.832884秒
（8.878751 token/s、sys 31.80秒）、8 workerは30.913908秒（8.281062 token/s、sys 70.81秒）だった。
4 workerは8 workerよりwall 6.7%短く、sys負荷も低いため、次の32K候補値とする。ただし反復pairedは未実施のため、runner既定値1は変更しない。

### Prefill execution-work gap capture

最新の通常lazy run `context-ladder/32k-run-20260917-123159-42514`はclean `7a7c66a`、exit 0、
tracked patch 0、identity一致。2,063 prefillは106.233166秒 / 19.41955 tok/s、生成token 339、
warm bank constructionとroute/index readbackは0、swapも0だった。このrunとpinned oMLX/ds4 codeの
比較結果は[gap audit](prefill-gap-audit.md)に固定した。

Metal command / dispatch総数を閉じる長時間captureは以下。1 model、公式checkpoint、2,063 prefill +
1 decode、5--10分、Unified Memory 340 GB、one-time checkpoint read約289 GB。`GPU`だけでなく
`Metal Application`単体も11 GB traceのfinalizeがtarget終了後に止まったため、Instrumentsは起動しない。
process-local hookをprefill区間だけ有効化し、command buffer / compute encoder / dispatchを数える。
hook overheadがあるためwall性能値に使わない。ログは`artifacts/prefill-gap/metal-日時-PID/`、
失敗時も全保持、resumeなし。partial traceのcountは採用しない。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_prefill_gap_metal_capture.sh
```

成功後は`result.json`と`metal-dispatch-counts.json`を併読する。command buffer、compute encoder、
dispatchは別のcountであり相互に読み替えない。paired oMLX countのreviewまで局所kernel作業は停止する。

clean `2e363b7`のno-Instruments runはprefill限定でcommand buffer 269,115、compute encoder
237,345、compute dispatch 12,358,908を測定した。paired oMLX countは次をユーザーが実行する。
同じ2,063 token、公式checkpoint、5--10分、Unified Memory上限340 GB、checkpoint read約289 GB、
review済みoMLX設定のEngram SSD offload有効・MTP weights保持、ログは
`artifacts/prefill-gap/omlx-metal-日時-PID/`、失敗時保持、resumeなし。resident Engramでの初回試行は
model load中にpeak footprint 408,665,719,208 bytesでMetal OOMとなり、prefill未到達なのでcountには使わない。
次の試行はpinned sourceにapp bundleのMLX 0.31.2を混在させたためempty packed cache初期化で失敗した。
runnerは既存oMLX venvのoMLX 0.7.0.dev2 / MLX 0.32.2を事前検証する。

paired成功run `omlx-metal-20260917-221148-47392`はexit 0、oMLX `b390b31`、next token 339、
全40 cache offset 2063。command buffer 1,595、compute encoder 918、dispatch 7,463だった。
current比はそれぞれ168.7x、258.5x、1,656.0xで、hook-to-hook wall比9.89xを十分説明する。
これでgap auditを閉じ、次は局所kernelではなく2,048-row transactional layer sweepへ進む。

最初のsweep gateは2x128-token oracle chunkと1x256-token transactional layer sweepを比較する。
hidden/pre-mix/logits relative RMS <0.002、logits argmax、route ties、state/publication/hash、
invalid-request atomicityを確認する。5--10分、Unified Memory上限240 GB、checkpoint read-only、
ログは`artifacts/prefill-gap/layer-sweep-backbone-日時-PID/`、失敗時保持、resumeなし。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_layer_sweep_backbone_check.sh
```

`prefill-gap/layer-sweep-backbone-20260917-222206-47915`をreview済み。clean
`0790fb7`、exit 0、tracked patch 0で、hidden / pre-mix / logitsはbitwise exact、
logits argmax、route ties、persistent state/publication/hash、invalid-request
atomicityもexactだった。bank construction 40、loaded experts 15,360、active/cache/peak
154.19/12.54/162.06 GB、最大RSS 164.05 GB、peak footprint 169.27 GB、swap 0。
165.43秒はoracleとcandidateを含むgate wallでありperformance値ではない。この結果で最小sweep
correctness gateを閉じる。

production generation prefillは`DSV41_RUNTIME_LAYER_SWEEP=1`でのみ最大4096-tokenのtransactional
sweepを選択し、defaultとdecodeはreference scheduleのまま維持する。最初の2,063-token full-path観測は
resident atlasとqualified chunk attentionを使うが、batched dense QMMを無効のまま固定し、schedule反転を
単独測定する。5--10分、Unified Memory上限340 GB、checkpoint one-time read約289 GB、read-only、
ログはcanonical `artifacts/context-ladder/32k-run-日時-PID/`、失敗時保持、partial state publishなし、
resumeなし。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_layer_sweep_prefill_measurement.sh
```

初回 `context-ladder/32k-run-20260917-222949-48397` はclean `b970316`だったが、
model launch前にmacOS標準Bashの`set -u`が空のtarget-environment配列展開を拒否して終了した。
exit 1、tracked patch 0で、checkpoint load/prefill/state mutationには未到達のためqualification値を
含まない。runnerはtrace/direct両分岐で空配列を明示的に分岐し、空の場合は`env` wrapper自体を
省略するよう修正した。失敗ディレクトリは保持し、修正版をfresh rerunする。

修正版の`context-ladder/32k-run-20260917-223115-48574`をreview済み。clean `3dd18e6`、
exit 0、tracked patch 0、identity一致、token 339、swap 0。2,063-token sweepは
65.403334583秒 / 31.542734 tok/sで、clean layer-majorの106.233166秒から40.829831秒
（38.43%）短縮した。model constructionは40 banks / 15,360 experts、request-local
constructionは0、route/index readbackも0。peak MLX 304,384,904,163 bytes、peak footprint
309,932,423,624 bytes。これで2K full-pathへのsweep接続とwall改善を確認するが、単回runなので
反復performance qualification、32K、256Kは未qualification。

残存する約6.0xのoMLX差について次の変更を選ぶため、同一sweepのcomponent profileを実行する。
同期境界が通常lazy scheduleを乱すためtotal wallは65.403秒と比較せず、Attention/MoE/post-MoEの
支配比率とper-layer分布だけに使う。5--10分、Unified Memory上限340 GB、checkpoint read-only、
ログはcanonical context-ladder directory、失敗時保持、resumeなし。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_layer_sweep_component_profile.sh
```

`context-ladder/32k-run-20260917-224750-48769`をreview済み。clean `be602bd`、
exit 0、identity一致、token 339、swap 0。40層のGPU-completion component合計
60.378607秒のうちAttention 41.179713秒（68.2%）、MoE 17.837538秒（29.5%）、
post-MoE 1.129525秒（1.9%）だった。同期を含むprefill 62.822960秒は通常scheduleとの
performance比較には使わない。この結果により次の支配項をAttentionと確定した。

最初のone-dispatch QK/softmax/AV候補はclean
`attention/fused-chunk-backbone-20260918-013048-51465`、`392288c`でrejectした。chunk 0 hidden
relative RMSは0.000928321だったが、pre-mixは0.00896325で固定上限0.002を超えたため、route/state
gate前に停止した。peak footprint 142,117,727,720 bytes、swap 0。133.14秒wallはperformance値に
使わない。scalar MLX Steel split-Kと異なるSIMD reductionが40層で増幅したため、kernelは削除した。

代替候補はMLX 0.32.2 scalar oracleのM=64/K=512、BM32/BN32/BK16、WM2/WN2、split-K
partitionとordered accumulationをcomplete 64-key blockで保持し、独立token軸だけをblockごとの
2 dispatchへbatch化する。直接float32比較では1-column tailが最大0.000046、71要素不一致だったため、
short tailはnative scalar Steelに残す。
softmaxと既存qualified batched AVは変更しない。63/64/65/128 live-row合成fixtureと公式checkpoint
layer 3のposition 0/128はいずれもbit-exactになった。40層gateは約5分、Unified Memory上限240 GB、
checkpoint read-only、失敗ログ保持、resumeなしで、次をfresh stateから実行する。

この限定前のclean run `attention/batched-splitk-qk-backbone-20260918-015331-53102`
（`1b802f2`）はhidden RMS 0.000640952、pre-mix RMS 0.00811268でrejectした。peak footprintは
142,208,691,760 bytes、swap 0。107.69秒wallはperformance値に使わない。

```sh
bash tools/benchmark/run_batched_splitk_qk_backbone_check.sh
```

clean gate `attention/batched-splitk-qk-backbone-20260918-122034-57470`（`c5321ca`）は2つの
128-token/40層chunkでhidden、pre-mix、logitsがbit-exact、route tie、persistent state、publication、
hash、continuation、invalid-request atomicityもexactで通過した。batched full-width callは21,937、
scalar tail callは8,583。peak footprint 165,259,752,192 bytes、最大RSS 160,569,196,544 bytes、swap 0。
180.44秒はoracleとcandidateを両方実行するためperformance値ではない。

次はqualified candidateだけを65.403秒のlayer-sweep baselineへ重ねる。約5〜10分、Unified Memory
上限340 GB、checkpoint read-only、失敗ログ保持、resumeなしで実行する。

```sh
bash tools/benchmark/run_batched_splitk_qk_prefill_measurement.sh
```

このfull-path result review前は`DSV41_RUNTIME_BATCHED_SPLITK_QK`のdefaultを0のまま維持する。

full-path `context-ladder/32k-run-20260918-122624-57828`（`5a68d8d`）はexit 0、tracked patchなし、
token 339、state position 2063、40 banks、route/index readback 0、swap 0。prefillは62.619395秒 /
32.9451 token/sで、scalar-QK layer-sweep baseline 65.403335秒から2.783939秒（4.26%）短縮した。
scalar QK callは613,439から24,711へ減り、full-width batch callは100,428。splitとordered
accumulationの2 dispatchを数えるとQK Metal dispatchは約1,226,878から250,278へ4.90倍削減した。
一方でattention shape group 17,910、AV batch 117,579が残り、pinned oMLX 10.901336秒との差は
まだ5.74倍ある。次のcomponent profileは通常wall比較ではなく、残存Attention bucketの帰属に使う。

```sh
bash tools/benchmark/run_batched_splitk_qk_component_profile.sh
```

review済みcomponent run `context-ladder/32k-run-20260918-124010-58187`はexit 0、
tracked patchなし、prefill 59.934584秒 / 34.4209 token/sだった。Attentionは
41.179713秒から37.928151秒へ3.251562秒（7.90%）短縮したが、57.544021秒のlayer
component wallの65.9%を占める。17,910 shape group、117,579 AV batch、12,378 producer
token-serial callは残存したため、QK/AVを別targetとする段階は終了する。

最初の候補`DSV41_RUNTIME_PACKED_CHUNK_ATTENTION=1`はpinned oMLXの公式DeepSeek V4.1
packed attentionを現在のsplit pooled-cache layoutへ適応したもの。DwarfStarの同型kernelは
FP16 Q/K/Vのため採用しない。candidateは1 tokenあたり1つの256-thread threadgroupで、
QK/mask/online softmax/BF16-rounded PV/AV/sinkをfusionし、pooled cacheを直接decodeする。
local-only最小fixtureはbit-exact、packed-pooled fixtureはrelative RMS 0.000210066でgate内。
これはfull-path qualificationではない。約5分、240 GB Unified Memory、checkpoint read-only、
失敗ログ保持、resumeなしの40-layer gateは次で実行する。

```sh
bash tools/benchmark/run_packed_attention_backbone_check.sh
```

このgateのreview前に2K wallをqualification扱いしない。通過後のisolated measurementは
`bash tools/benchmark/run_packed_attention_prefill_measurement.sh`で再現する。

最初の40-layer run `attention/packed-fused-backbone-20260918-131753-60269`は
chunk-0 hidden relative RMS 0.00328311（max absolute 2048、mean absolute 0.671945）で
固定上限0.002を超えたためrejectした。clean `8dda737`、swap 0。performance結果ではない。
公式layer 2→3の3-token fixtureでもRMS 0.005740を再現し、差はstate/routing以前のoMLX MMA
reductionに局所化された。閾値は変更しない。

最初のfallbackはpacked work listをchunk最大幅へpaddingし、qualified済みSteel reductionへ渡したが、
2回目の40-layer run `attention/packed-fused-backbone-20260918-132819-61332`もrejectした。hidden RMS
0.000780165は上限内だった一方、pre-mix RMSは0.00817721（max 0.0319417、mean 0.00193994、
508 values）だった。selected幅のpaddingがreduction geometryを変え、その差をmHCが増幅した。

修正版はoracleと同じ`(raw_width, selected_count)` host groupとSteel QK/AV shapeを維持する。
各groupのlocal rowとselected packed pooled rowだけを1 device dispatchでexact-shape tensorへ
materializeし、per-token gather/decode graphを除去する。したがって17,910 group除去やattention
operation fusionはまだ主張しない。公式2-layer 2x128-token fixtureではhidden/pre-mix/logitsが
bit-exact、state/publicationもexactに戻った。更新済みrunnerは同じ40-layer gateでこの限定候補を
検証する。

40-layer gate `attention/packed-fused-backbone-20260918-133655-62171`をreview済み。
clean revision `3878755`、tracked patch 0 bytes、exit 0、swap 0。2つの128-token chunkで
hidden/pre-mix/logitsはbit-exact、route tie、state/publication/hash、invalid-request atomicityも
exactだった。telemetryはpacked attention chunk 76、batched split-K QK 23,810、scalar QK
9,594。これによりexact-shape packed materializationだけを2K full-path測定へ昇格する。
performanceおよび17,910 shape group削減は未qualifiedのままとする。

一度だけ実行した2K packed-materialization測定
`context-ladder/32k-run-20260918-140056-62434`をreview済み。clean `ea27464`、
tracked patch 0 bytes、exit 0、swap 0。prefillは48.749250秒 / 42.3186 token/sで、
比較対象62.619395秒から13.870145秒（22.15%）短縮した。この結果でmaterialization単体の
探索は終了し、局所QK/AV/tail最適化へは進まない。

Phase 4の次candidateは`DSV41_RUNTIME_WIDE_ATTENTION=1`。DwarfStar MIT heads8
indexed-attentionのwork ownershipを公式BF16 Q/local-KVとFP4/E4M3 packed pooled cacheへ
適応し、1 compressed layer x chunkを1 dispatchとしてQK/mask/online softmax/BF16 PV/AVまで
所有する。index-sourceが固定`[tokens,512]` device row planをpublicationへ載せ、後続reuse layerは
同じplanを再利用する。standalone local/packed fixtureのRMSは0.000195373 / 0.000256431。
component結果に過ぎず、full-backbone昇格は次の約5分、240 GB、checkpoint read-only、
失敗ログ保持、resumeなしのgateで行う。

```sh
bash tools/benchmark/run_wide_attention_backbone_check.sh
```

このgateのclean run `attention/wide-backbone-20260918-225410-67687`はreject。
revision `eee85ac`、tracked patch 0 bytes、exit 1、swap 0。hidden relative RMSは
0.0111344（max 2048、mean 1.52399、2,599,165 values）で、固定上限0.002を超えた。
したがってDwarfStar型row-serial reductionは無効のままとし、gateを緩和しない。dense device
planとchunk-atomic publicationはfull backboneまで到達しているため、今回のrejectはそれらではなく
attention reduction arithmeticに帰属する。

次のcandidate `DSV41_RUNTIME_FIXED_TILE_ATTENTION=1`は同じ`[tokens,512]` device planを共有し、
128 local + 512 pooled metadata slotsを常に10個の64-row tileとしてqualified済みSteel split-K QK /
BF16-rounded PVへ渡す。可変selected-count group、short-tail host grouping、token/head host loopは使わない。
これは最終one-dispatch fusionではなく、固定tile/padding semanticsを分離して検証するPhase 4 bridge。
約5分、240 GB、checkpoint read-only、失敗ログ保持、resumeなしのgateをagentは実行しない。

```sh
bash tools/benchmark/run_fixed_tile_attention_backbone_check.sh
```

最初のfixed-tile run `attention/fixed-tile-backbone-20260918-231912-68890`もreject。
clean `855756b`、tracked patch 0 bytes、exit 1、swap 0。hidden RMSは0.0111286
（max 2048、mean 1.45948、2,597,542 values）。global token 0のraw-width差またはpaddingが原因という
仮説を短いfixtureで確認したが、129-row exact segmentとinvalid-padded
640-row pathはbit-exactだった。このためbootstrap workaroundは採用しない。次は公式producer layerと
first reuse layerについてchunk 0..127 / 128..255をdownstream amplification前に比較し、packed work-list /
publicationとattention arithmeticを分離する。この結果をreviewするまで2K測定や新しい局所kernelへ進まない。

code comparisonでchunk 0の具体的な差を特定した。exact pathはtoken 0をraw width 1で処理し、token 1以降の
128-row causal windowをleading paddingでright-alignする。fixed materializerはchunk全体の`q_offset == 0`を
使ってlive prefixをleft-alignしていた。row identityは同じでもpooled boundaryに対するslotが異なる。
修正版はfixed windowをright-alignし、同じisolation runnerでdense-contentをexact shapeへ戻した比較と
complete dense reduction比較を別々に出力する。runner passまでは未qualified。

clean isolation `attention/fixed-tile-isolation-20260918-234217-70452`ではdense contentが両chunkで
bit-exact。dense reduction RMSは`1.68876e-05` / `4.10141e-05`、producerは`0.000498815` /
`0.000375034`、first reuseは`0.000424763` / `9.84098e-06`で全て0.002未満だった。
publication/window/position checkにも到達した。exit 1はdiagnostic exact graphがproduction scalar-QK
counterへ混入したharness defectによる。diagnostic後にtelemetry snapshotを復元するよう修正したが、
promotionにはexit 0の再実行を要求する。

exit-0 rerun `attention/fixed-tile-isolation-20260918-234431-70930`をreview済み。
clean revision `77ca6c4`、tracked patch 0 bytes、exit 0、swap 0。同じsemantic値を再現し、
publication/window/positionとproduction topologyもpassした。これにより2-layer isolationだけを
40-layer backbone gateへ昇格する。production/performance/2Kは未qualified。

40-layer rerun `attention/fixed-tile-backbone-20260918-234534-71087`はright-aligned候補でも
final hidden RMS `0.0111282`（max 2048、mean 1.45624、2,597,265 values）でreject。
layer 2/3 isolationはpassしているため、後続layerの増幅または後続index-source publicationを切り分ける。
次のrunnerは1 chunkについて各layerのattn-in/out、post-attn、ffn-in、MoE-out、hidden、pre-mixを比較し、
最初に0.002を超えるstageを報告する。diagnosticのみでperformance/qualificationではない。

`attention/fixed-tile-layer-localization-20260918-235113-71445`をreview済み。layer 0/1は全境界で
bit-exact。layer 2はattn-out `0.000478654`、post-attn `0.0000561539`、ffn-in `0.00041744`だが、
MoE-outで`0.00642699`へ増幅し、layer 3以降へ伝播した。末尾の`producer window: bit mismatch`は
semantic divergence後にもexact final stateを要求したdiagnostic harness defectであり、stage trace自体は
有効だがrunをPASSとはしない。harnessからその不正な最終assertionを除去した。fixed paddingの局所RMSを
理由にpromotionしない。次はexact work-list上でragged tailのQKだけ、AVだけを64 shapeへ変更して寄与を
分離し、device-wide ragged-tail operationが保存すべきreduction contractを決める。この診断は
production kernel/performance qualificationではない。

attribution rerun `attention/fixed-tile-isolation-20260919-015117-73330`はexit 0、swap 0。
chunk 0のdense reduction 115 mismatches中QK tailが106（RMS `1.55003e-05`）、AV tailが9
（`6.70324e-06`）。chunk 1はQK tailが112 mismatchesとRMS `4.10141e-05`の全量を占め、
AV tailはbit-exactだった。publication/window/positionもexact。次candidateはselected widthを
materializerからdevice metadataとして出し、native Steelと同じ3 width class（1–32: BN16/P16、
33–39: BN16/P8、40–63: BN32/P8）を固定dispatchする。host selected-count grouping/readbackは
導入しない。`DSV41_RUNTIME_RAGGED_TAIL_QK=1`は公式2-layer isolation通過までopt-inである。

opt-in run `attention/fixed-tile-isolation-20260919-020313-74339`はexit 0、swap 0。
chunk 1のdense reductionはbit-exactになり、chunk 0はAV由来9 mismatches、RMS
`6.70324e-06`だけが残った。producer RMSは`0.000178388`、first reuseは`0.000105944`へ低下。
publication/window/positionはexact。40-layer gate前に同じfixtureでAV mismatchを持つtokenの
count/first/lastだけを追加確認し、request-frontier固有か一般ragged AVかを決める。

`attention/fixed-tile-isolation-20260919-020514-74675`ではAV mismatchは8 tokens、first 67、
last 122で、token 0固有ではなかった。chunk 1は引き続きAV bit-exact。ただし9 BF16 valuesだけを
理由に新kernelは追加しない。ragged-QK付き40-layer stage traceでhidden/pre-mix/logits hard gateへの
増幅を確認し、必要な場合だけgeneral ragged AVを実装する。localization runnerは
`DSV41_RUNTIME_RAGGED_TAIL_QK`を引き継ぐ。

ragged-QK trace `attention/fixed-tile-layer-localization-20260919-020702-74845`ではlayer 0–2が
全段bit-exact。layer 3 reuse attn-outでRMS `0.000158488`が生じ、layer 4 attn-outで
`0.00972281`へ増幅した。first gate failureはlayer 4 attentionへ移ったが未解消のため、general
ragged AVが必要。MLX Steel regular GEMMと同じtail-first K reductionをdevice work-listへ適応し、
width 1/33/40/63 synthetic fixtureはBF16 exact。`DSV41_RUNTIME_RAGGED_TAIL_AV=1`は公式
2-layer/40-layer gate通過までopt-in。

最初のBM32/BN32/BK16 ragged-AV候補は公式isolation
`attention/fixed-tile-isolation-20260919-022833-75572`でrejectした。chunk 0のdense-reduction差を
9から11 BF16 valuesへ、producer RMSを`0.000178388`から`0.000266951`へ悪化させた。pinned
MLX 0.32.2のlarge-device float32 NN selectorを再確認し、AV固有の正しい構成
BM64/BN32/BK32、WM2/WN2へ修正した。synthetic width 1/33/40はbit-exact、width 63は
padded-K diagnosticに対して1 element、RMS `6.8384e-10`、max `1.19209e-07`。誤った候補は
promotion対象外であり、修正版も公式2-layer gateを再通過するまではopt-inのままとする。

修正版の公式isolation `attention/fixed-tile-isolation-20260919-023456-76747`はclean
`0303aa6`、exit 0、swap 0。ragged QK/AVを両方有効にした状態で、2 chunksのdense reduction、
producer、first reuseがすべてbit-exact、publication/window/positionもexactだった。これは2-layer
境界だけを閉じる結果であり、promotion前に40-layer stage-localizationを通す。

40-layer localization `attention/fixed-tile-layer-localization-20260919-023624-77030`は
promotion gateを閉じなかった。layer 0–2はbit-exactだが、layer 3 attn-out RMS
`0.000155028`からlayer 4 attn-out `0.00895643`へ増幅し、固定上限`0.002`を超えた。
layer 3 attn-inはbit-exactなので、次の同一runnerはlayer 3/4のqr、q/kv、attention core、
inverse RoPE、grouped projection、output linearも記録する。projection起因かwork-list/core起因かを
一回で確定する診断であり、新kernel実装またはthreshold緩和ではない。

attribution rerun `attention/fixed-tile-layer-localization-20260919-024057-77425`ではlayer 3の
qr/q/kvがbit-exactで、最初の差はattention coreの2 BF16 elements、RMS `8.99269e-07`だった。
grouped projectionで71、output linearで1,189 elementsへ増幅した。MLX 0.32.2 regular Steelは
`batch_size * M * N`でtileを選び、128-token fixed AVだけがlarge-matmul閾値を越えるため、
token-serial oracleとreduction構成が変わっていた。candidateはhost token loopへ戻さず、全64-row
AV blockをdevice token軸へまとめたままoracleのBM64/BN32/BK32、WM2/WN2で実行する。ragged tailも
同じoperationで処理する。synthetic attention suite通過後もopt-inのまま、公式isolationと40-layer
gateを再実行する。

all-block AV修正版の公式isolation
`attention/fixed-tile-isolation-20260919-024624-77938`はclean `1d95ff9`、exit 0、swap 0。
両chunkのdense reduction、producer、first reuse、およびpublication/window/position stateは
bit-exact。2-layer境界のみqualifiedとし、40-layer backboneとperformanceは未qualified。

続く必須40-layer rerun
`attention/fixed-tile-layer-localization-20260919-024736-78210`は、この原因仮説をrejectした。
layer 3 coreの2 BF16差（RMS `8.99269e-07`）、grouped projectionの71差、output linearの
1,189差、layer 4 attn-out RMS `0.00895643`はtail-only runと同一だった。したがってfull-block
AVのbatch-size依存tileは残差を説明しない。all-block custom AVはrollbackし、qualified済みの
ragged-tail operationだけをopt-inで残す。次の同一runnerはarithmeticを変更せず、layer 3 coreの
first/last mismatch tokenとdevice selected widthを出力し、ragged tailとfull-block scheduleを
切り分ける。この診断が通っても40-layer/full-path promotionにはならない。

`attention/fixed-tile-layer-localization-20260919-030145-79111`ではlayer 3 coreの差は
exactly 2 tokens、first=1/last=2で、両方のselected widthは1だった。layer 0–2 exact、layer 4
attn-out RMS `0.00895643`を含む以降の値は再現した。したがってfull-blockではなく最小pooled
tail境界に限定された。次のrunnerは同じlayer 3実入力に対し、width-one QKだけ、またはAVだけを
native token-scalar shapeへ置換した結果を別々にoracle coreと比較する。production arithmetic、
publication、state、promotion gateは変更せず、このattributionだけではqualificationにならない。

`attention/fixed-tile-layer-localization-20260919-140336-81730`ではnative width-one QK置換だけが
layer 3 coreをbit-exactにし、native width-one AV置換は2差を残した。pinned MLX 0.32.2との
コード比較で、ragged QK producerだけが公式`steel_gemm_splitk`の`gemm_loop`後threadgroup
barrierを欠いていた。device work-list、dispatch数、materialization、host loopを変えずに公式と
同じ同期を復元した。次は同じ40-layer localizationを再実行し、結果を読むまでpromotionしない。

clean rerun `attention/fixed-tile-layer-localization-20260919-141217-82333`はbarrier追加前と
全数値が同一であり、同期単独の原因仮説をrejectした。公式同期は保持し、width-one partial layout仮説を
次に検証した。

clean exact-layout rerun `attention/fixed-tile-layer-localization-20260919-141802-83005`も
全数値が同一で、この仮説をrejectした。pinned MLX 0.32.2のdispatcherを再確認すると、
`min(M,N)==1`はSteel GEMM selectorより前に`gemv_axbpy`へ分岐していた。したがってwidth-one QKの
oracleはsplit-Kではなく`GEMVKernel<float,4,1,1,32,4,4,false>`である。このnative geometryを
token device軸へ展開するfixed GEMV classを追加し、width 2–63は既存Steel classを維持する。
host token loop、selected-count readback、shape-count dispatchは追加しない。短いsynthetic attention
fixtureはbit-exactだが、同じ40-layer localizationの結果確認まで未qualified。

公式2-layer isolation `attention/fixed-tile-isolation-20260919-145102-84647`は、このfixed
GEMV candidateでpassした。両chunkのdense reduction、producer、first reuse、および
publication/window/position stateはbit-exact、exit 0、swap 0。これはproducer/reuse境界のみを
qualifiedし、40-layer backbone/full-path performanceは未qualifiedのままである。

clean 40-layer localization `attention/fixed-tile-layer-localization-20260919-145317-84992`では、
layer 0–19の全記録stageがbit-exactとなり、従来のlayer 3 core差とlayer 4 gate failureは解消した。
残差はdecoder境界へ移り、layer 20 attn-out RMS `0.000536897`が最初の非zero、layer 21
attn-out RMS `0.013216`が最初のgate failureだった。revisionはclean `ba03593`、exit 0、swap 0。
次の同一runnerはproduction arithmeticを変えず、decoder layer 20/21の
qr/q/kv/core/inverse-RoPE/grouped/linearとlayer 20 mismatch-token selected widthを記録する。

最初のinstrumented run `attention/fixed-tile-layer-localization-20260919-145833-85349`は、
layer 20 reporterがencoder prefixを固定参照したため新規inner-stage出力前に停止した。直前までの
aggregate値は前runを再現したが、このfailed runから新しいsemantic結論は出さない。reporterは
layer ownershipに応じてencoder/decoder prefixを選択するよう修正した。

prefix修正後のrun `attention/fixed-tile-layer-localization-20260919-150155-85601`はlayer 20
coreまで到達した。qr/q/kvはbit-exact、attn coreでRMS `4.03974e-05`、1,425 BF16差が初めて発生し、
inverse RoPEでも同じ差だった。その後serial oracleだけが`attn_grouped`をtoken内と共通出口で二重記録し、
shape mismatchで停止した。重複観測は除去した。このfailed runは後段をqualifyしないが、decoder残差を
projection前のattention coreへ限定する。

complete decoder trace `attention/fixed-tile-layer-localization-20260919-151442-85887`では、
layer 20 qr/q/kvがbit-exact、core RMS `4.03974e-05`・1,425 BF16差、inverse RoPEは同じ差、
grouped RMS `0.000159245`、linear RMS `0.000536897`だった。core差はtoken 0の1件だけでselected
widthは1。layer 21は継承した入力差を増幅してgateを超えた。次の同一runnerはlayer 20実入力で
native scalar width-one QK/AVを別々に置換し、production arithmeticを変えずにdecoder producer残差を
attributionする。

clean completed run `attention/fixed-tile-layer-localization-20260919-152549-86166`では、layer 20の
native width-one QK置換とAV置換はいずれも元のcore値（RMS `4.03974e-05`、max `0.00390625`、
1,425 BF16差）を変えなかった。したがって残差はwidth-one kernelではない。token 0の有効行はlocal
slot 127とpooled slot 128の2行だけであり、serial oracleではcompactな2行、fixed scheduleでは別々の
64-row blockとしてonline-softmax合成される。短いsynthetic fixtureでもこのsparse境界だけでRMS
`9.87334e-05`、max `0.00390625`、30 BF16差を再現し、全129行有効のpadding fixtureはbit-exactのまま。
次のtraceはtoken 0だけcompact 2-row reductionへ置換してattributionする。production arithmeticと
fixed-tile enablementは変更しない。

follow-up `attention/fixed-tile-layer-localization-20260919-195319-87353`では、layer 20の
`attn_core_compact_token0`がRMS/max/bit mismatchすべて0となり、元coreとnative QK/AV置換は
1,425差のままだった。原因はblock間online-softmax合成順序で確定した。production候補は
request-boundaryをhost-known work-list metadataとして保持し、固定`[64,2]` QK/softmax/AV graphを1個だけ
追加する。width 1かつpooled row有効の判定はdevice上で行い、selected-width readbackとhost token loopは
追加しない。token 1–127は従来の10 tile scheduleを維持する。短いtoken-zero fixtureはbit-exactだが、
次の40-layer localization確認までfixed-tile attentionは未qualified・disabledのままである。

```sh
DSV41_RUNTIME_RAGGED_TAIL_QK=1 \
DSV41_RUNTIME_RAGGED_TAIL_AV=1 \
bash tools/benchmark/run_fixed_tile_attention_layer_localization.sh
```

```sh
bash tools/benchmark/run_fixed_tile_attention_isolation_check.sh
```

clean production-boundary rerun
`attention/fixed-tile-layer-localization-20260919-201526-88363`をreview済み。
revision `40ddbb6`、tracked patch 0 bytes、exit 0、swap 0。layers 0--39の
production stageはすべてrelative RMS/max/bit mismatchが0、
`first_gate_failure_layer=-1`、final persistent stateもexactだった。layer 20の
native-width-one attribution差はreject済み診断graphの値であり、実際に選択されるproduction coreと
request-boundary graphはbit-exact。この結果でlocalization gateを閉じ、fixed-tile scheduleを
production full-path candidateへ昇格する。ただし2K wall reviewまではruntime defaultを0に保つ。

次は順序付きでfull-backbone gateと2K測定を行う。前者は2x128 chunk、logits/continuation、route ties、
persistent state/publication/hash、invalid-request atomicityをragged QK/AV有効のproduction構成で確認する。
5分程度、Unified Memory上限240 GB、約578 GB logical read-only checkpoint reads、resumeなし。

```sh
bash tools/benchmark/run_fixed_tile_attention_backbone_check.sh
```

そのartifactをpassとしてreviewした後だけ、2,063-token transactional sweep + 1 decodeを測る。
5--10分、Unified Memory上限340 GB、約289 GB one-time read-only checkpoint reads、resumeなし。
結果はcanonical `artifacts/context-ladder/32k-run-日時-PID/`へ保存する。単回runなので反復performance、
32K、256K qualificationにはしない。

```sh
bash tools/benchmark/run_fixed_tile_attention_prefill_measurement.sh
```

production full-backbone artifact
`attention/fixed-tile-backbone-20260919-203130-89172`をreview済み。clean
`1b26648`、tracked patch 0 bytes、exit 0、swap 0。2つの128-token chunkでhidden、pre-mix、
logitsがbitwise exact、route ties、persistent state/publication/hash、continuation、
invalid-token atomicityもexactだった。fixed-tile calls 76、batched split-K QK calls 760、
packed/wide calls 0、scalar QK calls 0。peak MLX 158,284,371,083 bytes、最大RSS
160,446,562,304 bytes、peak footprint 166,793,032,328 bytes。172.25秒はoracleとcandidateを
両方実行するcorrectness gate wallでありperformance値ではない。この結果でproduction full-path
gateを閉じ、上記2K measurementを解禁する。測定結果をreviewするまではruntime defaultを0に保つ。

isolated 2K production observation
`context-ladder/32k-run-20260919-203604-89428`をreview済み。clean `8d015a8`、tracked patch
0 bytes、exit 0、next token 339、swap 0。2,063-token prefillは43.229745秒 / 47.7218 tok/sで、
比較可能なpacked exact-shape runの48.749250秒 / 42.3186 tok/sから5.519506秒（11.32%）短縮、
throughputは12.77%上昇した。batched QKは112,248から6,460（94.24%減）、AV batchは
131,418から6,460（95.08%減）、scalar QK/AVは0。peak MLXは304,385,559,523 bytes、最大RSS
275,054,755,840 bytes、peak footprint 310,843,160,624 bytes、decode 1 tokenは0.319738秒だった。
memoryは比較runと実質同じで、decode単発値からregression/improvementは判定しない。

これはfixed-tile scheduleをより速いproduction full-path candidateとしてacceptする単回観測であり、
performance hard gateやruntime default変更には不足する。次は2 warmups後にbaseline/candidate順を交互に
最低5組測る。既定12 process、60--120分、逐次実行のためpeak Unified Memory上限340 GB、aggregate
logical checkpoint read約3.5 TB、checkpoint read-only、swapなしを要求する。各childは独立しており、
失敗時はrootを保持し、同じ`DSV41_PAIRED_RUN_DIR`でexit-0 childを検証・skipしてresumeできる。

```sh
bash tools/benchmark/run_fixed_tile_attention_paired_qualification.sh
```

paired qualification `context-ladder/fixed-tile-paired-20260919-205611-90245`をreview済み。
rootと全12 childはclean `b5e47f2`、exit 0、tracked patch 0 bytes、recorded identity一致、
next token 339、request内swap増分0だった。5組すべてでfixed-tileが勝ち、prefill平均は
packed exact-shape baselineの49.288916秒から43.653977秒へ5.634939秒（11.43%）短縮した。
wall平均からのthroughput向上は12.91%、paired短縮の95% t区間は5.452--5.818秒で0を跨がない。
最小/最大短縮も10.99% / 11.71%でrun順に依存しない。1-token decode平均は
0.310786 / 0.310601秒でregressionなし。

candidateは全runでfixed-tile 646、batched split-K QK 6,460、AV batch 6,460、scalar QK/AV 0、
index readback 0、40 lifetime banks / 15,360 loaded expertsだった。candidateの平均MLX cache増分は
1,048,845,601 bytesだが、平均peak allocation差は442,352 bytes、平均peak footprint差は
501,428,616 bytes（0.16%）に留まり、最大footprint 310,870,768,360 bytesで340 GB予算内だった。

この結果でfixed-tile scheduleをproduction defaultへ昇格し、この5-run分布を新しい内部2K
production baselineとして固定する。未指定時はresident packed expert atlas、transactional layer
sweep、batched split-K QK、fixed-tile、ragged QK/AVを選び、per-layer finite scanとhost index
diagnosticsを無効にする。`tools/benchmark/run_fixed_tile_attention_prefill_measurement.sh`をbaseline
runnerとし、旧packed exact-shape pathは`tools/benchmark/run_packed_attention_prefill_measurement.sh`
で明示的に選ぶ比較/oracle fallbackとして保持する。各環境変数の明示値はdefaultを上書きできる。

この昇格は2K internal production prefill baselineの更新であり、32K、256K、API、長時間decode、
外部oMLX performance qualificationを意味しない。以後のoptimizationは43.653977秒平均を内部baseline
として同一条件のpaired gateで比較する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_omlx_prefill_dispatch_audit.sh
```

32Kへの次の長時間測定は、メモリフットプリントとI/O量が大きいため、実行前に以下のコマンドを用意する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
DSV41_RUNTIME_PACKED_EXPERT_BANK=1 DSV41_RUNTIME_COMPACT_EXPERT_BANK=1 \
DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=0 DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES=8589934592 \
DSV41_CONTEXT_EXECUTION=layer_major DSV41_RUNTIME_EXPERT_IO_THREADS=4 \
DSV41_CONTEXT_WALL_BUDGET_SECONDS=7200 DSV41_CONTEXT_PROJECTED_WALL_LIMIT_SECONDS=0 \
bash tools/benchmark/run_context_32k.sh
```

scopeは1 model、32K context、base prefill 28656、teacher continuation 4096、tail 2048、decode 16。
数十分〜2時間、Unified Memoryはcache limit 8 GiBとcompact bankにより約100 GB未満を目標とするが、
checkpoint readはroute unionに応じて数TB規模を見込む。結果は`artifacts/context-ladder/32k-run-日時-PID/`に保存され、resumeはできない。このrunは診断であり、32K / performance qualificationではない。

32K実測 `context-ladder/32k-run-20260916-133920-19687` はexit 0で完走した。compact-only / layer-major /
4 I/O worker / cache limit 8 GiB、base 28656、teacher 4096、tail 2048、decode 16の条件で、base
3456.331秒、teacher head 252.046秒、teacher tail 252.264秒、prefill合計3960.641秒（8.2694 token/s）、
プロセス全体3977.449秒だった。bank constructionsは10840（271 chunks × 40 layers）、loaded expertsは
1,076,905。active 11.48 GB、cache 8.43 GB、peak 15.77 GB、最大RSS 96.62 GB、swap 0で、cache境界と
memory headroomは維持された。一方、involuntary context switch 92.7M、page reclaim 1.17B、論理expert
read 1M超が残り、約66分を要する。これは32K動作確認とデータ収集であり、performance / 32K / 256K
qualificationではない。

次の最適化判断のため、context ladderは各compact bankのroute union（selected / unique / 前chunkとの
overlap / 完全再利用）を`result.json`とprogress JSONLへ記録する。重複率が高い場合のみbounded bank
reuseまたは次chunkのspeculative row prefetchを検討し、予測ミスはcanonical loadへfallbackする。重複率が
低い場合はprefetchを追加せず、bank constructionとMetal allocator境界の削減を優先する。

telemetry付き2064-token run `context-ladder/32k-run-20260916-151728-20227` はexit 0、prefill
243.586秒（8.4693 token/s）だった。680 bank batchesでselected 495,120、unique 70,736、連続chunk間
overlap 58,754（unique基準83.06%）、完全union再利用1回を記録した。loaded expert 70,736、bank
construction 680、active/cache/peak 11.39/8.53/14.92 GB、swap 0。SSDから毎回全unionを読み直す設計には
明確な再利用余地があるが、unionは毎回変化するため単純なbank保持だけでは不十分である。次はboundedな
layer/expert row cacheまたは二重bankへの差分prefetchを、bit-exact fallback付きで短いpaired probeとして
実装・比較する。今回もperformance / 32K qualificationではない。

I/O telemetry付き再測定 `context-ladder/32k-run-20260916-153052-20781` では、prefill 243.880秒に対し
expert bankのTensorFile readは41.018秒、bank構築全体は42.998秒（約17.6%）だった。read_callsは212,208、
route union値は前runと同一で、完全再利用も1回だった。このため、部分overlapをそのままrow cacheへ変換しても
全体短縮は限定的と判断し、次段はMLX/Metal dispatch、assignment中間tensor、eval/synchronize境界の削減を
優先する。I/O worker数やcache上限は現設定を維持する。

assignment chunk 256の診断 `context-ladder/32k-run-20260916-161056-21739` はexit 0で、prefill
243.066秒（8.4874 token/s）だった。128行chunkの243.880秒から約0.33%短縮し、expert readは40.974秒、
active/cache/peakは11.39/8.42/14.92 GBで変化しなかった。これはassignment中間処理の小さな改善を示すが、
このharness単体では外部parityを証明しないため、既定値は変更しない。512行chunkを同条件で測定し、
さらに改善するか、またメモリ・parityに影響しないかを確認する。

### Production 16K / 32K regime transition (2026-09-19)

fixed-tile production default後の最初のlong-context harness監査で、`sweep` modeが各phaseの残り全tokenを
一度に`forward_packed_sweep`へ渡し、4,096-token API上限を超えることを確認した。従来のcompact-only
32K artifactはこのproduction scheduleより前の証拠であり、現production pathの16K / 32K測定を代替しない。
harnessは各phaseを最大4,096 tokenのtransactionへ分割し、phaseごとのattention invocation/copy telemetryと、
component profile時のAttention / MoE / post-MoE wall deltaを保存するよう修正した。2K production defaultや
モデル演算は変更していない。

長時間runは次のstage-resumable runnerをユーザーが実行する。agentは自動起動しない。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_long_context_regime_measurements.sh
```

scopeはproduction wallと同期付きcomponent profileを16K、32Kで各1回、teacher continuation 4,096、
tail 2,048、decode 16。1 modelずつ逐次実行し、Unified Memory上限340 GB、checkpoint read-only、
総所要はcache/thermal条件により数十分〜数時間を見込む。rootは
`artifacts/context-ladder/long-regime-日時-PID/`、各attemptのraw result/progress/resource/config/identityと
集約`summary.json`を保持する。失敗attemptは削除せず、表示された同じrootを
`DSV41_LONG_REGIME_DIR`へ渡して再実行すると完了stageをskipし、失敗stageだけfresh model stateから再開する。
model state途中再開はしない。全4 resultとresource logをreviewするまで16K / 32Kをpassと呼ばない。

測定前に最適化順を結果へ合わせて変更しないため、判断面を固定する。production wall/TPS、teacher
continuation、decode mean/p95、peak bytes、Attention/MoE/post-MoE share、および16K→32Kの倍率を比較する。
decoder workがcontextとともに主要増分ならCED/deferred decoderへ進み、attention/index/cache copyの増分が
支配するならlong-context index/state/cache schedulingを先に閉じる。bounded replayはどちらの場合も
partial encoder-only frontierを公開しない回復契約としてCED promotion前に必要である。

4 stage artifact `context-ladder/long-regime-20260919-221200-93567`をreviewした。全childはexit 0、
revision `6c33f77`、同一tracked patch SHA-256
`6e311162dae777fce10b690fd7d718bbced3d9a4df9fcda5b759a30d024e0816`で、patchはこの測定を可能にした
context分割/telemetry/CED geometry変更と現在の該当diffに一致した。production flags、40 resident banks、
zero route/index readback、zero scalar QK/AV、state position、production/profile間の生成token列は一致し、
全runでswap 0だった。clean commit artifactではないためrelease evidenceにはせず、次の構造判断に用いる。

production prefillは16Kで320.690691秒 / 51.0398 tok/s、32Kで702.897563秒 / 46.5957 tok/s。
32K/16K wall比2.1918、throughput比0.9129。teacher continuation 4,096は81.234402秒から
120.067573秒へ47.8%増えた。component shareはAttention 58.08%→57.25%、MoE 39.22%→40.17%、
post-MoE 2.69%→2.59%で、計算支配比率はregime間でほぼ不変だった。したがってこの結果だけを理由に
decoder CEDを最初の変更とはしない。

一方、final MLX cacheは28,428,571,508 bytesから80,989,539,136 bytesへ52.56 GB増え、process peak
footprintも331,383,618,952 bytesから383,973,031,328 bytesへ同量増加した。32Kは340 GB予算を
43.97 GB超過している。MLX peak allocation自体は304,393,348,603→304,410,027,515 bytesとほぼ一定で、
active state増加ではなくreclaimされないallocator cacheが超過の直接原因である。decodeも16K mean/p95
3.188/3.473秒から32K 4.372/11.964秒へ悪化し、32Kの最初2 stepに5.992/11.964秒stallがある。
このartifactは32K動作観測を閉じるが、memory hard gate失敗のため32K qualificationはfailとする。

次candidateはlong-context時だけ16 GiBのMLX allocator cache limitを適用する。通常の明示cache limitが
最優先で、8,192 tokens未満と未指定時の既存production挙動は変えない。generation/API pathは
prompt + output reserveで同じpolicyを選ぶ。defaultはcandidate reviewまで0（disabled）のままである。
2K / 16K / 32Kを逐次比較する長時間runnerは次。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_long_context_cache_candidate.sh
```

scopeは2K production non-regression、16K/32K wall/decode/memory、1 modelずつ、checkpoint read-only、
想定30--60分、340 GB hard budget。stage-level resumeのみで、失敗時は表示されたrootを
`DSV41_CACHE_CANDIDATE_DIR`へ渡す。32K peak footprint、swap、token/state/topologyをreviewするまで
cache policyをdefault化せず、CED実装へ進めない。

cache candidate artifact `context-ladder/cache-candidate-20260920-011227-94806`をreviewした。2K / 16K /
32Kはすべてexit 0、revision `6c33f77`、同一tracked patch SHA-256
`ba5a7298cf0cdf7949e6f072198581f042b55f442abe32d78fc4d9db77d676b1`。各runは40 resident banks、
route/index readback 0、fixed-tile/ragged QK/AV、scalar QK/AV 0、正しいstate positionと基準と同じ生成token列、
swap 0を保持した。2Kではthreshold未満のためeffective limit 0で、prefill 43.329848秒、token 339、
peak footprint 310,878,287,968 bytes。cache policyは実行経路を変更していない。

16Kはprefill 320.555650秒でunbounded 320.690691秒と同等、teacher continuationも81.306422秒対
81.234402秒だった。final cacheは28,428,571,508から17,157,254,412 bytes、peak footprintは
331,383,618,952から323,197,894,024 bytesへ低下した。decode mean/p95は3.155/3.452秒で悪化なし。

32Kはprefill 639.081491秒 / 51.2486 tok/sで、unbounded 702.897563秒 / 46.5957 tok/sから
wall 9.08%短縮、throughput 9.99%増加。固定4,096-token continuationは120.067573→81.868415秒、
decode mean/p95は4.372/11.964→3.162/3.468秒。final cacheは80,989,539,136→17,174,477,252 bytes、
peak footprintは383,973,031,328→323,262,642,920 bytesへ60.71 GB低下し340 GB budget内へ戻った。
これはsingle baseline/candidate観測なので9.08%を反復performance qualificationとは呼ばないが、memory
hard gateを修復し、wall/decode regressionが観測されず、2K pathを変更しないlifecycle policyとして十分な
昇格証拠である。8,192 tokens以上の16 GiB limitをproduction defaultへ昇格し、環境変数`0`を明示した
unbounded pathを比較fallbackとして保持する。

次のCED sliceはdecoder suffixそのものの前にatomic ownershipを接続する。
`begin_deferred_prefill`は最大16,384-token encoder frontierをprivate stateへ作り、source sessionを変更しない。
`finish_deferred_prefill`だけがdecoder成功後にstateをcommitし、失敗・破棄・reuseではsource frontierを
publishしない。初版finishはfull decoder sweepを保持し、性能改善を主張しない。256-token full sweepとの
bit/state gateは数分かかるため次のuser runnerで行う。

```sh
bash tools/benchmark/run_deferred_transaction_backbone_check.sh
```

scopeはhidden/pre-mix/logits、route ties、全persistent state/publication/hash、discard/reuse/invalid atomicity。
5--10分、最大340 GB、checkpoint read-only、resumeなし。review通過後だけ、このtransaction内部へ
`1 + (39-L)*127` decoder suffixとbounded replayを実装する。

transaction gate artifact `prefill-gap/deferred-transaction-20260920-015305-96073`をreviewした。
exit 0、revision `6c33f77`、tracked patch SHA-256
`36e01d725051083d0e3f04269688604bf61eb76c97cde448ceeb45492c45e79e`で現在のgate差分と一致した。
256-token full decoderとのhidden/pre-mix/logits、全persistent state/publication/hash、実行route tiesは
bit-exactで、move/discard/reuse/invalid requestはsource stateを公開しなかった。swapは0、maximum resident
set sizeは32,186,302,464 bytes、peak footprintは38,053,417,640 bytes。この結果でtransaction ownership
gateをpassとし、exact suffix実装へ進む。

最初のsuffix candidateはDwarfStarの固定実装と同じ依存境界を使う。layer 20はskipped prefixの
global/index cacheと128-row raw windowだけを準備し、layer 21..39は直前127 input rowsからraw windowを
seedする。実行行数はlayer `L`で`1 + (39-L)*127`、最終layerは1行になる。attention、mHC、MoEの
演算は既存blockをそのまま呼び、production dispatchには未接続である。full decoderとの2,541-token
state gateは次を実行する。

```sh
bash tools/benchmark/run_deferred_suffix_backbone_check.sh
```

scopeはfinal hidden/pre-mix/logits、実行対象route ties、全decoder persistent state/publication/hash。
10--20分、最大340 GB、checkpoint read-only、resumeなし。failure artifactを保持しsource stateは非公開の
ままとする。このgateのreview前にproduction context ladderへ接続しない。

最初のsuffix gate `prefill-gap/deferred-suffix-20260920-020506-96918`はexit 1でrejectした。revisionと
tracked patch SHA-256 `9c973877e4812d43036a8b52db3bad76680b0f2d1fec43f22fbbbc4b20fa4cfe`
は実行時差分に一致し、swap 0、peak footprint 39,006,262,392 bytesだった。失敗は
`shared attention chunk plan mismatch`で、layer 20がstart 127で公開した128-row planを、127行縮んだ
layer 21のstart 254から再利用しようとしたことが原因。通常sweepの同一chunk境界という前提がCEDでは
成立しない。各publicationのimmutable device selectionを新境界でdevice arrayへ再packするfallbackを追加した。
index queryの再実行、host readback、model semanticsの変更はない。失敗transactionはsource stateを公開して
おらず、同じrunnerをfresh stateで再実行する。

修正後のartifact `prefill-gap/deferred-suffix-20260920-021359-97264`をreviewした。exit 0、revision
`6c33f77`、tracked patch SHA-256
`5ed8a6f333cab84188076dfb86afe0024b7de580dff805e59584501a8ceaa6d5`は実行時差分と一致した。
2,541-token full decoderに対し、final hidden/pre-mix/logits、全persistent state/publication/hash、実行対象
route tiesがbit-exact。discard/reuse/invalid atomicityも維持した。swap 0、maximum resident set size
34,085,126,144 bytes、peak footprint 73,761,281,264 bytes。この結果でexact suffix primitiveをpassとする。

production接続candidateは環境変数`DSV41_RUNTIME_DEFERRED_DECODER=1`でのみ有効。各phaseで8,192 tokens
以上残る場合、最大16,384-token private encoder transactionを作りexact suffixでcommitする。2K、短い
teacher tail、decodeは既存scheduleのまま。defaultはreviewまでOFFで、API defaultも未変更。full context
measurementは次。

```sh
bash tools/benchmark/run_deferred_decoder_context_candidate.sh
```

scopeは2K inactive non-regression、16K/32K CED chunk count、wall/decode/memory、production topology、state、
生成token、swap。1 modelずつ、20--45分、340 GB budget、checkpoint read-only。stage-level resumeのみで、
失敗時は表示rootを`DSV41_CED_CANDIDATE_DIR`へ渡す。review後にのみdefault/APIへ接続する。

candidate artifact `context-ladder/deferred-decoder-candidate-20260920-022230-97711`をreviewした。
2K / 16K / 32Kはexit 0、revision `6c33f77`、同一tracked patch SHA-256
`b5dfa512d978bc36c1a711573b1861a75915a02e4e01f93b38c3bd4467ef3feb`。40 resident banks、
15,360 loaded experts、route/index host readback 0、scalar QK/AV 0、swap 0、正しいstate positionとbaselineと
同じ生成token列を保持した。2KはCED chunk 0で43.815064秒、cache policyもinactive。16KはCED chunk 1、
prefill 320.555650→221.277165秒（30.97%短縮）、32KはCED chunk 2、639.081491→399.417787秒
（37.50%短縮）。teacher continuationは16K 81.306422→82.709436秒、32K 81.868415→83.694392秒で、
decode mean/p95はそれぞれ3.155/3.452→3.123/3.415秒、3.162/3.468→3.138/3.429秒。
peak MLX bytesは実質同一、process peak footprintは16K 324,303,503,712 bytes、32K
325,972,182,832 bytesで340 GB budget内。semantic/memory/full-context candidate gateはpassとする。

ただしperformance default promotionにはrun間変動を超える反復証拠が必要で、single observationの
30.97%/37.50%をそのままqualificationとしない。32Kで2回使うものと同一のCED transactionを1回使う16Kで、
各variant 1 warmup後に5 alternating pairsを実行する。

```sh
bash tools/benchmark/run_deferred_decoder_paired_qualification.sh
```

scopeは12 sequential 16K processes、想定60--100分、340 GB、checkpoint read-only。candidate win 5/5、
run-order varianceを超えるprefill改善、teacher/decode/memory/swap regressionなしを要求する。失敗時はrootを
`DSV41_CED_PAIRED_DIR`へ渡して完了childを検証skipする。通過後にdefaultとnative generation APIへ昇格する。

paired artifact `context-ladder/deferred-decoder-paired-20260920-024600-98127`をreviewした。warmupを含む
12 childとrootはすべてexit 0、revision `6c33f77`、同一tracked patch SHA-256
`36216bd2312e4ea0ee28c81cb99977ea7004b805b7d0c34996c297598cca6066`だった。全測定で生成列、
state position、40 banks / 15,360 experts、route/index readback 0、scalar QK/AV 0、swap 0を保持した。
5 pairのprefill平均はbaseline 320.902721秒、candidate 221.347515秒で31.02%短縮、candidate win 5/5。
baseline range 1.694135秒、candidate range 0.988108秒に対し各pairの改善は30.83--31.23%で、順序差を
十分超える。decode meanは0.23%、p95は1.11%改善、MLX peakは両方304,393,348,603 bytes、process
peak footprint平均はcandidateが0.43%増だが324.52 GBで340 GB内だった。

ただしteacher continuationは5/5で遅く、平均81.420167→82.488852秒（1.31% regression）。candidate
最良値もbaseline最悪値より遅いため、事前契約の「teacher regressionなし」をpass扱いしない。
CED defaultはOFFのまま保持する。native generation APIには同じ可変chunk scheduleをopt-inで接続したが、
default promotionとは呼ばない。phase telemetry上はteacherのattention row/indexer/call/concat topologyと
active/cache memoryが一致しているため、次は同期component profileでattention / MoE / post-MoEのどこに
差があるかを限定する。

```sh
bash tools/benchmark/run_deferred_decoder_continuation_profile.sh
```

scopeは16K baseline/CED各1 fresh process、phase-local component wall、生成/state/topology/memory/swap。
15--30分、最大340 GB、checkpoint read-only。失敗または中断時は表示rootを
`DSV41_CED_CONTINUATION_PROFILE_DIR`へ渡してexit-0 stageを検証skipする。原因を解消した後にteacher
non-regressionを再確認し、それからdefaultを変更する。opt-in native APIのfull-model parity gateは
`tools/benchmark/run_deferred_decoder_native_api_check.sh`として準備済みで、fallback/CEDの16K生成列と
next positionを比較する。

最初のcomponent artifact `context-ladder/deferred-continuation-profile-20260920-052800-656`をreviewした。
root/baseline/candidateはexit 0、revision `6c33f77`、同一tracked patch SHA-256
`0f1689308f1afedd5ae6fba8feab84c24c451a7fadc2307cf149bef898307931`。生成/state/topologyは一致し、
swap 0、process peakはbaseline 322,416,572,944 bytes、candidate 321,979,252,392 bytesだった。
同期profile下のteacher wall差は+0.261秒（0.32%）まで縮んだ。2 phase合計でAttentionはcandidateが
0.089秒、post-MoEは0.159秒速い一方、MoEだけが0.514秒遅く、同一1,280 component callsで差を説明する。
したがってattention topologyやCED state再構築を再調査せず、phase-local layer別MoE時間を次artifactに
保存する。特定layerだけならCED suffixのshape warmup、全層ならallocator/cache/execution-stateとして扱う。

layer別artifact `context-ladder/deferred-continuation-profile-20260920-054723-1241`もexit 0、revision
`6c33f77`、同一tracked patch SHA-256
`b1b15849312e2e20e619bca29a7405cd5bd50a60fd4ee485274acbc45b9baa6a`で、semantic/resource gateは
cleanだった。このrunではcandidate teacherが0.457秒速く、MoE差は全40層合計+0.083秒に縮小した。
最大のlayer 36 MoE差+0.129秒も同layer Attention -0.221秒に相殺され、2回の同期profile間で再現する
layer局在や追加callはない。同期profileをさらに反復して1.31%のlazy wall差を説明する根拠はない。

ここでpinned DwarfStar `8db1d1d`のouter dispatchを再照合し、より重要なschedule差を確認した。ds4は
同じchunkのencoder後にsuffixを即完了しない。`count >= 16384`かつ`remaining-count >= 8192`のときだけ
encoder-only frontierを非公開にし、次のsweepを`decoder_pending`としてexact suffixで完了する。repositoryの
`should_defer_decoder()`はこの条件を表していたが、full-context候補は使用せず、各8K以上のchunkを即時
suffix完了していた。これはexact decoder-suffix primitiveとしては合格だが、DwarfStar CED outer scheduleの
qualificationではない。

実装をpending/resumeへ更新した。新しいmove-only transactionは最初の16K encoder stateとlayer 20
producer cache/windowだけをprivateに保持し、8K--16Kの次encoder sweep後に各decoder reuse layerを127-row
warmupから再構築して一度だけpublishする。16K以下はfull decoderのままなので、旧16K paired resultは
primitiveのperformance evidenceとして保持するがproduction CED promotionには使わない。最初のgateは次。

```sh
bash tools/benchmark/run_deferred_pending_backbone_check.sh
```

scopeは24,576-token full packed sweep対16K private encoder-only + 8K exact resume。hidden/pre-mix/logits、
全persistent state/publication/hash、executed route ties、source atomicityをbit-exactで要求する。15--35分、
最大340 GB、checkpoint read-only、resumeなし。review前はdefault OFFを維持する。

pending gate artifact `prefill-gap/deferred-pending-20260920-061215-1886`をreviewした。exit 0、revision
`6c33f77`、tracked patch SHA-256
`8f74a89b96dcec73e85c3f32d370049aed8bde905ee8c42c612ddd695cc2cc53`。24,576-token full sweepに対し、
16K private encoder-only + 8K resumeのfinal hidden/pre-mix/logits、全encoder/decoder persistent
state/publication/hash、executed route tiesがbit-exactで、source revision atomicityとreuse rejectionもpass。
swap 0、maximum resident set size 58,791,247,872 bytes、peak footprint 157,712,878,200 bytesだった。
pending primitiveをpassとし、次はproduction topologyで2K/16K inactive、32K exactly one pending transactionを
測定する。

```sh
bash tools/benchmark/run_deferred_decoder_context_candidate.sh
```

runnerは各stageでCED count 0/0/1、state、生成数、40 resident banks / 15,360 experts、zero readback、
zero scalar QK/AV、swap、revision/patchを検証してからresume markerを作る。20--45分、最大340 GB、
checkpoint read-only。失敗時は表示rootを`DSV41_CED_CANDIDATE_DIR`へ渡し、検証済みstageだけskipする。

full-context artifact `context-ladder/deferred-decoder-candidate-20260920-063643-2271`をreviewした。rootと
2K/16K/32K childはexit 0、revision `6c33f77`、同一tracked patch SHA-256
`beca9c0ce6319a8de869e0ad791e1d266e13de4a4ad9d70afaebfe0a63a73407`。CED countは要求通り
0/0/1。全stageで正しい生成列/state、40 banks / 15,360 experts、route/index readback 0、scalar QK/AV 0、
swap 0を維持した。2K prefill 43.595390秒、16K 322.094514秒でinactive pathの範囲内。32Kは
cache-bounded baseline 639.081491→385.448176秒（39.69%短縮）、baseだけでは
557.213077→301.412038秒（45.91%短縮）。旧immediate-suffix候補399.417787秒よりも速い。
decode mean/p95はbaseline比0.28%/0.56%改善、MLX peakは同じ304,410,027,515 bytes、process peak
326,437,061,776 bytesで340 GB内だった。生成16 token列とstate position 32767もbaseline exact。

単回teacher continuationは81.868415→84.036138秒（2.65%退行）なので、この観測だけでdefaultへ
昇格しない。CEDが実際にactiveな32Kでwarmup後5 alternating pairsを実行する。

```sh
bash tools/benchmark/run_deferred_pending_paired_qualification.sh
```

scopeは12 sequential 32K processes、110--180分、最大340 GB、checkpoint read-only。candidate win 5/5、
順序差を超えるprefill改善、teacher/decode/memory/swap non-regression、生成/state/topology exactを要求する。
中断時は表示rootを`DSV41_PENDING_PAIRED_DIR`へ渡し、検証済みchildだけskipする。

paired artifact `context-ladder/deferred-pending-paired-20260920-121235-3691`をreviewした。runnerの最初の
pair directory作成不備はmodel外で修正し、既存3 childをrevision/patch/result/resourceから検証して同じrootで
resumeした。最終的にrootと12 childはexit 0、revision `6c33f77`、同一tracked patch SHA-256
`da650b19372d0d3817faf8dbc33b6005d057122333e4e5a5b5acb41a49b60be9`。candidateは全runで
exactly one CED、baselineは0。生成/state/topology、40 banks / 15,360 experts、zero readback、zero scalar
QK/AV、swap 0を保持した。

5 pair平均はprefill 639.677212→385.077299秒（39.80%短縮）、base 557.805715→301.192429秒
（46.00%短縮）、candidate win 5/5。各pair改善39.64--39.95%はbaseline range 1.732秒、candidate
range 1.434秒を十分超える。decode mean/p95は1.11%/1.75%改善、MLX peakは同一
304,410,027,515 bytes、process footprint平均は1.06%増の326.27 GBでbudget内。ただしteacherは
81.871496→83.884870秒（2.46%退行）かつ5/5で遅いため、事前contractをpassしない。defaultはOFF。

semantic work/call/stateがexactで同期profileでは差が消えることから、次のbounded candidateはCED commit後の
idle MLX allocator cacheだけを一度clearする。live model/persistent stateは保持する。

```sh
bash tools/benchmark/run_deferred_pending_cache_transition_candidate.sh
```

scopeは32K 1 process、8--15分、最大340 GB、checkpoint read-only。単回観測はこのtransitionをreject
できるがpromotionには使わない。teacher改善がなければ削除し、あれば同じ5-pair contractを再適用する。

cache-transition artifact `context-ladder/deferred-pending-cache-20260920-162259-6005`をreviewした。root/child
exit 0、revision `2265b63`、tracked patchはempty、cache-clear設定1、CED transaction 1。生成16 token列、
state position 32767、production topology、40 banks / 15,360 experts、zero readback、zero scalar QK/AV、
swap 0を保持した。prefillは382.591689秒で、clearなしpaired candidate平均385.077299秒より0.65%短い。
baseも300.654586秒で0.18%短い。teacherは81.937102秒となり、clearなし83.884870秒から2.32%改善し、
baseline平均81.871496秒との差は0.08%まで縮小した。単回decode mean/p95は3.213733/3.521494秒で、
paired baseline平均より0.63%/0.66%遅い。process peakは323,122,660,016 bytes、swap 0。

この観測はcache composition仮説を棄却せず、単回ではpromotionもしない。同じwarmup + alternating 5-pair
contractをcache normalization込みで再実行する。

```sh
bash tools/benchmark/run_deferred_pending_cache_paired_qualification.sh
```

scopeは12 sequential 32K processes、110--180分、最大340 GB、checkpoint read-only。candidateだけがatomic
publication後にidle allocator cacheを一度clearする。中断時は表示rootを`DSV41_PENDING_PAIRED_DIR`へ渡して
同じcommandを再実行し、revision/patch/config/result/resource検証済みchildだけをskipする。

cache-normalized paired artifact
`context-ladder/deferred-pending-cache-paired-20260920-164151-6227`をreviewした。rootとwarmupを含む12 childは
exit 0、revision `2265b63`、同一tracked patch SHA-256
`1d5353cb84a275223cb70d694f8310b2a8f5404ea3bb07d25c62692fe5a335ac`。baseline/candidateのcache-clear設定は
0/1、CED countは0/1で、全runの生成列/state/topology、40 banks / 15,360 experts、zero readback、zero scalar
QK/AV、swap 0、build/source identityが一致した。

5 pair平均はprefill 639.584333→382.164674秒（40.25%短縮）、base
557.767773→300.860958秒（46.06%短縮）でcandidate win 5/5。各pair改善40.15--40.39%はbaseline range
2.244秒、candidate range 1.556秒を十分超える。teacherは81.816560→81.303716秒（0.63%改善）、decode
mean/p95は3.203087/3.523623→3.201355/3.504421秒（0.05%/0.54%改善）。MLX peakは同一
304,410,027,515 bytes、process footprint平均は0.14%減の322,869,491,157 bytes、swap 0だった。

事前contractをpassしたためpending CEDとpost-publication idle-cache normalizationをproduction defaultへ昇格する。
明示的な0はfull-decoder comparison/oracle fallbackとして保持する。native generation APIについてはdefault値を
使用する24,576-token full-model gateを最後に実行し、fallbackとの生成列/next-position一致とresourceをreviewする。

```sh
bash tools/benchmark/run_deferred_decoder_native_api_check.sh
```

scopeは2 fresh official-model processes、12--25分、最大340 GB、checkpoint read-only。中断時は表示rootを
`DSV41_CED_API_DIR`へ渡して再実行し、検証済みstageだけをskipする。API promotionはartifact reviewまで未確定。

native API artifact `native-model-api/deferred-decoder-20260920-185438-8518`をreviewした。root、fallback、
productionはexit 0、revision `2265b63`、同一tracked patch SHA-256
`146a6d0e4a1141f66d2e2902c1063a41a0b3be8bf0f964fb045ef6cbe440981b`、同一identity SHA-256
`97a3f4d08dec8d9763d378521cfa5cd6f585139c0f0b40cd378300b173290d4d`。明示full-decoder fallbackと
no-environment production defaultはいずれも4 token `361 362 1990 295`、next position 24580、stopped falseで
一致した。両stageともswap 0。wallはfallback 525.24秒、production 307.49秒、peak footprintは
324,284,251,648→321,527,170,336 bytesだった。

これでpending CED、atomic publication後のidle-cache normalization、native generation APIのpromotion
contractは完了した。両policyはno-environment production defaultであり、明示的
`DSV41_RUNTIME_DEFERRED_DECODER=0`と`DSV41_RUNTIME_DEFERRED_DECODER_CLEAR_CACHE=0`はfull-decoder
comparison/oracle fallbackとして残る。16K以下はschedule条件により従来のfull decoderを維持する。
