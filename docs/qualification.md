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

40層resident atlas候補は`DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=1`で、初回使用後のimmutable bankを
chunk境界で解放しない。packed bank無効時は設定を拒否する。layer 0 / 128-token probeでは明示的な
release要求後もbankが保持され、serialとのfull MoE bitsが一致した。40層2×128のload / reuse / memory
検査は次をユーザーが実行する。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_resident_atlas_prefill_check.sh
```

1 model、40 layer、2×128 prefillとsample 1 token。chunk 0で40 bankを最終bufferへ直接構築し、
chunk 1で再利用する。resultの`packed_expert_bank_constructions`がexactly 40でなければ失敗する。
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
