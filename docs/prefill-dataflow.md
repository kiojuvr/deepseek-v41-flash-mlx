# Prefill dataflow redesign

状態: 2026-09-16の構造見直し。現在のtoken-serial referenceとlayer-major packed-bank経路は
correctness oracleとして保持するが、どちらもM3 performance candidateにはしない。

## 判断の根拠

報告済みoMLX 32K prefillは193.6 token/sである。一方、repo canonicalの2,048-token診断は
5.7553 token/s、layer-major 256-token診断は1.97410 token/sだった。context、warm state、
load範囲を揃えたpaired comparisonではないため倍率をqualification値には使わないが、差を
micro-kernelだけで埋める前提は棄却する。

layer-major経路は1層7,219,445,760 bytesの全expert bankを128 tokenごとに構築・解放する。
40層・32Kでは約73.9 TBの論理checkpoint readになる。さらに現状はattentionをtokenごとに
進め、routing scoreとindex selectionをhostへ同期し、token出力をstack / concatenateしている。
これはpacked MoEの局所的な高速化より上位のdataflow costである。

固定oMLX sourceには、chunk-wide packed CSA2、device上のrouting sort、grouped expert native
dispatch、compile境界、capacity-backed cache、Engram prefetchが存在する。C++へ移したことや
`gather_qmm`を使ったことだけを性能優位の根拠にしない。

## Prefill execution plan

prefillは最大128 tokenのtileを単位にし、各layerを次の順で進める。

1. capacityを予約済みのSWA / compressed KVへtileを直接書き、causal maskを保ったpacked
   attentionを1回のexecution boundaryで実行する。tokenごとの`mx::eval`、cache concat、
   全履歴のdequantizeを行わない。
2. attention後の全hidden rowからgate scoreを一括計算する。Top-K、tie検出、route weight、
   token/expert assignmentのsortをdevice上で完了し、通常経路ではrouteごとのhost同期をしない。
3. 確定したassignmentをexpert-majorに並べ、resident packed weightへgrouped dispatchする。
   expert ID昇順のFP32 reductionと既存のactivation量子化境界を保つ。全assignmentの一時的な
   unpermuted outputを作らず、最終token rowへreduceする。
4. attention、shared expert、routed expert、mHCのscratchをlayer間で再利用する。session stateは
   tile全体が成功しGPU completionが確認された後だけcommitする。

これはlayer-major correctness scheduleの単純な延長ではない。prefill / decode / replayは別の
execution planを持ち、decodeは1 token用の常駐weight・固定shape graphを使う。

reference pathの1-row reduction scheduleとbitwise parityはcorrectness oracleとして保持するが、
optimized prefill candidateの制約にはしない。candidateではtoken dimensionをbatchのまま保持し、
FP8/FP4精度とstate semanticsを維持した上でreduction orderの差を別gate（logit誤差、routing、生成、
continuation）で評価する。現行128-token chunkのlazy-eval変更は2064-token測定で約0.1%の差に留まり、
主要ボトルネックとはみなさない。次はlayer wall、QMMのM形状、submission/readback、graph materialization、
Engram stallを同一runで収集し、under-batchingとserializationを先に除去する。

## Expert residencyと先読み

第一candidateはbackbone / MoEを最終packed layoutのままUnified Memoryへ一度だけresident化し、
SSD runtime I/OをEngram backingへ限定する。40層のpacked routed expert payloadは
288,777,830,400 bytesであり、報告済みoMLX peak約292.8 GiBと同じ規模にある。ただし既存の
40-bank probeは比較用model、一時copy、allocator cacheを重ねてmemory pressureを起こしたため、
既存`PackedExpertBank`を40個保持する実装は採用しない。checkpointから最終所有bufferへ直接loadし、
source copy、per-expert object、repack duplicateを持たないresident atlasが必要である。

容量上resident atlasが成立しない場合だけ、layerまたはexpert slotをbounded resident setにする。
その場合もgate確定後のexact expert setを非同期にprefetchし、readyなshared expertや別streamの仕事と
重ねる。missは必ずcanonical weightを待ち、zero・近似expert・route変更で代用しない。

同じlayerのrouteはattention後のhiddenに依存するため、事前に正確には決まらない。予測を使えるのは
次layer / 次tileのI/O先読みだけであり、計算結果を決めるものではない。導入する場合はcoverage、
precision、余分なread bytes、eviction、stall削減を記録し、予測失敗時にもbit-exactなfallbackを使う。

## 実装順とpromotion gate

1. batched gate / deterministic Top-K / assignment sortをdevice residentにし、1 layer / 128 tokenで
   既存route ID、weight bits、tie record、MoE出力と比較する。
2. duplicateなしresident expert atlasと所有権を実装し、load後steady-state bytes、source read bytes、
   compression / swap、全backboneの短いbit parityを確認する。
3. SWA、compressed producer、reuse consumerをchunk-wide packed attentionへ置換し、state / publication /
   logitsのchunk-token exactnessを通す。
4. prefill execution planへ統合し、GPU command数、host sync数、checkpoint / Engram read bytes、各phase
   wall、overlap率、scratch high-water markを保存する。
5. 短いpaired oMLX比較で構造改善を確認してから32K runnerを再開する。

32K compact-only実測 `context-ladder/32k-run-20260916-133920-19687`（4 I/O worker）は、28656 base +
4096 teacher + 16 decodeを3977.45秒で完走した。prefillは3960.64秒（8.2694 token/s）、bank construction
10840、loaded expert 1,076,905、active/cache/peakは11.48/8.43/15.77 GB、swap 0だった。compact bankと
cache上限はメモリ面では成立するが、1M超のexpert loadと約66分のI/O/dispatchがボトルネックであり、32K性能を
qualifyしない。次はchunkごとのroute union重複（unique、前chunk overlap、完全再利用）を記録し、重複が実測で
十分な場合だけoMLX型のbounded reuse / 次chunk row prefetchへ進む。同一layerのrouteはhidden依存なので、予測は
I/Oヒントに限定し、数値経路は常にcanonical fallbackを保持する。

完全一致unionについては、既存bankをそのまま再利用する短絡を追加した。部分overlapでは従来どおり
canonicalな新bankを構築するため、現時点で数値・所有権のリスクを増やさない。

単独kernelの速度、bank構築後だけのwarm値、I/Oを除いたGPU時間ではpromotionしない。長時間runは
再現script、scope / resource、log path、failure handling、可能なresume境界を用意してユーザーへ渡す。

2026-09-16進捗: gateの128 score rowを1つのlazy graphへ積み、route scoreのhost同期をtokenごとの
128回からtileごとの1回へ統合した。2-D gate matmulへの単純置換は実checkpointのrouted outputで
bit mismatchを起こしたためrejectし、canonicalなone-row reduction scheduleを維持している。
Top-7、expert assignment、expert ID昇順の6-route reductionはdevice kernelへ移した。通常演算は
device ID arrayを直接`gather_qmm`へ渡し、hostへはtile単位のtie / error監査だけを返す。
layer 0 / 128 tokenのfull MoE短縮検査ではserial individual expertとのshared / routed / totalがbit一致し、
bank warm後のmedianは170.790208 msから23.039500 ms（7.4129倍）だった。この値は1 layerの
bounded probeであり、resident atlas、attention、full-path performanceをqualifyしない。

resident atlasのload pathはMLX allocatorの最終所有bufferを先に確保し、各expert tensorをそこへ
直接`pread`する。従来の巨大なhost `std::vector`からMLX bufferへの再copyを除去した。layer 0の
reviewed probe `expert-bank/run-20260916-083356-14519`では7,219,445,760-byte bankの全比較bitsが一致し、
構築1.9446秒から0.6915秒、最大RSS 9,855,123,456 bytesから7,438,286,848 bytesへ減少した。
単一run比較なので速度閾値には使わないが、payloadとほぼ同じsteady allocationで構築できるownership
gateとして採用する。`DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS=1`では
`TextBackboneReference` construction中に全40層をprivate atlasへ構築し、全層成功後だけmodelへpublishする。
各MoEは同じimmutable atlasを共有し、chunk境界のrelease要求はno-opになる。失敗時は未publish objectの
destructorが完成済みbankを解放する。token-serial / compact pathのownershipは変更しない。

旧lazy-resident 40層run `context-ladder/32k-run-20260916-083752-15291`ではbank constructionがexactly 40、chunk 0
52.155秒、bank再構築のないchunk 1は20.367秒（6.2847 token/s）だった。prefill全体は72.528秒、
3.5297 token/s。最終MLX activeは300,232,626,504 bytes、最大RSS 274,643,566,592 bytes、peak
footprint 304,490,591,320 bytes。swapは0だが、OS compression / decompressionは約44 GiBずつ発生した。
resident atlasによってchunkごとのcheckpoint readは除去できたが、oMLXとの差の主因は残る
token-serial attention / dispatchであり、memory headroomも未qualifiedである。

model-initialization atlasへの移行後はcontext runnerがmodel constructionのbank count/read timeを別記録し、
各warm chunkの`bank_constructions`を0として検査する。新ownershipのfresh 40層runは未判定であり、上記
旧runは数値・memoryの参考値に留める。

次段としてpure SWA、compressed producer、reuse consumerのprojectionをtile単位へまとめ、attention
内部のtoken順・causal state・index publicationは維持しつつper-token output同期を除去した。短い
実checkpoint検査ではSWA、producer、reuseのoutput / window / global state / publicationがtoken-serial
経路とbit一致した。これはchunk-wide packed attention kernelそのものではなく、その前段のexecution
boundaryである。40層2×128 promotion run
`expert-bank/layer-major-backbone-20260916-084839-16111`でhidden / pre-mix / logits / state /
publication / hash / route ties / invalid-token atomicityがtoken-serial経路とbit一致したため、
full pathのcorrectness gateはpassとする。二重modelと80 bank構築を含むためこのrunのwallは
performance evidenceに使わず、resident atlasでsteady chunkを再測する。

producer compressorはtokenごとのprojection / eval / recurrent concatenateを廃止した。ratio 2では
chunkの全row projectionを単一lazy graphへ積み、開始時の未完了1 rowを先頭へ結合して2-row poolを
まとめ、ratio 1もcommit境界まで評価を遅延する。matmulのone-row reduction geometry、groupごとの
softmax / RMSNorm、BF16化位置はreference bit parityのため維持した。最終packed cacheを一度だけcommitし、
各token publicationにはcache prefix viewと正確なpending compressor stateを渡す。

index keyも全completed groupをcommit前にまとめてmaterializeする。index queryはQ / weight projectionの
reference reductionを維持しながら、chunk全queryと最終index Kのscore、causal prefix mask、32-head
weighted sumを一つのGPU graphで計算する。causal mask後のcandidate block選択とTop-KはGPU stable
argsortへ移した。score降順はscoreの符号反転に対するstable昇順とし、同点では元のrow順、すなわち
最小IDを選ぶ。選択IDはdevice上でposition順へ再sortする。hostへ戻すのは最大513 ID / score、error、
publicationが保持するcandidate maskだけで、全scoreは同期しない。

専用gateでは512-row継続stateから128-token chunkを進め、全128 publication prefix、最終state、
640-row score、80 block中64 blockのcandidate source、layer 24 candidate consumer、512-row Top-K、
全score同値時の最低ID tieをtoken-serial CPU referenceとbit比較した。exit 0、実行部1.43秒、最大RSS
150,683,648 bytes、peak footprint 566,838,088 bytes。8-token・layers 0..2の実checkpoint probeでも
hidden / pre-mix / state / publicationがbit一致した。40層、performance、long-contextは未qualified。

40層2×128 fresh promotion `expert-bank/compact-layer-major-20260916-222621-27115`もexit 0で、hidden /
pre-mix / logits / 全state / publication / hash / route ties / invalid-token atomicityがindividual referenceと
bit一致した。VM compression増分0、decompression 5 pages、swap 0。40層correctness gateはpassしたが、
比較用二重modelと80 bank構築を含むwall / memoryはperformance根拠にしない。次はcompact-only 2063-token
prefillを`run_device_index_prefill_measurement.sh`で再測し、全score host同期除去のfull-path効果を確認する。

compact-only再測 `context-ladder/32k-run-20260916-223119-27624`は226.851秒、9.0941 token/sで、同条件の
変更前243.880秒、8.4591 token/sに対してwall 6.98%短縮、throughput 7.51%上昇した。route union、
loaded experts、read calls、tie countsは一致し、bank totalも42.998秒から42.239秒への差だけだったため、
device index execution boundaryのbounded full-path改善として採用する。単発観測でありperformance
qualificationではない。最初のchunkだけcacheが約79.9 GiBへ達したため、8 GiB limitをchunk後でなく
model構築前から適用するrunner修正とfresh memory checkを次に行う。

修正後の128-token full-path check `context-ladder/32k-run-20260916-223749-27811`では、最初のchunk
cache 7.92 GB、decode後8.56 GB、compression / decompression / swap増分0となり、設定した8 GiB境界を
最初から維持した。prefillは13.604秒、9.4087 token/s。これはcache lifecycleのbounded gateであり、
2063-token performanceや32Kを代替しない。

full 2063-token再測 `context-ladder/32k-run-20260916-223941-28024`でも、全progress sampleは
約7.38--7.99 GiB、prefill終了時8,582,960,092 bytes、decode後8,432,915,464 bytesで8 GiB設定内、
compression / decompression / swap増分0だった。最大RSSは22,865,559,552 bytes。prefillは
226.037秒、9.1268 token/sで、cache修正前device-index runからwall 0.36%短縮し、変更前同条件から
wall 7.32%短縮、throughput 7.89%上昇した。したがってcache修正に速度regressionはなく、device index
executionのbounded full-path改善とcache lifecycleを採用する。ただし単発観測であり、反復paired
performance、32K、外部parity、256Kは未qualifiedである。

2026-09-17の次段では、packed block 3種のmHC pre/postをtoken slice loopから`[T,4,5120]` graphへ
移した。layer 0の2x128 continuation、layers 0--2のcompressed state/publication、layers 0--3のreuse
stateまでtoken-serial pathとbit一致した。さらにfull-resident pathではroute auditのCPU readbackを
通常時0にし、device IDをstable expert-major順へ並べてgate/up/downを実行し、inverse permutation経由で
既存のexpert-ID昇順reductionを維持する。128-token / 768 assignmentのbounded fixtureは従来batchおよび
serial accumulationとbit一致した。これはexpert-major orderingの最初の接続であり、non-empty tile
work-list dispatch、40層、full-path MoE wallは未qualifiedである。

同条件component profile `context-ladder/32k-run-20260916-233434-29168`では、40層×17 chunkの
GPU-completion wallをattention / MoE / post-MoE境界で測定した。prefill 225.927秒、layer合計
225.605秒、attention 71.066秒（component合計32.9%）、MoE 106.119秒（49.1%）、post-MoE
38.743秒（17.9%）、未分類9.678秒だった。事前固定したSWA進行条件（最大項かつ40%以上）を満たさない。
したがってSWA専用Metal kernelは保留し、hardware-native scheduleの次対象をMoE pathとする。

次のpure SWA chunk候補として、token別の64-key block順を保ったままQK / AVを3-D batched
`matmul`へまとめる試作を行った。しかし同一processの短いanalytic gateでtoken別referenceとBF16 bit
不一致になった。論理block順が同じでもMLXはbatch geometryで別reduction kernelを選ぶため、この経路は
priority 2のobservable parityを満たさずrejectし、production pathは元に戻した。次のattention candidateは
汎用batched `matmul`ではなく、tokenごとに公式の512-lane reduction順と64-key online-softmax境界を固定する
専用Metal kernelとして設計する。公式precisionに基づく非bitwise gateを別途定義するまでは、この順序を
緩めない。

旧resident rerun `context-ladder/32k-run-20260916-100044-16544`はsteady chunk 17.519秒、
7.3065 token/s、約43 GiBのcompression / decompressionだった。この時点ではfull atlasを
最終構造としなかったが、その後model-initialization lifetime、chunk-wide mHC、device route、
expert-major orderingを統合したfresh run `context-ladder/32k-run-20260917-003631-31787`では
2,063-token prefill 127.717秒、16.1529 tok/sとなった。40 bankはmodel構築中だけに作られ、17 chunkの
construction/load増分0、route readback 0、生成token 339。baseline 225.927秒 / 9.1313 tok/sから
wall 43.47%短縮、throughput 1.769倍の単発full-path observationである。

peak footprint 305,189,124,288 bytes、swap 0だが、compression / decompressionは約32.6 / 32.5 GiB。
したがって512 GiB targetのresident atlasをproduction candidateへ戻す一方、memory pressureは明示的な
gateのままとする。compact route-first経路は低memory fallback/referenceとして保持する。次はresident条件で
component wallを再測し、旧compact component profileから新しいbottleneckを差し引き推定しない。

resident component profile `context-ladder/32k-run-20260917-005021-32331`はexit 0、clean commit
`7567138`、identity一致。prefill 126.641秒 / 16.290 tok/s、layer合計126.130秒のうちattention
69.652秒（55.22%）、MoE 53.882秒（42.72%）、post-MoE 2.390秒（1.90%）だった。model-init bank 40、
warm construction 0、route readback 0を再確認した。したがってPhase 2のtoken post-loop除去は支配項から外れ、
次のfull-path対象はattention/index/publicationである。

同じ128-token / 768-assignment形状でDwarfStar型dynamic tileとoMLX grouped MXFP4 primitiveを短時間比較した。
前者は数値不一致かつ低速、後者もcurrent expert-major gather-QMMの9.57 ms（paired 9.55 ms）に対し
11.15 ms（paired 10.98 ms）であり、いずれもproductionへ採用しない。局所QMMのためにPhase 4を遅らせない。

Phase 4の第一段として、index top-k relative rowとcandidate maskをdevice arrayのままpublicationへ渡し、
後段index sourceのrepublish、candidate consumer、main-KV gatherまでhost vectorを介さない経路を追加した。
`DSV41_RUNTIME_INDEX_DIAGNOSTICS=1`はreference/tie診断用のhost materializationを保持し、resident runnerは
0にしてreadback telemetryが0でなければ失敗する。token別attention本体とchunk frontierのatomic commitは
未完了であり、この変更だけでPhase 4完了とはしない。

commit `9bfeb71`のfull-path run `context-ladder/32k-run-20260917-084044-36764`はexit 0、tracked
patch 0、identity全件一致。prefill 129.090秒 / 15.981 tok/s、生成token 339、model-init bank 40、warm
construction 0、route/index readbackはともに0だった。直前126.641秒より1.93%遅く、単発差を改善とは
扱わない。MLX peak 302,822,401,315、process peak footprint 305,053,726,576 bytes、swap 0、
compression/decompression約30.80/30.66 GiB。このrunはhost境界除去を確認したがperformance gateではない。

DwarfStarの現`ds41_attention_batch`はpublication/indexをbatch化し、raw/mixed/indexed attentionをbatch
kernelで処理して最後にring frontierをcommitする。対応fixtureはstate/indexをexact、attention outputを
relative RMS `<0.002`で判定する。この境界を採用し、reuse layerの可変長KVを共通padded tensorへ組み、
64-key online-softmax blockと公式BF16 probability castを維持するopt-in chunk graphを追加した。3-token
fixtureではdevice publication併用時もoutput RMS gateとstate/continuation/fork/resetを通過した。40層へ
昇格する前に2×128でroute/index/state/publication exactとhidden/pre-mix/logits RMS/argmaxを確認する。

最初の40層gate `attention/chunk-backbone-20260917-085608-37522`はclean commit `62517e2`、identity一致、
tracked patch 0だったが、chunk 0 hiddenのrelative RMS `0.00554266`（max abs `2048`、mean abs
`1.02469`、BF16 mismatch 2,595,308）で事前固定した`0.002`を超えたためrejectした。state/route判定より
前に停止しておりperformance測定には使わない。rank-3 padded MLX matmul候補は既定offのままproductionへ
昇格しない。scalar operatorのMLX `vmap`も短時間fixtureでvectorization未対応となったため、次はprofileで
正当化済みのattention専用kernel境界とchunk末尾atomic frontier commitをDwarfStarから適応する。

新候補は1 threadgroupで1 token×8 headを担当し、KV rowを8 head間で共有する。referenceの64-row
online-softmax block、BF16 probability丸め、sink順を保持する。完全paddingの先頭64-row blockで
`-inf - -inf`がNaNになる初期実装は、referenceと同じmaximum `-1e30`へ直した。masked synthetic fixtureと
公式checkpointの3-token reuse output/state/continuation fixtureはpass。既定offのまま、同じ2×128 / 40層
runnerで昇格判定する。

2×128 compact promotionでは実route unionの合計loaded expertが10,543（80 bank constructions、
individual / compact比較の合計）で、full bankの30,720 expert loadの34.3%だった。
ただし比較用individual modelのcacheが同一processに残るため、測定の156 GB cacheは
compact単独の上限ではない。compact-onlyのI/Oとallocator cacheの分離が次の課題である。

oMLXの`prefill_boundaries.py`、`scheduler.py`、DeepSeek V4.1の`moe_offload.py` / `storage.py` /
`packed_attention.py`を設計の基準として参照する。chunkのcache block境界、prefill transientの
admission価格、allocation時のcache回収、SSD rowの先読みを汎用設計として取り入れる。
`DSV41_RUNTIME_MLX_CACHE_LIMIT_BYTES` はchunk境界でMLX cache limitを更新するopt-in測定で、
既定0のためactive stateとcorrectnessは変更しない。

8-worker I/O試験では1-worker比でprefill wallが321.950秒から255.764秒へ短縮し、
8.066 token/sに達した。loaded expert数は不変、cacheは8 GiB上限、swapは0。sys timeと
involuntary context switchは増えたため8 workerを最適値と断定せず、スイープ測定を残す。
