# 圧縮attentionの接続状況

2026-09-13。layer 2のratio-2 `CompressorReference`を追加した。公式checkpointのBF16 wkv / wgateをFP32へ昇格し、tokenごとの固定scheduleでprojectionする。2 tokenごとにchannel別softmaxでpoolし、BF16 cast後に公式norm weightを使う。出力はindexerが消費するRoPE未適用latent。group先頭のposition（0、2、4…）を併せて返す。

未完了groupは最大1行のFP32 KVとscoreを保持する（payload計4096 bytes）。完了後は空に戻す。公式の2 slot配列に残る使用済み行は保持しない。前回groupの行を参照しない意味論を維持し、公式state buffer全体とのbit一致は主張しない。呼び出し全体の評価成功後にstateをcommitする。

## 生成・再利用の境界

固定公式`inference/config.json`ではKV producerは2 / 8 / 14 / 20、index producerは2 / 8 / 14 / 20 / 24 / 28 / 32 / 36。layer 2〜19はratio 2、20〜39はratio 1。candidate producerは20。これらを単純なlayerごとの独立KV cacheへ置換しない。

`IndexKeyReference::before_quantization`でlatent → BF16 wk projection → k_norm → compressed RoPEを接続した。出力はFP4量子化前のBF16中間値であり、canonical index cacheへ保存してはいけない。index用FP4はgroup 32 / E8M0 scale、main KVはgroup 16 / E4M3 scaleであり、両形式の`kv_quant_reference`を追加したが、cacheへのpublicationはまだ残る。

`compressed_rope_reference`はtheta=160000、original=65536、factor=16、beta_fast/slow=32/1の公式YaRN式を使う。128次元のindexと512次元のmain latentそれぞれ末尾64次元を回転し、明示position列でgroup先頭位置を受け取る。pure SWA用のtheta=10000・YaRNなしRoPEは使わない。indexerはRoPE前のlatentを必要とするため、その生成をmain KVの回転・量子化より前に置く。global KVはUnified Memory residentの契約を維持する。

layer 2のcompressorからindex/main FP4 cacheへの保存まで接続した。layer 2 Block、candidate selection、他layerへのshared cache公開、decoderのencoder最終状態からのKV投影はまだ接続していない。

## 短い検証

実layer 2 weightで3 token chunkと逐次処理のlatent bit一致、pending KV / scoreのFP32 bit一致を確認した。4 token目への継続とgroup先頭position、fork分離、reset、不正position / NaN入力拒否、zero入力のzero latentも通過。既存layer 0/1の短い検査も通過した。

これは公式CUDAや独立CPU oracleとの比較ではない。RMSNormの既知差も未解決のまま引き継ぐ。full-path / long-context qualificationではない。

追加の短いGPU検査で、compressed RoPEの位置0 / 65536 / 262143におけるchunk/tokenwise bit一致、非回転部分の不変、position 0での数値同一、範囲外position拒否を確認した。実compressor latentから実index wk / normを使う量子化前keyは有限値・繰り返しbit一致を確認した。YaRN周波数や非ゼロ位置での回転を独立oracleと比較した検査ではない。rank-3 queryとinverse側も未検証。

再現コマンド（短い検査、全expertはloadしない）:

```sh
build-mlx/dsv41-swa-attention-test /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash artifacts/checkpoint/summary.json
```

## FP4量子化reference

`kv_quant_reference`はindex `[1..128,128]` またはmain KV `[1..128,512]` のBF16入力を受け、packed E2M1 bytes、scale bytes、BF16復元値を返す。`metal/attention/kv_quant.metal`でgroupごとのabsmax、scale生成、±6 clamp、nearest-even FP4丸め、復元のBF16 castを明示する。先行要素をlow nibbleへ置くlocal packingであり、公式packed tensorとのbyte照合は未実施。

indexはabsmaxのfloorが6×2^-126、scaleはamax×(1/6)のlog2切り上げ。mainはfloorが6×2^-9、scaleはamax/6をE4M3へ丸める。ゼロgroupのscale codeは両形式とも1だが、表す値は異なる。expert用FP4 weight packingやFP8 activationの1e-4 floorとは混同しない。

短いMetal検査で正負の全7 midpointのnearest-even code、unit scale、packing、ゼロgroupの最小scaleと復元、chunk/tokenwise復元bit一致、NaN拒否を確認した。既存の実weight検査も通過。独立CUDA oracle、全BF16値、scale丸め境界を網羅した検査ではなく、optimized pathへの昇格には使わない。

mainのamax/6が448を超える場合は保守的にエラーとし、公式castのoverflow領域との同一動作はまだ実装しない。scaleを黙って飽和させることはしない。非有限入力もエラー。結果返却前にerror flagのGPU評価を同期するreferenceである。量子化済みKVをさらに量子化する経路ではなく、公式の初回量子化境界を実装している。

## Layer 2 global cacheへの保存

`GlobalKVProducerReference::append`はlayer 2 attentionのnormalized hiddenを受け、compressor → index wk/norm/RoPE/FP4と、同じRoPE前latent → main RoPE/FP4を接続する。layer 1出力やembeddingをそのまま渡すAPIではない。

`GlobalKVState`はcompressorのpending stateと、main packed `[rows,256]` / scale `[rows,32]`、index packed `[rows,64]` / scale `[rows,4]` のuint8 MLX arrayを保持する。persistent BF16復元値やSSD backingは持たない。1 completed groupあたり356 bytes、ratio 2のlayer 2について平均178 bytes/input token（allocator等を除く）。全producerの890 bytes/token見積りとは対象範囲が異なる。

row jはtoken position 2jのgroupを表し、そのgroupの2 token目が完了して初めて公開する。append開始時にpositionとrows=floor(position/2)を検査し、生成groupの連続性も検査する。作業用stateで両cacheの評価を完了してからcompressorとともにcommitする。forkはimmutable MLX arrayを共有し、appendで新しいarrayを作る。

現referenceはappendのたびにarrayを連結するため、履歴長に比例したcopy costがある。long-context向けallocatorや固定capacityの書き込みを実装したことにはならない。chunkごとのallocation量や全path性能の昇格判定にも使わない。

実layer 2 weight、sinで作った5 token入力で、chunk/tokenwiseの両packed bytes・scale bytes・pending compressor bitsの一致を確認した。途中groupでの未公開、6 token目の公開、fork分離、reset、不正position / NaN入力拒否も通過した。index量子化後やmain保存途中への障害注入はまだ行っていない。公式独立oracleとの比較は引き続き未完了。

## Layer 2 index query / score / Top-K

`IndexQueryReference`に実FP8 wq_bとBF16 weights_projを接続した。入力はattentionのnormalized hidden xとnormalized wq_a出力qr。queryはprojection → compressed RoPE → group-32 FP4 round-tripを経る。保存済みindex keyはpacked nibbleとE8M0 scaleからBF16へ復元し、再量子化しない。

scoreはquery-key積和 → ReLU → weights_proj出力（128^-0.5 × 32^-0.5倍）との積 → head方向のsum。公式のBF16境界を明示するが、MLX積和 / reductionと公式のbit一致は未検証。decode一token専用で、cache position=query position+1、rows=floor((position+1)/2)を要求する。未完了groupや未来位置のkeyを参照しない。空cacheは空候補を返す。

Top-Kは最大512をscoreで選び、position順へ並べ直してwindow offsetを加える。CPU同期とsortを使うreferenceで、選択境界が同点なら公式tie oracle未確定として拒否する。layer 20以降のcandidate filterや一括prefillのmaskはまだない。現在はindex key全履歴を一時BF16へ復元するため、long-contextのworking set最適化も未完了。

短い検査で513要素の既知scoreから512候補を選ぶ境界・offset・position順、空score、同点境界拒否を確認した。実query weightと2行の実packed cacheについて、到達可能な2候補を返し、未来cacheを拒否した。qrは合成入力であり、attention wq_aとはまだ接続していない。2候補は全件選ばれるため、この実weight検査はscore順位の正しさを証明しない。数値oracleとlayer 2 Attentionへの接続が次の条件。

## Layer 2 Attention接続

`CompressedLayerReference`がnormalized attention inputから出力まで接続する。wq_a / q_normのqrをmain queryとindex queryが共有し、window KVにはcompressed RoPE後の公式FP8 round-tripを適用する。global producerを現在tokenまで進めてからindex queryを実行し、到達可能なmain KVの選択行だけをBF16へ復元する。復元はE2M1 nibbleとE4M3 scaleの積であり、再量子化しない。

128 window slotと最大512 global slotを一つのonline softmax / sink計算へ渡す。その後inverse compressed RoPE → grouped wo_a → FP8 wo_b。既存masked attention境界を最大640 slotへ拡張し、pure SWAの有効行だけを受けるwrapperは128行制限を維持する。

`CompressedLayerState`はwindowとGlobalKVStateを持つ。chunkの作業用copyで両方を進め、全出力を評価してからcommitする。実layer 2 weight、合成5 token入力で出力、window、両packed global cache、pending compressorのchunk/tokenwise bit一致を確認した。6 token目への継続、fork分離、reset、position / NaN拒否も通過。[検査記録](../artifacts/global-kv/attention-check.json)にsource / binary hashを保存する。

この検査ではglobal候補数が512以下なので全候補が選ばれる。大規模候補の順位、公式の一括prefill reductionとの一致、選択行復元の独立oracle、実行途中の障害注入は未検証。layer 2 mHC / MoE / Blockとtext経路への接続は完了し、[3層ユーザー検査](text-triple-validation.md)もレビュー済み。layer 3以降のcache再利用は次段階。連結copyやindex全履歴の一時復元を使うreferenceであり、long-context性能の候補へ昇格させない。

## Layer 2からの再利用snapshot

`SharedAttentionReference`でlayer 2のglobal cache、位置順のTop-K候補、query positionを一緒に保持する。layer 2 Attentionが出力評価に成功したtokenだけをpublicationへ保存する。consumerはlayer 3〜7かつ同じquery position、同じwindow offsetでなければ拒否する。layer 8は次のproducerなのでこのsnapshotを受け取れない。

候補数はmin(512,completed groups)、候補は一意・昇順・cache内であることを確認する。cacheはimmutable MLX arrayを共有するため、その後producerを進めても以前のsnapshotは変化しない。request間のsnapshot取り違え防止は呼び出し側のownership契約であり、この型はrequest IDを検証しない。

短い実weight検査でlayer 3〜7への候補取得、layer 2 / 8および別token位置の拒否、producer継続後の旧snapshotの位置・行数保持を確認した。layer 3の実attention接続は以下の追記を参照。

現在はchunk最後のpublicationだけをstateに残す。layer 3以降をつなぐ際は、tokenごとにproducerとconsumerを進める実行順序へする必要がある。layer 2でchunk全体を進めた後、その最後の候補を過去tokenへ配ることは禁止する。短いlocal検査を既存3層ユーザー検査の新binaryでの再実行とみなさない。

## Layer 3 consumer Attention

`ReusedLayerReference`がlayer 3の実Q/KV、norm、sink、wo_a / wo_bをloadし、一tokenのnormalized hiddenとlayer 2 publicationを受け取る。compressor、global cache、index queryを自分で生成せず、publicationの候補を使う。自身のwindowはcompressed RoPE後に公式FP8 round-tripし、選択global行と同じattentionへ渡す。出力評価成功後に自身のwindow / positionだけを更新する。

復元・grouped projectionのhelperはlayer 2と共通。global keyはpublicationのimmutable cacheから読み、自身のstateにはwindow以外のKVを保存しない。呼び出しごとにconsumer positionとpublication query positionを照合する。

短いGPU検査でlayer 2 / 3の実attention weightと合成normalized inputを使い、3 token分のsnapshot生成とconsumer実行を確認した。保存済みsnapshotによるreset後の出力・window bit再現、4 token目のfork出力bit一致、古いtokenのpublication拒否も通過。producer bytesを変更していないことを確認した。[検査記録](../artifacts/global-kv/consumer-check.json)にbinary / source hashを保存した。

検査ではlayer 2と3へ独立した合成attention入力を渡しており、layer 2 Block出力からlayer 3 Blockへの接続ではない。その後layer 3のmHC / MoE / Blockとtext経路を接続し、[4層ユーザー検査](text-quad-validation.md)のlocal整合性通過を確認した。layer 4〜7の一般化、公式oracle比較は未完了。

## Consumer layer 3〜7の一般化

`ReusedLayerReference`と`ReusedBlockReference`はlayer引数（既定3）で3〜7を選択する。各layerのQ/KV・norm・sink・出力projectionとmHC / MoEをそのownerから読む。MoE ownerも0〜7へ拡張した。pure SWAの0/1制限は維持し、consumerでlayer 2・8以降を指定するとload前に拒否する。

layer 4〜7それぞれの実attention weightで、同じlayer 2の3 token分publicationを独立した2つのconsumer stateへ与え、出力・windowのbit一致を確認した。既存layer 3とproducer側の短い検査も通過。layer 8拒否も確認した。[検査記録](../artifacts/global-kv/consumer-range-check.json)。

この検査は各layerに合成normalized inputを直接与えるもので、layer 3→4→5→6→7のBlock連鎖ではない。追加層のresident MoE / Blockはbuild確認のみ。4層ユーザー検査は一般化前binaryの証拠として保持する。次はtoken逐次の8層text接続と、その後のlayer 8 producer切り替え。layer 8のglobal cacheをlayer 2のpublicationへ混ぜない。
