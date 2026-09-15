# SWA stateとattentionの接続境界

2026-09-13。layer 0 attentionをQ/KV生成からstate、attention、出力projectionまで接続した。token逐次scheduleのlocal referenceであり、Block / MoEと公式oracle比較は未完了。

## State

`SwaReferenceState`は128×512のBF16 bit列をringに保持する。入力は公式FP8量子化・復元後のKVであり、このクラスでは追加量子化しない。ring本体は1 layerあたり131,072 bytes。append時の一時領域も最大131,072 bytesで、履歴長には依存しない。`chronological_rows()`は最大131,072 bytesの独立したsnapshotを返す。

連続position、shape、有限値を全件検査してから更新する。大きいchunkは最後の128行だけを保存する。自身のringを入力に渡しても更新前のsnapshotと同じ結果になる。copyしたstateは独立し、resetはpositionとringを消去する。これは同期CPU referenceであり、GPU完了とsession commitの連携は未実装。

公式`get_window_topk_idxs`に対応するdecode index順序も公開する。全履歴の保存やSSD backingは行わない。Bounded Replayの再計算・復元・APCを実装したことにはならない。

## Attention

`swa_attention_reference`の入力はBF16 Q `[64,512]`、時系列順の有効KV `[1..128,512]`、FP32 sink `[64]`。公式minimal inferenceの`inference/kernel.py`にある64行単位online softmaxを演算境界として、最大値更新、分母、value積和前のBF16 cast、sinkの分母への加算を明示する。checkpoint revisionはM1 baselineのまま。

有効行だけを詰めた入力であり、公式decodeの128 slot内の`-1` paddingまで含めたblock分割とは区別する。CUDA GEMM/reductionとの一致、非自明なscoreでの丸め、paddingによるschedule差は未検証。公式式を記述したことだけでbitwise exactとは判定しない。

`SwaProjectionReference`にlayer 0の実checkpoint Q/KV projectionを接続した。Qはwq_a → q_norm → wq_b → RoPE、KVはwkv → kv_norm → RoPE → 公式FP8 activation round-trip。activationのabsmax floorは公式の1e-4を維持する。入力はBF16 `[1..128,5120]`、出力はQ `[tokens,64,512]`、KV `[tokens,512]`。共通linearのtokenごとの固定scheduleを使用する。

`swa_rope_reference`は末尾64次元を隣接complex pairとしてFP32で回転してBF16へ戻す。theta=10000、YaRNなし、positionは0..1048575。inverseも同じ境界で提供し、layerの出力projection直前に使用する。MLXのpow / sin / cosと公式PyTorchの周波数生成・complex積和とのbitwise一致は未判定。

`SwaLayerReference`がQ/KV生成 → stateへの追加 → causal window選択 → masked attention → inverse RoPE → grouped wo_a → wo_bを接続する。wo_aは公式convert.pyと同じ32×32 scale対応でcanonical FP8をBF16へ復元し、8 groupそれぞれ4096→1024を投影する。ここではwo_a入力をFP8へ量子化しない。その後8192→5120のwo_bが共通FP8 linearを使う。global KV / CSA2 / indexerは対象外。

## Layer stateとschedule

`SwaLayerState`はMLX BF16 arrayを時系列順で最大128行保持する。先のCPU `SwaReferenceState`へのKV往復は行わない。copyしたstateはimmutable arrayを共有し、更新時に新しいarrayを作る。128→129 tokenでは最古の1行を捨てる。boundedなlive stateであり、Bounded Replayやprefix reuseではない。

入力chunkは常にtoken逐次で処理する。position 0は1つの有効slot、以後は公式decodeの順序に合わせ、未使用slotを前に、有効行を後ろに置く128 slotを作る。無効slotはKV=0、score=-infとして扱う。先頭64 slotがすべて無効でも、running maxの初期値-1e30によりNaNを回避する。有効KVだけを詰める既存診断関数は残す。

公式の一括prefillはこのtoken逐次scheduleとは異なる。特に短いtopkの64行paddingやGEMM reductionまで一致したとは主張しない。chunk/tokenwiseのlocal bitwise基準を固定し、公式意味論との独立比較は別の未完了gateとする。

各tokenでGPU評価と有限値検査を完了し、chunk全体の出力評価が成功してから呼び出し元のstateをcommitする。position不一致・非有限入力は更新前に拒否する。これは同期referenceで、schedulerの非同期transactionではない。保持KVは最大128 KiBだが、旧state、評価中array、出力、weightを含む実peakは別に測定する。

## 短い検証

- ReleaseおよびASan/UBSanのCTestは4件通過。SWAでは1、127、128、129、255、256、257、1025 tokenのwrap、chunk/tokenwise一致、公式indexの選択順序、fork/reset、不正入力でのstate不変を確認した。self-alias回帰の追加後にも両buildのSWA testを再実行し、通過した。
- GPU解析解検査は通過。Q=0、KV=1、sink=0で、有効行数1、63、64、65、127、128について全出力がBF16の`N/(N+1)`と一致した。empty KVを拒否した。
- 実checkpointのattention出力比較、全layer、long-context、性能は未検証。単体検査をfull-path qualificationに数えない。
- Q/KV追加後のGPU検査も通過。RoPEの最初のpairはposition 0 / 1 / 127 / 128 / 262143 / 1048575の順逆回転で解析解とBF16数値一致。解析解のsin(0)とcomplex積和で符号付きゼロが異なり得るため、この解析解検査は数値比較とし、oracle exactnessとは呼ばない。全32 pairを含むRoPEのchunk比較はbitwise一致。
- 実layer 0 Q/KV weight、sinで生成した2 token入力、position 127→128について、有限値・chunk/tokenwise bitwise一致・owner破棄後の遅延評価のbitwise一致を確認。公式出力との比較ではない。結果とsource identityは[検査記録](../artifacts/swa/projection-check.json)に保存する。
- layer接続後、実weightで2 tokenのattention出力・KV stateのchunk/tokenwise bitwise一致、3 token目への継続、fork分離、reset再現性、不正position / NaN入力拒否を確認した。128 token投入後、129 token目で最古行だけが消え、最新行が独立Q/KV projectionとbitwise一致することも確認した。mask解析解は有効行数1 / 63 / 64 / 65 / 127 / 128で通過。これは実checkpoint oracleとの比較や長時間qualificationではない。

再実行（数秒規模、checkpoint loadなし）:

```sh
ctest --test-dir build -R swa_state_contract --output-on-failure
ctest --test-dir build-sanitize -R swa_state_contract --output-on-failure
build-mlx/dsv41-swa-attention-test
build-mlx/dsv41-swa-attention-test /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash artifacts/checkpoint/summary.json
```

GPU検査はMetal利用可能な通常の実行環境を必要とする。失敗時は非ゼロ終了し、state検査の詳細は各buildの`Testing/Temporary/LastTest.log`に残る。

layer接続後の結果と実行時source / binary hashは[layer検査記録](../artifacts/swa/layer-check.json)に保存する。先のprojection検査記録はその時点の履歴として保持する。
