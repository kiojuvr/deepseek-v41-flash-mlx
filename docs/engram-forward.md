# Engram native forward reference

状態: 2026-09-12。row IDs / hidden → packed row read → activation FP8 → projection → gate → residualをC++ / MLX / Metalで接続した。**公式oracleとの数値差が残るため未qualified。** token→logitsのモデル全体は未完成で、optimized pathへの昇格は行っていない。

## 公式意味論と訂正

固定snapshotの `inference/model.py:328` の `Engram` は、lookup、FP8 `wkv`、gate、residualから成る。**convolutionは存在しない。** 以前の設計メモにある「Engram short convolution」は本snapshotに該当しないため訂正した。forward内にconvolution stateはなく、履歴を持つのは外側の `NgramHashState`。モデルは各対象Blockに入る前にEngramを適用する。

1. 24行 × 256要素を公式BF16に復元し、6,144要素へflattenする。
2. `kernel.py` の `act_quant` に従い、各32要素のFP32 absmaxを `1e-4` 以上にclampする。`amax / 448` を上側の2冪へ丸めたE8M0 scaleを使い、activationをE4M3へround-to-nearest-evenで量子化する。これは公式projectionの意味論であり、追加のモデル量子化ではない。
3. `[25600,6144]` のE4M3 weightと `[800,192]` のE8M0 scaleを使う。32要素の積和ごとにactivation / weight scaleを適用してFP32で集積し、BF16へcastする。元weightをBF16行列へ置き換えてactivation量子化を省くことはしない。
4. 出力を4 copies分のkeyと共有valueへ分ける。`q_weight * k_weight` はFP32。hidden / keyの正規化は各copyの5,120要素に対して別々に行い、epsilonは公式configの `1e-20`。
5. 正規化dotへ `dim**-0.5` を掛け、`copysign(sqrt(max(abs(dot),1e-6)), dot)` をsigmoidへ入れる。zeroの符号も保持する。
6. maskがFalseのtokenのgateを0にし、FP32のhiddenへ `gate * value` を加えてBF16に戻す。

## Ownershipとscope

`EngramLayerReference` はbatch 1、BF16 hidden `[tokens,4,5120]`、1–128 tokensに限定する。row IDsは外部のhash stateからlayer別に24 IDs / tokenを渡す。まだmodel schedulerとの統合はない。

各layerのresident payloadは157,521,920 bytes（projection weight / scale、q / k）。2層で約300.449 MiB。巨大embedding backingは従来のmmap / preadを使い、resident projectionを再量子化・全量dequantizeしない。CPUの読み取りbufferからMLX所有arrayへコピーする。load時の一時コピーとallocator peakはこのpayload値とは別で、runtime peak qualificationは未実施。

Metal projectionは1出力 / threadの明示的な32要素FP32 reductionで、性能最適化版ではない。MLX 0.32.2のsafe math設定を使い、Metal helperにFP contractionを無効にするpragmaを付けた。CUDA tensor coreとreduction順序が同じであるとは仮定しない。

返されたgraphが入力を所有することを、layer objectを破棄してから評価するprobeで確認した。row stagingもforwardの呼び出し終了後には破棄される。全履歴のgraph保持やprefetch機構は導入していない。非有限activationはNaNを伝播する表現へ変換し、probeは非有限出力をエラーにする。productionでの数値異常・cancel処理はmodel runtime側の未実装事項。

## Oracleの境界

[manifest](../artifacts/engram-layer/manifest.json)はcheckpointの公式source、fixture、exporterのhashと環境を記録する。

- **activation / projection:** `kernel.py` の数式をCPU PyTorchへ転記した比較reference。公式TileLang / CUDA kernelは実行していない。E4M3 castはPyTorchを使い、projectionは32要素ごとのFP32 GEMM・scale適用を行う。この一致だけで公式CUDA projectionとの一致を宣言しない。
- **gate / residual:** AST抽出した公式 `Engram.forward` を実行する。lookupとprojectionは固定の実row / CPU projection出力を返す境界に置き換え、残りの算術文は変更しない。returnにdot / gateの捕捉だけを追加する。
- **local execution:** 同じnative referenceのmmap / pread、一括 / tokenwiseをbitwise比較する。外部oracleとの差とは独立の検査。

hiddenは合成入力。両層各3 tokensにzero hidden、masked token、seed固定のrandom hiddenを含む。実encoderからのhiddenではなく、画像入力全体やn-gram mask履歴とのend-to-end試験でもない。

## 現在の結果

[native結果](../artifacts/engram-layer/native-verify.json)は `status: numerical_mismatch`、probeの終了codeは**2**。失敗を成功へ変換せず保存する。新しい許容差を設定していない。

| 比較 | 観測 |
| --- | --- |
| 全65,280 finite BF16 bit patternsを含むactivation入力 | FP8 bytes / scaleがCPU転記referenceと一致。全block組合せの網羅ではない |
| 実rowを使ったactivationとscale | 両layer、両read modeで一致 |
| projection BF16 | 各layer 76,800要素、CPU転記referenceと全一致 |
| native一括 / tokenwise | projectionとresidualの全bitsが一致 |
| native mmap / pread | residualの全bitsが一致 |
| mask Falseのhidden | 入力bitsを保持 |
| 不正shape / dtype / mask / row数・範囲 | 拒否 |
| layer 1の公式residual | 61,440要素でBF16一致。ただし中間FP32 dot / gateは不一致 |
| layer 14の公式residual | 61,440要素中1要素、BF16 1 ULP差 |

layer 14の差はtoken 2 / copy 1 / channel 2904（いずれも0起点）。native bitsは `0xb733`、公式は `0xb734`、absolute errorは `5.960464477539063e-8`。その要素のrelative errorは約0.00556であり、absolute errorが小さいことだけを理由に受け入れない。

FP32 dotはlayer 1で最大18 ULP、layer 14で最大3 ULPの差、gateはそれぞれ最大1 / 3 ULPの差がある。同じprojectionを与えたgate単独でもresidualの同じ差を再現した。さらに**公式gate値を注入すると両layerのresidual全要素が一致**する。したがって今回の不一致はgate算出側までに局在し、row I/O、projection出力、gate以後のresidualだけの問題ではない。reduction / rsqrt / sigmoidのどの演算が各差に寄与したかの完全な分離は今後行う。

次はこのFP32境界の差を診断し、公式oracle → local referenceの判定根拠を揃える。CUDA projectionの独立比較も未完了。ここまでをM2完了やruntime性能改善と扱わない。

## 再現

現workspaceのfixtureとbinaryは準備済み。今回の検査は1秒程度で終了し、ユーザー実行の長時間jobは追加していない。

```sh
cmake --build build-mlx -j 8
build-mlx/dsv41-engram-layer-probe \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --m1-summary artifacts/checkpoint/summary.json --row-fixtures artifacts/engram \
  --fixtures artifacts/engram-layer --output artifacts/engram-layer/native-verify.json
```

終了code 0はこのfixtureの比較bitsがすべて一致した場合、2は数値差、1は入力検査・I/O等の失敗。0であってもCUDA projectionやfull modelのqualificationを意味しない。現在の期待する診断結果はcode 2。

fixturesの再生成（offline専用Python、production dependencyではない）:

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/export_engram_layer_fixture.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --verification artifacts/checkpoint/verification.json \
  --m1-summary artifacts/checkpoint/summary.json --row-fixtures artifacts/engram \
  --output artifacts/engram-layer
```

2026-09-13の再実行に関するbuild / 結果identityは[診断provenance](../artifacts/engram-layer/gate-diagnostic-provenance.json)に記録する（以前のrunのprovenanceを遡及再構成したものではない）。fixture bytesはgit対象外で再生成可能。以前のrow lookupとcache比較の証拠は、その時点の履歴として保持する。

## 2026-09-13: gateの演算別診断とomlx対照

[native中間値](../artifacts/engram-layer/native-gate-trace.json)、[CPU診断](../artifacts/engram-layer/gate-diagnosis.json)、[固定omlx対照](../artifacts/engram-layer/omlx-gate-comparison.json)を追加した。以前の`native-verify.json`はそのまま保持した。runtimeのgate算術・許容値は変更していない。

| FP32比較（最大ULP） | layer 1 | layer 14 |
| --- | ---: | ---: |
| hidden mean-square | 1 | 1 |
| key mean-square | 2 | 4 |
| weighted sum | 16 | 4 |
| dot | 18 | 3 |
| 同じnative meanをCPUへ注入したrsqrt（hidden / key） | 1 / 1 | 1 / 1 |
| 公式dot注入後のsigned sqrt | 0 | 0 |
| 公式dot注入後のunmasked sigmoid | 1 | 2 |
| 同じnative signed-rootをCPUへ注入したsigmoid | 1 | 1 |

reductionだけでなく、同じ入力のrsqrt / sigmoidにもbackend間差が観測された。oracleは元の公式ASTを再実行し、既存dot / gate / residual fixtureのbits再現を確認してから比較する。CPUの中間値採取は診断であり、CUDA実行の代替ではない。

固定omlx commit `b390b31e0c6831225fed0f24d278eb1db7fcb68b`の`Engram.__call__`をAST抽出し、同じBF16 hidden / q / kと固定projectionを与えた。**両layerのdot / gateはnativeと全bit一致**。omlxにもlayer 14の同じflat index 48984で`0xb733`対CPU `0xb734`のresidual差が出た。このfixtureでnative固有の差ではないことを支持するが、omlx全体の精度・本runtimeのM2合格を証明しない。

数値差をCPU PyTorchとMLXの演算差として追跡し、omlxへの追従だけを目的にgateを変更しない。独立した[モデル入口・RMSNorm](model-entry.md)の接続を進めた。CUDA projectionと全モデルの比較は引き続き未実行。

### 診断の再現

いずれも既存fixtureを使う短い検査。native probeは前述commandのoutputだけを`artifacts/engram-layer/native-gate-trace.json`へ変更し、終了code 2の結果を保持する。

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/diagnose_engram_gate.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --fixtures artifacts/engram-layer --native-result artifacts/engram-layer/native-gate-trace.json \
  --output artifacts/engram-layer/gate-diagnosis.json
/Volumes/SDXC-512/glm53-flash-mlx/.venv/bin/python tools/reference/compare_omlx_engram_gate.py \
  --omlx /private/tmp/dsv41-omlx-0.7.0.dev2 --fixtures artifacts/engram-layer \
  --native-result artifacts/engram-layer/native-gate-trace.json \
  --output artifacts/engram-layer/omlx-gate-comparison.json
```

Python / MLXはこのoffline診断にのみ使用する。omlxモデル全体・依存packageのインストールは不要だが、固定source checkoutとMLX 0.32.2、CPU oracle側にはPyTorchが必要。診断scriptのcode 0は診断完了であり数値一致を意味しない。[今回のprovenance](../artifacts/engram-layer/gate-diagnostic-provenance.json)を参照。
