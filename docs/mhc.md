# mHC reference境界

2026-09-13。`HCReference::mixes`、`hc_pre_reference`、`hc_post_reference`をC++ / MLXで追加した。batch one、BF16 `[1..128,4,5120]`。実checkpointの各layerのattention / FFN用FP32 projection・scale・baseをresidentに読む。layer 0–39に限定する。

mix projectionはchunk長によらず1 rowずつ行い、4 copiesをflattenした20,480要素全体のmean squareで正規化する。公式configに固定したnorm epsilon 1e-20、hc epsilon 1e-6、20 Sinkhorn iterationsを使う。softmax + epsilon、最初のcolumn normalization、その後19回のrow / column normalizationという公式順序を保持する。

hc_preはcopy方向でFP32積和してBF16へcastする。hc_postはcomb[source,destination]のsource方向を縮約し、post×sublayer出力とFP32で加算してBF16に戻す。新しい融合Metal kernelへの最適化はしていない。

## 結果と限界

[fixture identity](../artifacts/mhc/manifest.json)、[結果](../artifacts/mhc/native-verify.json)を保存した。実layer-0のattention / FFN両weightと、実embeddingからcopyごとに振幅を変えた2 tokensを使用する。CPU側は公式式のPyTorch転記であり、公式CUDA Sinkhorn kernelは実行していない。

- 一括 / tokenwiseのpre・post・combがnative内でbitwise一致。
- ownerを破棄してから残りのgraphを評価できることを確認。
- CPU転記とのBF16 collapse 20,480要素、展開出力81,920要素が全一致。
- FP32係数はattention pre 7/8、post 4/8、comb 22/32、FFN pre 3/8、post 5/8、comb 28/32でbitが異なる。最大absolute差は1.7881393432617188e-7。

係数も比較境界なのでstatusは`numerical_mismatch`、終了codeは2。BF16出力が今回一致したことを理由に合格へ変更しない。独立したoMLX mHCとの比較や異なるlayer / hiddenの網羅は未実施。full Blockとモデル全体も未接続。

公式Blockへの接続順序は、incoming preでattention入力をcollapseし、新しいattention preでFFN入力をcollapseし、FFN preを次のBlockへ返す。今回のmHC境界はそのための部品であり、attention / MoEの代替出力を返す仮Blockは実装していない。

## 再現

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/export_mhc_fixture.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --summary artifacts/checkpoint/summary.json --verification artifacts/checkpoint/verification.json \
  --entry artifacts/model-entry --output artifacts/mhc
cmake --build build-mlx -j 8
build-mlx/dsv41-mhc-probe \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --summary artifacts/checkpoint/summary.json --fixtures artifacts/mhc \
  --output artifacts/mhc/native-verify.json
```

今回のGPU検査は約0.45秒。長時間jobは起動していない。fixture用Pythonはofflineのみ。次はattention / compressor / indexerとrequest stateを接続し、MoEとともに実Blockへ統合する。mHCのCPUとの差はM2の未解決項目として保持する。
