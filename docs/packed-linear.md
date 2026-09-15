# 共通packed linear reference

2026-09-13。`PackedLinearReference`を追加した。公式FP8 / FP4 weightをlosslessにMLX uint32へpackし、E8M0 scaleをQMMのrow layoutに対応させる。FP8の32×32 scaleは32行へ展開し、FP4のrow別scaleはそのまま使う。weightの再量子化は行わない。

入力はbatch one、BF16 `[1..128,K]`、Kは32の倍数。公式minimalのactivation量子化（FP32 absmaxを1e-4以上にclamp、上側2冪scale、E4M3 round-to-nearest-even）を使い、量子化bytes・scale・復元BF16を独立した比較境界として返す。FP4 weightにもactivationはFP8を使う。dense BF16 linear、expert routing、Blockは今回の対象外。

## 評価順を固定した理由

初回の一括QMMは `layers.0.attn.wq_a` の10入力で、token単位QMMと12,800要素中1要素がBF16 1 ULP異なった。[初回結果](../artifacts/linear/initial-batched-qmm.json)を保持する。同じ量子化済み入力のoMLX / Python一括QMMとは全bit一致しており、nativeのrepack差ではない。

referenceは入力chunk長によらず1 rowずつQMMを呼ぶscheduleへ固定した。`project_batch_diagnostic`は差の診断用に残し、optimized candidateにはしない。この決定は速度目的ではなく、prefill / decodeのlocal比較基準を一貫させるため。1-row scheduleがすべての入力で公式CUDAと一致する保証はない。

## 確認した結果

[Release結果](../artifacts/linear/native-verify.json)と[ASan / UBSan結果](../artifacts/linear/sanitize-verify.json)、[fixture manifest](../artifacts/linear/manifest.json)を保存した。

| 実checkpoint行列 | 形式 | 比較した出力数 |
| --- | --- | ---: |
| `layers.0.attn.wq_a` | FP8 | 12,800 |
| `layers.0.ffn.experts.0.w1` | FP4 | 23,040 |
| `layers.0.ffn.experts.0.w2` | FP4 | 51,200 |

各10入力に実embedding、zero、2^-60、2^60を含む。固定scheduleではCPU転記referenceとの出力比較とtokenwise / chunk比較が全bit一致した。packed weight bytes、展開scale、activation bytes / scale / BF16復元も一致。小さいwidthのpartial threadgroup、invalid shape / dtype、owner破棄後のgraph評価も確認した。

CPU側は公式kernelの数式をPyTorchへ転記したもので、公式TileLang / CUDA実行ではない。status `reference_fixtures_exact` はこのfixtureの一致のみを意味する。モデル全体やoptimized pathのqualificationではない。

## oMLXとの差を保持する

[固定oMLX比較](../artifacts/linear/omlx-comparison.json)では、公式下限1e-4に対し、oMLX v0.7.0.dev2のGPU activation kernelが448×2^-126を下限にする差を確認した。2^-60入力では公式経路が0へ丸める一方、oMLXは非zeroを保持し、projection出力にも差が出る。今回の通常embedding入力ではactivation出力は一致した。実sessionの精度への影響は未測定。

共通linearは公式下限を維持する。oMLX対照は、同じ公式量子化済み入力を与えたQMMと、元のoMLX activation経路との差の診断を分けている。oMLXに合わせるためだけに意味論を変更しない。

## 再現

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/export_linear_fixture.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --m1-summary artifacts/checkpoint/summary.json --verification artifacts/checkpoint/verification.json \
  --entry-fixtures artifacts/model-entry --output artifacts/linear
/Volumes/SDXC-512/glm53-flash-mlx/.venv/bin/python tools/reference/compare_omlx_linear.py \
  --omlx /Users/kioju/omlx-0.7.0.dev2 --fixtures artifacts/linear
cmake --build build-mlx -j 8
build-mlx/dsv41-linear-probe \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --m1-summary artifacts/checkpoint/summary.json --fixtures artifacts/linear \
  --output artifacts/linear/native-verify.json
```

Pythonはoffline fixture / 診断のみ。native実行はC++ / MLX / Metal。Releaseの今回の検査は約0.52秒で、長時間jobは起動していない。raw fixtureはgit対象外で再生成可能。元checkpointはread-only。

次はこの共通linearと既存RMSNormを使いmHC / Blockへ接続する。一括QMMへの置換やthroughput最適化を先行しない。既存Engramの独立serial projectionは比較referenceとして残す。
