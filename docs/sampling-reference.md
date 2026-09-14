# Sampling / generation reference

2026-09-14。`sample_reference`、`greedy_reference`、`TextGenerationReference`を追加した。**公式RNGとのsampling exactnessは未検証であり、M2合格ではない。**

## 実装境界

- `greedy_reference`はFP32 logitsのargmax。非有限値を拒否する。
- `sample_reference`は`temperature <= 0`でgreedy、それ以外は公式`sample`と同じ形のGumbel-max: `softmax(logits/temperature)`をFP32で計算し、`argmax(probs / Exp(1))`を取る。
- RNGはsplitmix64による文書化した参照列で、`(0,1)`のuniformから`Exp(1) = -log(u)`を作る。**公式torch RNGとは異なる**ため、固定seedでの再現性はnative内で保証するが、公式kernelのtoken列とは一致しない。sampling exactnessは別gate。
- `TextGenerationReference::generate`はpromptをprefillし、1 tokenずつdecodeしてcommitted tokenを返す。`stop_ids`で停止する。stateは`TextBackboneState`が継続する。

## 高速検査（checkpoint不要）

`dsv41-sampling-test`がgreedy argmax、temperature 0のgreedy化、dominant logitのほぼ確実な選択、同一seedの再現、uniform分布での全カテゴリ到達、非有限値拒否を確認する。ユーザーrun不要。

## モデル生成（ユーザー実行）

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
MAX_NEW=16 TEMPERATURE=0 SEED=0 bash tools/benchmark/run_text_generate.sh
```

full backboneのため数分以上かかる。`artifacts/text-generate/run-日時-PID/`へ`test.log`、`exit-code.txt`、identityを保存する。

2026-09-14のユーザーrun: greedy `run-20260914-124922-18696`（exit 0）が `20 49 438 223 20 28 19 271 80 26402 20 49 11 438 223 18`、temperature 0.7 `run-20260914-125105-18707`（exit 0）が `20 49 438 223 20 982 223 19 1492 223 864 438 223 18 16 779` を16 token生成した。binary / M1 summary / Engram metadata / provenanceのSHA256は現物と一致。記録は[reviewed-result](../artifacts/text-generate/reviewed-result.json)。これはnative内の生成loop・state継続・RNG再現性の確認であり、公式RNGとの一致やM2合格ではない。

修正版MoE/full-backbone後の再実行でも、greedy `run-20260914-214412-28826` は同一の16 token列を生成し、temperature 0.7 `run-20260914-214445-28897` は `20 49 438 223 18 14 25 7 343 3556 7308 14361 114636 37009 46254 1075` を生成した。両runともexit 0、`next_position=20`、binaryとcheckpoint/Engram identityは一致した。公式sampling oracleとの一致は未検証である。
レビュー記録は[`reviewed-result-20260914-corrected.json`](../artifacts/text-generate/reviewed-result-20260914-corrected.json)に保存した。

## 未完了

- 公式`sample` / torch RNGとのtoken列一致は未検証。greedyのlogits選択はnative内で決定的だが、公式との一致判定は行っていない。
- `logits`差のM2基準（[correctness契約](correctness.md)）は未固定。samplingのtoken列一致だけで合格にしない。
- DSpark、stop sequenceのparser連携、API経路、streamingは未接続。
