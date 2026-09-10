# Correctness contract

状態: M0。速度よりcorrectnessを優先し、未検証をpassに数えない。

## Canonical source

公式checkpointのtensorとconfigをcanonical sourceとする。[architecture](architecture.md)の公式minimal inferenceを意味論のoracleにする。vLLMはproduction構造の参考で、公式tensorや意味論を上書きしない。

checkpoint storage dtype、packed logical dtype、scale encoding、group size、rounding、clamping、accumulation dtype、cast境界を別々に記録する。I8等の保存dtypeを見ただけで数値形式を決めない。公式FP4 weights / KVのdecodeは許可するが、unofficial quantization、再量子化、非可逆変換をbaselineへ導入しない。変換物は元tensor、revision、digest、変換規則へ逆引きできるようにする。

## Referenceを二段階で検証する

1. **公式oracle → local reference。** C++ / MLX / Metal上に読みやすいlocal referenceを保持する。CUDA等との算術差は演算・tensorごとに記録し、absolute / relative / ULP errorとdiscrete decisionを比較する。許容値が必要なら公式oracleの観測に基づきM2で事前固定する。現時点で数値許容値は未設定でありoracle一致は未達。小型fixtureだけでreleased checkpointの一致を主張しない。
2. **local reference → optimized path。** 同じ入力、weights、state、backend条件に対して、境界tensor、logits、persistent stateの有効領域をbitwise一致させる。token列の一致、近いlogits、task scoreだけではexactnessを満たさない。演算順やfusionによりbitが変わったcandidateは不合格とする。

異なるstorage layoutはlogical順に正規化して比較する。packed KVはpayload / scaleと、公式のcastを再現してdecodeした値の両方を検証する。padding等の未定義領域は明示的に除外し、有効領域の不一致を隠さない。referenceの算術をcandidateに合わせて変更する場合は別の意味論変更としてoracleから再検証する。

cross-backendでbitwise一致を保証できないことをoptimized pathの許容差へ転用しない。Top-Kやroutingのtie policyが未定義ならfixtureとreference規則を先に確定し、結果の違いを浮動小数点誤差として片付けない。

## 必須の比較境界

| 領域 | 比較する内容 |
| --- | --- |
| protocol | prompt bytes、token IDs、特殊token、tool / reasoning / image区切り |
| quantization | FP4 nibble順、E4M3 / E8M0 scale、zero group、rounding / saturation、cast |
| encoder / decoder | layer入出力、CED境界のmHC state・collapse・norm・global KV projection |
| CSA2 / indexer | source所有権、indexer K生成前のlatent、RoPE、候補block、Top-K、mask、共有・再利用 |
| SWA / compression | ring wrap、可視token範囲、未完了compression group、replay前後の継続state |
| MoE | scores、selection bias、expert IDs、routing weight、shared / routed expert合成 |
| mHC | mix算出、Sinkhorn、cast、次sublayerへ渡すpre-mix |
| Engram | token mapping、n-gram history、hash / row IDs、row bytes / scale、miss / hit結果 |
| output | norm、全logits、greedy token、固定RNG条件のsampling |
| DSpark | draft / confidence / target state、verify、accept / reject、commit / rollback |
| vision | preprocessing、patch順、2D-RoPE、downsample、projector、image span、Engram mask |

DSparkのsampling正当性は公式algorithmと照合する。乱数消費が異なる方式のtoken列を同じseedだけで比較せず、固定draft / RNG入力でverificationとstate commitを検証し、target分布の保持を別に確認する。

## Stateと実行schedule

plain reference forward、chunked prefill、CED prefill + Bounded Replay、tokenごとのdecodeが同じ継続状態を作ることを検証する。公式minimalの既存chunk制約を勝手に一般化しない。arbitrary chunk対応にはcompression端数、SWA、Engram、mHCを含めた独立の証拠が必要。

Bounded Replayにはglobal KVとtoken IDsだけで十分と仮定しない。再開位置で必要なhidden / mHC / compression / n-gram stateを特定し、replay対象・必要な前史・最大保持量を証明する。全履歴計算とのexact比較に合格するまでproduction pathへ昇格しない。

最低限、empty / 1 token、window境界127 / 128 / 129、compression ratio境界、candidate block境界、source切替、context上限、複数turn追記、reset / cancel / resumeを対象にする。後段ではDSpark accept / reject、imageとtextの境界、prefix再利用も加える。Engram cache evictionやI/O順序はlogitsとstateを変えてはならない。

NaN / Inf、範囲外index、未初期化read、stale graph、解放済みbuffer参照、session間state混入を失敗として扱う。長時間比較は同一入力列をteacher-forceして差分を追跡する試験と、greedyで自走する試験の両方を持つ。

## 証拠とpromotion

fixtureにはcheckpoint / tokenizer / oracle / local commitのidentity、入力token IDs、位置、seed、dtype、execution mode、期待出力を記録する。失敗時は最初の不一致layer / tensor / tokenを保存する。state digestだけに依存せず、差分を再現して値を比較できるようにする。

公式`model.py`のself-testは未初期化weightsによるshape / plumbing確認であり数値検証ではない。oracleを実行できなければその状態を明記し、推測fixtureをoracle結果と呼ばない。

昇格にはlocal exactness、full-path統合、[qualification](qualification.md)のwall / TPT改善、memoryとlifecycleの検証がすべて必要。実験kernelやpartial graphは実験として保存できるが、default切替やproduction candidateの呼称には使わない。reference pathはrelease後も残す。
