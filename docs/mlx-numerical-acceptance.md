# MLX referenceの数値的な着地点

2026-09-15: M2のMLX reference着地点を固定する。これは公式CUDA一致の宣言でも、既存の長文qualificationを完了するものでもない。

公式checkpoint・演算の意味論・cast境界・KV形式を維持したMLX referenceを
目標とする。異なるGPU backendのreduction順を全面的に再実装することを
開発の前提にはしない。一方、差が小さいという理由だけで意味論の差を
丸め差へ分類しない。公式CPU転記、固定oMLX、公式CUDA実行は別々の証拠である。

## 差の分類と処置

| 分類 | 例 | 処置 |
| --- | --- | --- |
| 意味論・データの差 | tensor、scale形式、mask、位置、cast位置、state更新 | referenceを修正し再検証。許容差で吸収しない |
| 同じ入力での算術差 | reduction順、transcendental、fusion | 最初の演算を局所比較し、absolute/relative/ULPと入力規模依存性を記録 |
| 離散判断の差 | routing、index top-k、greedy、停止条件 | score margin、同点規則、選択IDを確認。小さいtensor誤差だけでは受理しない |
| 上流差の伝播 | 後続layerやlogitsの差 | 固定境界入力の局所比較と、全経路teacher forcingを分けて検証 |

RMSNormもreductionを含むため、演算名や要素数だけでcross-backend bit一致の
可否を判定しない。既存correctness契約で未達となっている差は、根拠を伴う
基準の更新まで未解決として残す。local reference → optimized pathは引き続き
同じbackend条件でbit一致を要求する。公式KVのpayload/scaleや永続stateの
差を一般的なlogits許容差で免除しない。

## 基準を固定するまで

1. 同じ入力・weights・dtype・scheduleで最初の分岐を再現する。
   演算前の値を保存し、projectionとnormalizationをまとめて原因と呼ばない。
2. 小さいfixture、実coding入力、極小値・大きい値・丸め境界を使う。
   観測最大誤差に合わせるだけの閾値を設けず、演算特性と独立データで検証する。
3. 診断用データと判定用データを分け、許容値・適用dtype/演算/入力範囲を
   判定用runの前に固定する。範囲外は未検証扱いとする。
4. layer別・position別の誤差、logits差とtop候補margin、routing/index判断、
   persistent stateを確認する。4 tokenのargmax一致はM2判定の代用にしない。
5. 32Kから256Kのteacher-forced継続と生成を検証する。short traceで受理できた
   数値差を長文での安定性や性能の証拠として扱わない。

公式CUDAを実行できない間は「公式式CPU転記を比較したMLX reference」という
到達点を明記できる。ただし「公式CUDA一致」「M2 qualified」「RELEASE」とは
区別する。検証可能な到達点を積み上げながら実装を進める。

## 確認できた丸め境界の例

60 tokenのlayer 0 Q正規化では、同じnative Q入力に対する分散reduction差が
BF16出力の1要素差へ伝播した。nativeの分散をCPUへ差し込むと、その後の
逆平方根・乗算・BF16 castは一致する。[診断結果](cpu-attention-reference.md)に
各段階と丸め前の値を保存した。

このケースはvariance reductionのbackend差として分類できるが、RMSNorm全体を
無条件に「数ULPなら合格」へ変更する根拠にはしない。BF16中点付近のfixtureと
独立入力へ対象を広げ、局所誤差の境界と離散判断への影響を別に評価する。
現時点でrouting/index/greedyへの影響は未測定であり、M2判定は保留する。
