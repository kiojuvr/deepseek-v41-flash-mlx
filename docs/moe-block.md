# Layer 0 MoE / Block reference

2026-09-13。text layer 0のGate、Expert、resident MoE、mHCとSWAを含むBlockを実装した。ユーザー実行のlayer 0検査ログと終了code、binary / M1 summaryのSHA256を確認し、local Block整合性検査の通過を記録した。公式oracleとのcorrectness判定は未完了。

## 演算契約

GateはBF16 checkpoint weightをFP32へ変換し、FP32 logits → sqrt(softplus)を計算する。gate temperature=1、softplus threshold=20。biasはtop-6の選択だけに使い、選択されたbiasなしscoreをsum+1e-20で正規化して1.5倍する。text専用で、vision biasはまだ使用しない。

選択はCPUへ384 scoreを渡す同期reference。第6位と第7位が同値なら、公式tie oracle未確定としてエラーにする。選択集合内の同点はID順とするが、その順序による正規化reductionの公式一致も未判定。これはproduction routerではない。

Expertはw1 / w3をBF16で受け、FP32でgateの上限10、upの上下限±10を適用する。SwiGLUの結果にrouting weightを掛けてからBF16へcastし、w2へ渡す。routedは公式packed FP4、sharedはFP8。MoEはID昇順でrouted出力をFP32加算し、sharedを加えてBF16へ戻す。routed expertは初回使用時にloadしてcacheする（on-demand）。数値は全384 expert resident loadと同一で、full backboneのmemoryを抑える。expertのSSD streamingや明示的なevictionは導入しない。

Blockはincoming pre-mixでattention入力をcollapseし、attention側で計算したpreをFFN入力に使い、FFN側preを次Blockへ返す。layer 0にはEngramがない。norm → attention → mHC展開 → norm → MoE → mHC展開を接続し、chunk全体の出力評価成功後にstateをcommitする。全chunkをtoken逐次で処理するlocal reference。

## 確認済み範囲

短いGPU検査で、biasが選択だけに作用する解析例、top-6境界tie拒否、clampとroute weightの位置を確認した。実checkpointのGate、shared expert、選択された1つのFP4 expertは有限値、繰り返しbit一致、zero route weightのゼロ出力を確認した。既存SWA検査も通過。これらは公式CUDAとの独立比較ではない。

全384 expertをloadするBlock検査は、ユーザーrun `run-20260913-143331-24626`で終了code 0。合成2 tokenのchunk/tokenwise hidden・pre-mix・KV stateのbit一致、reset再現性、position拒否が通過した。保存されたbinary / M1 summary hashはレビュー時の現物と一致した。[レビュー記録](../artifacts/block/reviewed-result.json)にログ本文とidentityを保存した。384 expertすべてを実行した検査ではなく、routingで選択されたexpertを実行する検査である。このrunはeager load時点のbinaryであり、現在のon-demand経路でも数値は同一。

mHC / RMSNormの既知CPU差、attentionのoracle未比較は引き継ぐ。load時間、memory peak、throughputは今回のログから判定できない。全layer / logits / 256K qualificationは未完了。

## ユーザー実行

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_block_reference.sh
```

layer 0の全expertを一度loadし、合成2 tokenのchunk/tokenwise hidden・pre-mix・KV state bit一致、reset、position拒否を検査する。checkpointはread-only。FFN tensor payloadは7,258,804,992 bytes（6.7603 GiB）。attention・MLX staging・allocatorを含む実peakは未測定なので、少なくとも約12 GiBの空きを目安とする。モデル全体やembeddingはloadしない。初回SSD I/O状況により数分以上になる可能性がある。今回のユーザーrunは確認済みであり、変更なしで再実行する必要はない。

ログは`artifacts/block/run-日時-PID/test.log`、終了codeは`exit-code.txt`、binaryとM1 summaryのSHA256は`identity.txt`。失敗時は非ゼロ終了。中断・失敗したrunを上書きせず、同じコマンドで最初から再試行する。単一Blockの短い計算なので途中stateの永続化は行わない。完了後にログを確認して判定する。PASSでも公式oracleやfull-model qualificationの通過には数えない。
