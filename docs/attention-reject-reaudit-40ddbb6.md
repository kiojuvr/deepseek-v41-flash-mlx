# 40ddbb6 後の attention reject 再監査

日付: 2026-09-19

## 結論

過去の correctness reject を git history、`docs/`、保存済み artifact から再監査した。
`40ddbb6` (`Preserve request boundary attention topology`) によって reject 証拠が失効した
独立の高速 path は見つからなかった。新たに再実行する candidate は **0 件**である。

旧 fixed-tile 系統 (`855756b` 以降) だけは、後続の right alignment、ragged QK/AV、
width-one GEMV、最後に `40ddbb6` の request-boundary compact graph を共有しており、
古い reject を現行実装へ適用できない。この系統はすでに `40ddbb6` の最小 localization、
現行 40-layer production gate、2K production full-path の順で再検証済みである。

優先指定された `62517e2` と `c746c4d` は request-boundary/state/publication の failure ではない。
いずれも QK reduction topology が scalar MLX Steel split-K oracle と異なったための arithmetic
failure であり、`40ddbb6` の修正対象コードを使用していない。再実行しない。

## 40ddbb6 の修正境界

`40ddbb6` は `PackedAttentionWorkList` に `request_boundary` を追加し、固定窓の request 先頭で
local row slot 127 と pooled row slot 128 を別々の 64-row block として reduce する既存 fixed-tile
coreを、同じ2行を連続した compact topology で reduce する request-boundary graphへ置換した。
対象は `swa_packed_attention_work_list(..., fixed_window=true)` から
`swa_attention_fixed_tile_core` へ入る fixed-tile production pathだけである。

したがって、以下はこの修正の影響を受けない。

- `swa_attention_masked_chunk` の padded rank-3 MLX matmul
- DwarfStar-style row-serial custom Metal reduction
- packed direct MMA reduction
- exact packed pathの selected-width rectangular padding
- separately compiled split-K tail
- wide-attention row-serial reduction

`fixed_window` は `855756b` で導入されている。これより前の candidate、および fixed-tile coreを
呼ばない後続 candidateには `40ddbb6` の因果関係がない。

## Candidate 判定

| commit / candidate | 当時の reject 証拠 | 40ddbb6 との依存関係 | 証拠の現在性 | 再試験価値 / 最小再試験 |
|---|---|---|---|---|
| `62517e2` padded rank-3 MLX attention | `attention/chunk-backbone-20260917-085608-37522`。chunk 0 / 40-layer final hidden: RMS `0.00554266`, max `2048`, mean `1.02469`, BF16 mismatch `2,595,308`。位置は chunk-0 final hidden までで、token coordinate は当時未採取。state/route 前に停止。 | なし。rank-3 QK が regular GEMM、oracle の scalar QK が Steel split-Kを選ぶ reduction-selector 差。work list、fixed window、request boundaryを使用しない。 | **有効**。後続 `731c6aa` が exact shape bucketへ置換して bit exact gateを通過。 | なし。旧 rank-3 pathの復元・移植が必要で「現行 architecture の再利用」ではなくなる。再実行しない。 |
| `c746c4d` DwarfStar-style batched attention | `attention/chunk-backbone-20260917-092514-39493`。chunk 0 / 40-layer final hidden: RMS `0.0055142`, max `2048`, mean `0.845199`, mismatch `2,587,396`。token coordinate 未採取、state/route 前に停止。 | なし。1 token / threadgroup、8 headsの row-serial custom Metal QK/online-softmax reduction。fixed-tile work listを通らない。 | **有効**。原因は scalar Steel split-K と異なる SIMD/reduction order。kernelは削除済み。 | なし。再利用には削除済み kernelの再導入と arithmetic の再設計が必要。今回許可された最小 revalidation の範囲外。 |
| `392288c` fused chunk attention | `attention/fused-chunk-backbone-20260918-013048-51465`。chunk 0 hidden RMS `0.000928321`、pre-mix RMS `0.00896325`、pre-mix mismatch `508`。state/discrete checks 前に停止。 | なし。one-dispatch SIMD QK reductionが scalar Steel split-K topologyを保存しなかった。 | **有効**。kernelは `1b802f2` の exact Steel topology batchingに置換。 | なし。後続診断で request boundary が原因ではないと確定している。 |
| `1b802f2` batched split-K QK（tail込み） | `attention/batched-splitk-qk-backbone-20260918-015331-53102`。chunk 0 hidden RMS `0.000640952`、pre-mix RMS `0.00811268`、pre-mix mismatch `508`。 | なし。separately compiled one-column tailが float32 で最大 `0.000046`、71 values 差。 | **旧版には有効**。`c5321ca` が short tailを native Steelへ戻し、`attention/batched-splitk-qk-backbone-20260918-122034-57470` で40-layer bit exact。修正版は現行系統に統合済み。 | なし。修正版の gate 済みで、旧版を再試験する価値がない。 |
| `8dda737` direct packed one-dispatch MMA | `attention/packed-fused-backbone-20260918-131753-60269`。chunk 0 hidden RMS `0.00328311`, max `2048`, mean `0.671945`。official layer-2→3 / 3-token fixtureで consumer RMS `0.005740`。 | なし。state/routingより前の MMA reduction-order incompatibility。request-boundary publication failureではない。 | **有効**。direct fused kernelは diagnostic only のまま。 | なし。既存の短い fixtureが原因を分離済み。 |
| `7e903e2` rectangular packed work-list | `attention/packed-fused-backbone-20260918-132819-61332`。chunk 0 hidden RMS `0.000780165`、pre-mix RMS `0.00817721`、max `0.0319417`、mean `0.00193994`、mismatch `508`。 | なし。chunk-maximum selected-width paddingが reduction geometryを変更。`request_boundary`/`fixed_window` 導入前。 | **旧版には有効**。`3878755` が exact `(raw_width, selected_count)` group shapesを復元し、その successorが bit exact gateを通過。 | なし。修正版は現在の production architectureの祖先で、旧 rectangular pathを戻す理由がない。 |
| `eee85ac` integrated wide attention | `attention/wide-backbone-20260918-225410-67687`。chunk 0 hidden RMS `0.0111344`, max `2048`, mean `1.52399`, mismatch `2,599,165`。 | なし。dense planと transactional publicationは full backboneへ到達済みで、failureは wide row-serial reductionへ局在。wide kernelは fixed-tile coreを呼ばない。 | **有効**。後続診断が host grouping/state/publication を原因から除外。 | なし。performance pathとして再利用するには reduction kernelの新規設計になり、今回の revalidation 範囲外。 |
| `855756b` / `d419749` fixed-tile lineage | 最初の gate `attention/fixed-tile-backbone-20260918-231912-68890`: hidden RMS `0.0111286`, max `2048`, mean `1.45948`, mismatch `2,597,542`。後続 localizationでは layer 3 token 1–2 width 1、さらに layer 20 token 0 width 1へ原因を狭めた。 | **あり**。right alignment、ragged QK/AV、width-one GEMVに加え、layer 20 token 0の最後の差が `40ddbb6` の request-boundary compact topologyそのもの。 | **現行修正版への reject 証拠として失効**。旧 commit自体の failure記録としては有効。 | **再試験済み**。下記3段階が必要最小セットを超えて完了しているため、追加 rerunなし。 |

## Fixed-tile 系統の再検証済み証拠

1. 最小/局所 gate:
   `attention/fixed-tile-layer-localization-20260919-201526-88363` at `40ddbb6`。
   layers 0–39 の production stageがすべて bit exact、
   `first_gate_failure_layer=-1`、final persistent state equality asserted。
   layer 20の rejected native-width-one attribution値は diagnostic として残るが、選択された
   compact request-boundary coreは exact。
2. 既存 40-layer production gate:
   `attention/fixed-tile-backbone-20260919-203130-89172`。
   2x128 tokens の hidden/pre-mix/logits、route ties、state/publication/hash、continuation、
   invalid-token atomicityがすべて exact。fixed-tile calls `76`、batched split-K QK calls `760`、
   scalar QK calls `0`。
3. 2K production full-path observation:
   `context-ladder/32k-run-20260919-203604-89428`。
   2,063-token transactional prefill `43.229745 s` (`47.7218 tok/s`)、next token `339`、swapなし。
   packed exact-shape result `48.749250 s`に対して `11.32%`短縮。この単発値は statistical
   performance qualification、32K、API、256K qualificationではない。

## 再実行しないもの

- 上表の arithmetic reduction failure。`40ddbb6` と独立で、reject 証拠は現在も有効。
- 後続 fixtureで原因ではないと否定された padding-only、global-token-zero、all-block AV 等の仮説。
- trace missing / trace shape mismatch など instrumentation 自体の失敗。
- correctness通過後に現行経路へ性能で明確に負けた path。
- exact successorがすでに同じ40-layer gateを通った旧版。

この監査は保存済み証拠の妥当性確認であり、新しい implementation、checkpoint write、長時間
validationは行っていない。現行 fixed-tile path以外に「現在なら通る可能性があり、かつ既存コードの
まま bounded に再試験できる」candidateはない。

## 後続のdefault昇格

`context-ladder/fixed-tile-paired-20260919-205611-90245`の5組すべてでfixed-tileが勝ち、
mean prefillは49.288916秒から43.653977秒へ11.43%短縮、decode平均は横ばい、swap増分0、
memory増分もboundedだった。このreviewによりfixed-tile + ragged QK/AVを含むqualified resident
transactional構成をproduction defaultへ昇格し、その5-run分布を新しい内部2K baselineとする。
これは32K / 256K / API / 長時間decode / 外部performance qualificationではない。
