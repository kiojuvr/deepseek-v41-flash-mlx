# SWA optimized reduction: non-bitwise promotion gate

状態: reuse chunk-attention候補を評価中。reference bit-exact contractを変更しないまま、hardware-native SWA候補を評価するための
事前固定gateである。観測結果を見て閾値を緩めない。

## 進行条件

専用Metal kernelへ進む前に、同一2063-token入力・compact bank・cache・I/O条件で
`run_layer_component_profile.sh`を実行する。GPU completionをattention / MoE / post-MoE境界で同期し、
40層とcomponent別wallを保存する。同期により通常の226秒scheduleを摂動するため、絶対performance値では
なく支配項の順位にのみ使う。

SWA最適化を次の主要作業にする条件は、attention pathがcomponent合計の最大項であり、かつ合計wallの
40%以上を占めることとする。満たさない場合は最大項を先に最適化し、SWA kernel実装を保留する。

## Non-bitwise semantic gate

referenceとcandidateは同じcheckpoint、token列、cast位置、mask、position、sink、persistent-state
commit境界を使う。candidateだけがQK / AV reduction geometryを変更できる。次を独立して記録する。

- attention出力、各layer hidden / pre-mix、最終logitsのmax absolute、mean absolute、BF16 bit mismatch数
- 全layer・全tokenのrouting expert ID、route weight、境界margin、tie record
- layer 2/8/14/20/24/28/32/36のindex selected ID、candidate mask、境界margin、tie record
- SWA window、compressor pending、global KV packed bytes/scales、hash position、publication、最終state position
- 全token logits argmaxと、top-2 logit margin

短い診断入力と判定入力を分離する。診断は128 token以下、判定は少なくともcoding-agent teacher-forced
2,048 tokenと丸め境界fixtureを含める。判定入力を見る前に以下を固定する。

1. routing ID、index ID、candidate mask、tie処理、greedy argmax、停止判断は完全一致。
2. persistent stateのposition、packed KV bytes/scales、pending compressor、hash、publicationは完全一致。
3. 非有限値は0。40層candidateのhidden / pre-mix / logitsはrelative RMS `<0.002`、全tokenの
   logits argmaxは一致し、top-2 marginを反転しない。この閾値はDwarfStarのV4.1 batched-attention
   fixtureと同じで、40層結果を見る前に固定する。
4. candidate差の最初の発生点がattention reductionであり、projection、mask、cast、state更新差でない。

短いprobeで離散判断またはpersistent stateが一つでも分岐した場合、non-bitwise promotionはrejectする。
その場合に限り、SWAが上記40%条件も満たしていればreference reduction順を固定する専用Metal kernelへ進む。
短いprobeがpassしても、それだけでreference bit-exactness、M2、performance、32Kをqualifyしない。

## Reproducible wall breakdown

```sh
bash tools/benchmark/run_layer_component_profile.sh
```

1 model、2063-token prefill + 1 decode、約5--10分、Unified Memory目標30 GB未満・予算100 GB、
checkpoint read-only、SSD readあり。結果は
`artifacts/context-ladder/32k-run-日時-PID/{result.json,result.json.progress.jsonl,resource.log,identity.txt}`。
途中resumeはなく、失敗directoryを保持して同じcommandでfresh rerunする。

計測器の短い128-token確認 `context-ladder/32k-run-20260916-232539-28919`はexit 0。40層すべて
layer/component callが1回で、layer合計13.545秒、attention path 3.160秒、MoE path 7.351秒、
post-MoE 2.389秒だった。短contextではattentionは支配項でないが、global row依存費用を含む
2063-token結果までは進行条件を判定しない。

full profile `context-ladder/32k-run-20260916-233434-29168`をreview済み。exit 0、identityと固定条件は
一致し、40層すべてlayer/component callが17回だった。prefillは225.927秒、9.1313 token/sで、通常の
同条件run 226.037秒との差は-0.05%。layer合計225.605秒のうち、attention path 71.066秒
（layer合計31.5%、component合計32.9%）、MoE path 106.119秒（47.0%、49.1%）、post-MoE
38.743秒（17.2%、17.9%）、未分類layer overhead 9.678秒（4.3%）だった。attention内訳はpure SWA
layers 0/1が2.627秒、producer 2/8/14/20が9.925秒、reuseが58.514秒。attentionは最大項でなく40%にも
達しないため、事前固定した進行条件をfailとし、SWA専用Metal kernelは保留する。次のhardware-native
最適化対象は最大項のMoE pathである。このprofile単体はperformance / 32K qualificationではない。

resident component profile `context-ladder/32k-run-20260917-005021-32331`では構造変更後のattentionが
69.652秒、layer wallの55.22%で最大項となったため、進行条件を満たした。最初の候補は専用Metal kernel
ではなく、DwarfStarと同様にreuse attention rowsを一つのbatch graphへまとめる。短いfixtureだけでは
promoteせず、`run_chunk_attention_backbone_check.sh`の2×128 / 40層gateをreviewしてからfull-pathへ進む。
