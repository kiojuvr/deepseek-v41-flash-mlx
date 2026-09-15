# Decoder 20..39とfull backboneの検証

2026-09-14。`TextDecoderReference`（layer 20..39）と`TextBackboneReference`（token → encoder 0..19 → decoder 20..39 → final collapse / norm / head）を追加した。ユーザーrun `run-20260914-063734-8671`（decoder単独、終了code 0）と`run-20260914-063742-8680`（full backbone、終了code 0）のログ、binary / M1 summary /（backboneはEngram metadata / fixture provenance）のSHA256を確認した。すべてレビュー時の現物と一致し、metadataはfixture-provenanceと一致した。以下のlocal整合性検査は通過した。decoderのrouting tieは0、backboneは3で、いずれも最小expert IDでbreakした。[decoder記録](../artifacts/text-decoder/reviewed-result.json)と[backbone記録](../artifacts/text-backbone/reviewed-result.json)にログとidentityを保存した。

変更なしでの再実行は不要。以下は再現用手順。公式oracle比較、memory peak・経過時間・throughputは未測定であり、このPASSから推定しない。

## Routing tieの扱い

gateの`(scores+bias)`でtop-6境界が厳密に等しい場合、公式`topk`のtie-breakはoracle未確認である。従来は停止していたが、full pathを進めるため、reference MoEは**最小expert IDでtie-break**する非strict経路へ変更した。tie発生回数をprocess全体で数え、テストが報告する。今回の3件は未qualifiedのoracle差として記録し、公式logitsとの一致判定には使わない。index top-kの同点境界は従来どおり停止する。

## 実装境界

- `IndexQueryReference`がcandidate二段Top-Kを実装する。`select_candidate_blocks_reference`がlevel one（`candidate_topk_blocks=2048`、`candidate_block_size=8`、最新blockをpin）、そのmask内でlevel twoのtop-k（`index_topk=512`）を行う。
- layer 20が`candidate_source_layer`かつratio-1 kv_source / index source。`CompressedLayerReference`は`layer_==20`でcandidate source、`layer_>20`でcandidate consumerになる。
- 非sourceのindex source（24/28/32/36）は`ReusedLayerReference`内の`IndexQueryReference`で、layer 20のshared index K（`publication.cache()`）から自分のtop-kを再計算し、`SharedAttentionReference::republish`でpublicationを更新する。candidate maskはcandidate sourceが所有し、後続index sourceは上書きしない。
- layer 21–23、25–27、29–31、33–35、37–39は直前index sourceのtop-kを再利用する。
- final collapseは`hc_pre(h, pre_mix)`、続いて`RMSNorm`、`head.weight`[129280,5120] BF16をFP32へwideningしたmatmul。samplingは未実装でlogitsまで。

## 検証範囲

`dsv41-text-decoder-test`はsyntheticなhidden / pre-mixでdecoder 20..39を単独検査する（encoderから切り離す）。`dsv41-text-backbone-test`は実token IDs `[0,42,1000]`と継続token `42`でencoder→decoder→logitsを検査する。どちらもchunk/tokenwise hidden・pre-mix・stateのbit一致、logitsのchunk/tokenwise bit一致、継続、fork/reset、不正入力拒否を確認する。

2026-09-15に`dsv41-text-backbone-test`を任意token fileへ拡張した。128 token以下のchunk列と
1-token逐次列について、各tokenのhidden / pre-mix、最終logits、SWA / compressed global KV・
scale・pending compressionを含むstate、continuation、fork、resetをbit比較する。既定3 token
検査は維持する。129-token境界の実行は次。

```sh
PROMPT_LENGTH=129 REPEAT_TOKEN=42 bash tools/benchmark/run_text_backbone_reference.sh
```

full backboneを約3 prompt pass＋continuation 2回相当実行するため数分以上、数百GB規模の
Unified MemoryとEngram SSD I/Oを使い得る。他のmodel runと同時に開始しない。
checkpointはread-only。build、command、
diff、source/binary/prompt identity、test log、exit codeは`artifacts/text-backbone/run-日時-PID/`へ
保存する。失敗時もdirectoryを保持し同じcommandでfresh run。途中resumeはない。結果レビュー
までは通過とせず、外部oracle、performance / peak memory、256K qualificationを含めない。

反復tokenでの129境界は`run-20260915-154605-51571`をレビューし、128+1と1×129で各tokenの
hidden / pre-mix、最終logits、公開state、continuation、fork、resetがbitwise一致した。
identity 9件も現物一致、routing tieは0。[レビュー記録](../artifacts/text-backbone/reviewed-prefill-129-exact-20260915.json)。
反復入力に偏らない次の比較では、既存60-token coding promptをcycleして129 tokenへ固定する。

```sh
PROMPT_LENGTH=129 \
  PROMPT_PATTERN_FILE=artifacts/logits-trace/prompt-tokens.txt \
  bash tools/benchmark/run_text_backbone_reference.sh
```

`expand_token_pattern.py`はvalidation専用のoffline Pythonでありproduction dependencyではない。
生成された129-token promptと元pattern、生成scriptのSHA256もrun identityへ保存する。実行資源、
失敗時のfresh run、resumeなし、未qualified範囲は上記の反復token検査と同じ。

初回多様prompt run `run-20260915-155222-51768`はexit 0、identity 11件が現物一致し、45種の
tokenを含む正確な129-token cycleで全tensor/state比較が通過した。
[レビュー記録](../artifacts/text-backbone/reviewed-prefill-129-diverse-exact-20260915.json)。ただし
表示されたtie count 3はchunk / serial / resetのprocess-wide合計で、個々のtie recordを経路間で
比較していなかった。このrunはtensor/state exactnessの証拠として保持するが、routing discrete
decisionの完了証拠にはしない。

testを強化し、chunk / serial / resetごとにtie記録をresetして、layer、token position、6位/7位
expert ID、両FP32 score bitsを完全比較する。canonical chunk側の各recordもtest logへ出力する。
上記と同じ多様129-token commandを再実行し、結果レビュー後にrouting schedule exactnessを判定する。

強化後の`run-20260915-161212-52162`をレビューし、exit 0、identity 11件の現物一致を確認した。
chunk / serial / resetのtie recordは完全一致し、唯一のtieはlayer 8 / token 14、6位ID 104、
7位ID 117、両score bits 1095107359（FP32 12.375761985778809）だった。
[レビュー記録](../artifacts/text-backbone/reviewed-prefill-129-route-ties-20260915.json)。これでlocal
schedule間のrouting discrete decision exactnessは通過。lowest-ID選択自体は公式`torch.topk`
式がtie順を規定しないため、外部診断とpolicy判断を分離する。

次のrunnerは同じ129-token promptをnative MLXとpinned oMLXへ逐次投入し、全境界traceに加え
tie boundary / ID / FP32 score bits / oMLX selected IDsを比較する。

```sh
bash tools/benchmark/run_m2_route_tie_comparison.sh
```

nativeとoMLXを順番に各1回loadしてfull traceを作る。数分以上、各modelのUnified Memory圧力は
数百GB規模になり得るため他のmodel runと同時に開始しない。60-token実績から129-token traceの
disk使用量は合計約1.5 GiBを見込む。checkpointはread-only。結果は
`artifacts/m2-route-tie/run-日時-PID/`へ保存する。失敗時はdirectoryを保持してfresh runでき、
片方の`manifest.json`まで完成していれば記録されたcompare command相当を手動再実行できる。
model途中stateのresumeはない。runner exit 0は観測完了であり、数値差やlowest-ID policyのPASS、
M2完了、性能、256K qualificationを意味しない。

`run-20260915-162023-52563`をレビューした。exit 0、identity 13件はレビュー時の現物と一致。
nativeはlayer 8 / token 14で上記1 tie、oMLXはtie 0だった。ただし両traceの最初のbit差は
`encoder.layer0.attn_q`にあり、layer 8 `ffn_in`は647255 / 660480要素が不一致
（max abs 0.346923828125）。tie token 14単独でも5045 / 5120要素が不一致
（max abs 0.106201171875）である。このためoMLX側のnon-tieはupstream activation差に
confoundされ、tie-break policyの反証にも確認にもならない。
[レビュー記録](../artifacts/m2-route-tie/reviewed-full-trace-20260915.json)。

次はnative traceのlayer 8 / token 14 `ffn_in`を固定し、pinned oMLXのGate式へ同じcheckpointの
gate weight / biasだけを与える。full modelはloadせず、該当tensorだけをcheckpointからread-onlyで
読む短い診断である。

```sh
bash tools/benchmark/run_m2_frozen_gate_tie_check.sh
```

通常は数秒〜数十秒、Gateの約256×5120 matmulと該当checkpoint shardの部分readのみで、出力は
`artifacts/m2-route-tie/frozen-gate-日時-PID/`の小さなJSON/logである。失敗時はdirectoryを保持し、
同じcommandでfresh runする（途中resume不要）。結果レビューまでは通過としない。この診断も
oMLX `mx.argsort`の観測であり、公式`torch.topk`、M2、性能、256Kをqualifyしない。

oMLX固定入力診断後は、同じ入力とweight / biasでcheckpointの`inference/model.py`から
`Gate` classを変更なしにAST抽出し、torch CPUの実`topk`選択を記録する。

```sh
bash tools/benchmark/run_official_frozen_gate_tie_check.sh
```

これも該当Gate tensorだけをread-onlyで読み、通常は数秒〜数十秒、出力は
`artifacts/m2-route-tie/official-frozen-gate-日時-PID/`の小さなJSON/log。失敗時はfresh run、
途中resumeは不要。CPUでの1観測であり、torch APIがtie順を保証しない以上、CUDA kernelや
将来versionへ一般化せず、結果レビュー前は通過としない。

固定入力の両runをレビューした。oMLX run `frozen-gate-20260915-163847-52834`と公式
torch CPU run `official-frozen-gate-20260915-164059-52892`はいずれもexit 0で、入力、source、
checkpoint index / config / shardのidentityは現物一致。両方とも境界ID `[104,117]`、score bits
`1095107359`を再現した。pinned oMLX `mx.argsort`は`104`、公式`Gate.forward`を実行した
torch 2.13 CPU `topk`は`117`を6番目に選択した。
[レビュー記録](../artifacts/m2-route-tie/reviewed-frozen-gate-20260915.json)。

この結果により、前runのoMLX non-tieがupstream差によることと、現canonical MLXのlowest-IDが
再現可能な明示policyである一方、公式torch CPUのこの実行とはdiscrete decisionが異なることを
確定した。`torch.topk`のtie順はAPI保証外なので、runtimeを観測依存のhighest-IDへ変更しない。
canonical local policyの回帰testを追加し、strict経路は引き続きtieを拒否する。公式CUDA / future
torchへのportable一致は未qualifiedのまま、M2 backend exactnessとは別証拠として保持する。
`dsv41-route-policy-test`を独立targetとしてbuild・実行し、strict拒否、lowest-ID `[0..5]`、
tie record `(5,6)`、counter 1を確認した。

## 資源と実行

decoder単独は20 Block分、full backboneは40 Block分のrouted expertを扱う。expertは初回使用時にloadしてcacheするon-demand経路（全384 expert resident loadと数値同一）にしたため、実行したtokenで選択されたexpertだけが常駐し、peakはBlock数・選択expert数に比例する。それでもattention・MLX staging・allocatorを含む実peakは未測定。Engramはmmapで必要rowのみ読み、checkpointはread-only。初回I/Oで数分以上になる可能性があるため、エージェント側では起動していない。

```sh
cd /Volumes/SDXC-512/deepseek-v41-flash-mlx
bash tools/benchmark/run_text_decoder_reference.sh
bash tools/benchmark/run_text_backbone_reference.sh
```

`artifacts/text-decoder/run-*`と`artifacts/text-backbone/run-*`へ`test.log`、`exit-code.txt`、binary / M1 summary /（backboneはEngram metadata / provenance）のSHA256を含む`identity.txt`を保存する。失敗・中断時は同じコマンドで新しいrunとして最初から再試行する。終了codeがないrunは未完了。ログとidentityのレビュー前に通過とは判定しない。

## 未接続・引き継ぐ差

- 公式oracle比較は未完了。logitsを公式minimal inferenceと照合していない。sampling / generation / DSpark / vision / APIは未実装。
- main compressed KVの値は既にE2M1 packed FP4、scaleがE4M3 block-16。以前の「E4M3のまま」という記述を訂正する。形式の置換ではなく、丸め・packed bytes・復元の独立oracle比較が未完了。index / attentionの独立比較も未完了。
- candidate二段Top-Kのblock選択・tie-breakは公式CUDA kernelと照合していない。top-6 routing tieは最小IDでbreakし、回数のみ記録する未qualified差。既知mHC / RMSNorm / Engram差を引き継ぐ。encoderの過去PASSをdecoder / backboneの証拠へ転用しない。
