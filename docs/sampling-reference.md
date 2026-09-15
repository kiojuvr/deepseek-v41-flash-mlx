# Sampling / generation reference

2026-09-14。`sample_reference`、`greedy_reference`、`TextGenerationReference`を追加した。**公式RNGとのsampling exactnessは未検証であり、M2合格ではない。**

## 実装境界

- `greedy_reference`はFP32 logitsのargmax。非有限値を拒否する。
- `sample_reference`は`temperature <= 0`でgreedy、それ以外は公式`sample`と同じ形のGumbel-max: `softmax(logits/max(temperature,1e-5))`をFP32で計算し、`argmax(probs / Exp(1))`を取る。
- RNGはsplitmix64による文書化した参照列で、`(0,1)`のuniformから`Exp(1) = -log(u)`を作る。**公式torch RNGとは異なる**ため、固定seedでの再現性はnative内で保証するが、公式kernelのtoken列とは一致しない。sampling exactnessは別gate。
- `sample_from_exponentials_reference`はcaller固定のFP32 Exp(1)列を受け取り、RNGとsampling式を分離する。logits除算、softmax、確率/Exp除算をFP32で行い、shape・正値・finiteを検査する。local CPU reduction scheduleでありCUDA bit一致は主張しない。
- `TextGenerationReference::generate`はpromptをprefillし、1 tokenずつdecodeしてcommitted tokenを返す。`stop_ids`で停止する。stateは`TextBackboneState`が継続する。

## 高速検査（checkpoint不要）

CTest `sampling_reference`（実体は`dsv41-sampling-test`）がgreedy argmax、temperature 0の
greedy化、固定Exp(1)でのFP32式とtemperature floor、同一seedの再現、uniform分布での
全カテゴリ到達、非有限logits/temperature拒否を確認する。ユーザーrun不要。

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
- DSparkは未接続。API / streaming / stop sequence parserは後続で接続済みだが、
  sampling exactnessや広いstop条件のqualificationを代替しない。

## 2026-09-15: 固定Exponential公式式境界

以前のsampling referenceはsoftmaxとExp除算を`double`で計算し、公式の
`max(temperature,1e-5)` floorも欠いていた。検証済み公式`inference/model.py`に合わせ、
非zero温度の意味論をFP32へ修正してfloorを追加した。RNG列そのものはsplitmix64のままで、
公式torch/CUDA RNG一致とは分離する。このため以前のtemperature 0.7生成token列は新実装の
回帰期待値として使わない。

短いrunner `run_sampling_formula_check.sh`は公式source SHA256とM1 verificationを照合し、
固定logits・固定FP32 Exp(1)でPyTorch CPU式とnativeの選択IDを比較する。review済みrun
`formula-20260915-144454-49566`はtemperature 0.7でID 3、1e-8（公式floor適用）でID 2が
一致し、identity 7件も現物一致した。[レビュー記録](../artifacts/sampling/reviewed-formula-20260915.json)。
これは2つの離散decisionだけの局所比較であり、確率bit一致、公式CUDA RNG、full logits、
生成token列をqualifiedにしない。

修正後の非zero温度full generation lifecycleは次でユーザー実行する。

```sh
TEMPERATURE=0.7 SEED=7 MAX_NEW=4 bash tools/benchmark/run_generation_lifecycle.sh
```

full backboneを3回prefillするため数分以上、数百GB規模のUnified MemoryとEngram SSD I/Oを
使い得る。他のmodel runと同時に開始しない。checkpointはread-only。build、command、diff、
samplingを含むsource/binary identity、test log、exit codeを
`artifacts/generation-lifecycle/run-日時-PID/`へ保存する。失敗時はdirectoryを保持し同じcommandで
新規runを作る。途中state resumeは未実装。結果を読んでcallback有無のtoken列、cancel、後続
requestを確認するまで、この非zero温度full generationを通過としない。

`run-20260915-144620-49901`をレビューし、exit 0、記録identity 10件の現物一致を確認した。
temperature 0.7、seed 7でtoken列`20 49 271 2107`を4 token生成し、batch、逐次callback、
開始前cancel、1 token後cancel、cancel後fresh requestが一致した。
[レビュー記録](../artifacts/generation-lifecycle/reviewed-temperature-result-20260915.json)。
これはnative splitmix64列でのfull-model lifecycle通過であり、公式torch/CUDA RNG、API伝播、
M2、性能・peak memory、256Kはqualifiedにしない。

次の独立gateはrequestの非zero temperature / seedがHTTP nonstreamとSSEで同じ結果になること。
既に確認済みのdisconnect / no-usage / local stopを反復しないparity-only modeを追加した。

```sh
DSV41_CHECK_THINKING=disabled DSV41_CHECK_MAX_TOKENS=4 \
  DSV41_CHECK_TEMPERATURE=0.7 DSV41_CHECK_SEED=7 \
  DSV41_CHECK_PARITY_ONLY=1 \
  bash tools/benchmark/run_native_model_api_check.sh
```

serverを一度loadし、同一requestをnonstream 2回とSSE 1回、full backboneで実行する。数分以上、
数百GB規模のUnified MemoryとEngram SSD I/Oを使い得るため、他のmodel runと同時に開始しない。
checkpointはread-only。結果とbuild/source/binary identityは
`artifacts/native-model-api/run-日時-PID/`へ保存する。失敗時もdirectoryを保持し、同じcommandで
fresh runを作る。request途中のresumeはない。3応答の一致はnative splitmix64 seed再現性と
API伝播のfull-path検査だが、公式torch/CUDA RNG一致、M2、disconnect、local stop、256Kを
このrunでqualifiedにしない。

`run-20260915-145214-50184`をレビューし、exit 0、identity 16件の現物一致を確認した。
nonstream 2回とSSEはcontent `Hello! How can`、finish `length`、usage 5+4=9で一致し、
native投入記録3件もtemperature 0.7 / seed 7で一致した。生成中healthは200、SSEのBUSY retryは
0。[レビュー記録](../artifacts/native-model-api/reviewed-temperature-parity-20260915.json)。
これでnative splitmix64を使うrequest-local temperature / seedの短いAPI parityは通過とする。
公式torch/CUDA RNG、seed分布品質、M2、性能・peak memory、256Kは未qualifiedのまま。

次に、同じparity-only経路へJSON stop配列を渡せるようにした。以下は実出力で既知の
`help`と、その長い重複prefix `help you`を同時に指定し、さらに非matching Unicode候補も
同じ実tokenizerで構築する。

```sh
DSV41_CHECK_THINKING=disabled DSV41_CHECK_MAX_TOKENS=128 \
  DSV41_CHECK_TEMPERATURE=0 DSV41_CHECK_SEED=0 \
  DSV41_CHECK_PARITY_ONLY=1 \
  DSV41_CHECK_STOP_JSON='["help you","help","ヘルプ"]' \
  bash tools/benchmark/run_native_model_api_check.sh
```

serverを一度loadしてnonstream 2回＋SSE 1回を実行するため、数分以上、数百GB規模の
Unified Memory / Engram SSD I/Oを使い得る。checkpointはread-only。artifactは
`artifacts/native-model-api/run-日時-PID/`。失敗時は保存済みdirectoryを調査し同じcommandで
fresh runを作る。途中resumeはない。期待するmatching stopは`help`で、3 requestともAPI usage
境界でnative cancellationになり、出力へstop文字列を含めない。これは実tokenizer上の配列・
重複prefixと非matching Unicode候補を検査するが、Unicode文字列そのものの実モデルmatch、
reasoning/toolとの組合せ、disconnect、256Kをqualifiedにしない。

初回`run-20260915-150639-50492`は3つのAPI応答自体はcontent / finish / usageが一致したが、
SSEがlocal stopを返した直後にcheckerがnative terminal 3件を要求し、まだ協調cancel中だった
3件目を待たず失敗した。nonstream 2件のcancelled terminalだけを記録した時点でserverを終了した
ためqualificationには使わない。[失敗レビュー](../artifacts/native-model-api/reviewed-stop-array-harness-failure-20260915.json)。
checkerをtimeout付きpollへ修正し、SSE完了後のnative terminalを待ってから3件を照合する。

修正後の`run-20260915-151552-50748`をレビューし、exit 0、identity 16件の現物一致を確認した。
nonstream 2回とSSEはcontent `Hi there! How can I `、finish `stop`、usage 5+7=12で一致。
3件ともnativeはcommitted 7でcancelledになった。重複候補`help you` / `help`では先に成立する
`help`を抑止し、Unicode候補`ヘルプ`は安全に非一致だった。
[レビュー記録](../artifacts/native-model-api/reviewed-stop-array-parity-20260915.json)。
実Unicode match、reasoning/toolとの組合せは引き続き別gateとする。

## 2026-09-15: native生成通知と協調キャンセル

`generate`の省略可能な`GenerationControl`で、committed tokenを同期通知する。
`on_token(token, index)`のindexは0始まりで、falseを返すとそのtokenを含めて停止する。
別threadのキャンセルはatomic flagを`is_cancelled`から読む。model step間で確認し、
実行中のGPU処理は中断しない。sample評価中にキャンセルされたtokenはcommitしない。
callbackとmodelの例外は呼出元へ伝播する。将来のC ABI接続側でERRORへ変換する必要がある。

`GenerationResult.cancelled`を追加した。stop tokenとキャンセルが同時なら`stopped`と
`cancelled`が両方trueになり、bridgeの終了理由にはcancelledを優先する。
`next_position`はprompt＋committed token数を表す論理位置であり、KVのresume位置ではない。
request stateは各呼出しのローカルで破棄する。最後のtokenの後に不要なdecodeを実行しない。
batch呼出しも同じloopを使うが、full-path速度改善・memory改善は未測定である。

native生成にもprompt＋生成予算の262144上限と有限・非負温度の検査を適用する。
これは既存API admissionとの整合であり、256K qualificationの証拠ではない。
max-new=0はモデルを実行せず空結果を返す（C ABIの0拒否契約は維持）。

checkpoint不要の`generation_loop` CTestは、通知順、stop、キャンセルの各境界、
上限と整数overflow、例外伝播、終了後の追加計算がないことを確認する。

実モデル検証はユーザーが実行する。2026-09-15の
`run-20260915-082652-38793`を確認し、exit 0、記録されたbinary/source/metadataの
SHA256全8件が現物と一致した。prompt `0 42 1000 42`、temperature=0、seed=0で
`20 49 438 223`を生成し、`next_position=8`。callback有無のtoken列一致、
開始前キャンセル、1 token通知後キャンセル、キャンセル後の新規requestでの再現を確認した。
[レビュー記録](../artifacts/generation-lifecycle/reviewed-result-20260915.json)。
非zero温度、別threadからGPU実行中のキャンセル、HTTP/SSE、性能・peak memory、
256K qualificationはこのrunでは確認していない。

```sh
bash tools/benchmark/run_generation_lifecycle.sh
```

デフォルトは4-token promptからgreedy 4 token生成し、開始前キャンセル、1 token通知後の
キャンセル、同じモデルでの新規request＋逐次通知を検査する。実prefillは合計3回で、
full backbone weightsとEngram mmapを使用し、数分以上・数百GB規模のUnified Memory圧力が
あり得る。既存モデル検証と同時に実行しない。checkpointは読み取り専用。
`TEMPERATURE=0.7 SEED=7`で固定seedのsampling比較も可能。
`MAX_NEW`は2以上、`TOKENS_FILE`は任意のtext token列を指定できる。
`PROMPT_LENGTH`を指定するとartifact内に`REPEAT_TOKEN`（既定42）をその長さだけ並べたpromptを
生成する。`TOKENS_FILE`との併用は拒否する。128-token prefill境界の最初のfull-path検査は次。

```sh
PROMPT_LENGTH=129 REPEAT_TOKEN=42 MAX_NEW=2 TEMPERATURE=0 SEED=0 \
  bash tools/benchmark/run_generation_lifecycle.sh
```

full backboneを3回、各2 prefill chunk＋decodeで実行するため数分以上、数百GB規模のUnified
Memory / Engram SSD I/Oを使い得る。他のmodel runと同時に開始しない。checkpointはread-only。
ログ・生成prompt・identityは`artifacts/generation-lifecycle/run-日時-PID/`に保存する。失敗時は
directoryを保持し同じcommandでfresh run。途中resumeはない。これは129-token境界のstateful
prefillとcallback/cancel再現を検査するだけで、256K・性能・peak memoryはqualifiedにしない。

`artifacts/generation-lifecycle/run-日時-PID/`へbuild/test log、exit code、binary/sourceと
metadataのSHA256、revision、tracked diff、実行commandを保存する。終了コード非0は失敗。
失敗時はログを確認して同じコマンドを再実行する。request途中のresumeは未実装のため
必ず新規stateで再実行する。callback有無のtoken列一致は同一native実装内の検査で、
変更前binaryや公式oracleとの比較、HTTP/SSE、disconnect、256K qualificationは含まない。
