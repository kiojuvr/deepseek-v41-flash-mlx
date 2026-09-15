# M3 native bridge plan

## Request-local stop sequence（2026-09-15、短い実モデル検証確認済み）

OpenAI Chat Completionsの`stop`を、単一文字列または最大16文字列として固定recipeの
request変換へ渡す。独自のtoken ID matcherは使わず、recipeがdecoded text上でtokenを
またぐ一致を認識し、一致文字列を出力から除く。型違いと17件以上の配列はencoding前に
400で拒否する。空文字は固定recipeと同じくstopなしとして扱う。

非streamではrecipeが一致chunkまでのtokenだけをusageへ計上し、その後にnativeが
投機生成済みだったtokenを応答へ含めない。SSEではrecipeのstop finishを検出すると
atomic cancelを立て、token receiverを閉じ、native generationを次の境界で協調停止する。
一致chunkまでのcommitted token数をusageへ使い、stop finish、usage専用chunk、`[DONE]`
を成功経路として返す。nativeのcancelled terminalを待たずにclient側streamは完了できるが、
admission permitはowner workerが戻るまで保持する。実行中GPU kernelは中断しない。

checkpoint不要の38 unit testでは、非streamの一致抑止・投機tailのusage除外、tokenをまたぐ
SSE一致・native cancel要求、不正shape、既存reasoning/tool/EOS/error経路を確認した。
実モデル検証はまだ実行していない。既知のdisabled-thinking応答を使う再現手順は次の通り。

```sh
DSV41_CHECK_THINKING=disabled DSV41_CHECK_MAX_TOKENS=128 \
DSV41_CHECK_STOP=help bash tools/benchmark/run_native_model_api_check.sh
```

モデルを1回loadし、同じstop付きrequestを非stream 2回、usage有無のSSE各1回で比較する。
別のstopなし256-token SSEを切断し、stop付きrequestへの復帰も確認する。数分以上、数百GB
規模のUnified MemoryとEngram SSD読み出しを使い得るため、他のモデル実行と同時に開始しない。
checkpointはread-only。ログ、request、応答、SSE、identity、exit codeは
`artifacts/native-model-api/run-日時-PID/`へ保存する。失敗時はdirectoryを保持し、同じcommandで
新規runを作る。request途中のresumeは未実装。結果を読んで確認するまで実モデルstopを通過と
せず、公式oracle、性能・memory、256K qualificationにも昇格しない。

初回run `run-20260915-140604-47686`は、非stream 2回とusage付きSSEまでstop文字列除去、
finish=`stop`、usage 5+7=12が一致した。そのSSEの`[DONE]`直後、native cancellationが
admissionを解放する前に次のSSEを送って503となり、runnerがexit 1で終了した。APIのbusy応答
は契約どおりだがrunnerの待機不足だったため、以後はSSE開始時の503だけを既定timeoutまで
0.2秒間隔で再試行し、server logにlocal-stop由来の`terminal=cancelled`も必須とする。
このpartial runは全検査通過として扱わず、修正版runnerの再実行結果をレビューする。

修正版run `run-20260915-141110-47945`はexit 0で、identity 16件を現物と照合した。
disabled-thinkingの`Hi`に対する`help` stopで、非stream 2回とusage有無のSSEが本文
`Hi there! How can I `、finish=`stop`、usage 5+7=12で一致した。SSEのnative terminalは
両方とも7/128 tokenで`cancelled`、stopなし切断requestは1/256 tokenで`cancelled`となり、
その後のstop付きrequestも同じ本文・finish・usageへ復帰した。SSE間のadmission解放待ちは
各1回の503 retryで完了した。[レビュー記録](../artifacts/native-model-api/reviewed-stop-result-20260915.json)。
確認範囲は短いgreedy promptと単一ASCII stopに限定し、配列・重複/Unicode境界・reasoning/tool
との組合せ、公式logits、性能・memory、長文・256Kは未qualifiedである。

このreview後、非streamも生成tokenを容量8のchannelからrecipeへ逐次渡すよう変更した。
local stopを検出するとatomic cancelを立て、owner workerのterminalをjoinしてからHTTP応答を返す。
これにより従来の「nativeはEOSまで11 token、API usageはstopまで7 token」という投機tailを、
通常は7 token地点の協調停止へ縮め、応答返却時のadmission解放も保証する。cancelが次の境界に
間に合わずnative側がstop/lengthを先に確定するraceも、event列・committed数を検査した上で
有効terminalとして扱う。stopが一致しない場合は従来のnative terminalと全committed usageを使う。
checkpoint不要の42 test（stop配列、Unicode、重複prefixを含む）は通過した。

incremental変更後のrun `run-20260915-142311-48511`もexit 0、identity 16件一致を確認した。
nonstream 2回、usage有無のSSE、切断後復帰のstop付き5 requestすべてが、API usageと同じ
7/128 tokenでnative `cancelled`となった。本文、finish=`stop`、usage 5+7=12は全経路で一致。
stopなし切断は1/256 tokenでcancelされ、後続requestは同じ結果へ復帰した。これにより以前の
nonstream 11-token投機tailはこの実入力で解消した。[incremental stopレビュー](../artifacts/native-model-api/reviewed-incremental-stop-result-20260915.json)。
単一ASCII stopの短い実モデル範囲に限り、nonstream/SSE incremental stopを通過とする。

## Tool call API接続（2026-09-15、実モデル検証待ち）

Chat Completions request全体を固定`deepseek-recipe`の`ChatCompletionRequest::convert`へ
渡し、変換後のconversationをprompt encoding、同時に得た`ParsingOptions`を非streamと
SSEの両方へ渡す。`tools`、`tool_choice`、assistantの過去`tool_calls`、対応するtool結果は
公式recipeが検証・renderする。native runtimeはtoken IDの実行だけを担当する。
画像sourceはvision runtime未接続のため、公式変換後にも明示的に拒否する。caller側の
schema enforcementを必要とするfunction `strict: true`も、黙って無視せず明示的に拒否する。

checkpoint不要の短いテストでは、required choiceがtool parserの初期stageを選ぶこと、
thinking enabledとの不正な組み合わせをrecipeが拒否すること、過去のtool call/resultを
含むpromptを生成できることを確認した。固定DSML出力から、非stream応答とSSE deltaの
両方が同じfunction name/JSON argumentsを構築し、`finish_reason=tool_calls`、committed
tokenに基づくusageを返すことも確認した。`tool_choice:none`で定義とparserが無効化され、
named choiceで対象toolだけが残ることも確認した。実モデルでのtool生成結果はまだ未確認である。
invalid type、存在しないnamed choice、重複tool名もencoding前に拒否する。
実モデルrun `run-20260915-105315-43155`では、tool定義を含むpromptが128 tokenを超え、
native encoderの単一呼び出し上限により`invalid encoder request state`となった。生成器の
prefillを128-token chunkへ分割し、同じstateで位置を進める修正を入れた。下記runnerで
この修正後の実モデル経路を再確認する。

実モデル検証は次の専用runnerで行う。

```sh
bash tools/benchmark/run_native_tool_api_check.sh
```

modelを1回loadし、最大128 tokenのrequired tool callを非streamとSSEで各1回生成して
function/arguments/finish/usageを比較する。続いて、そのassistant tool callとtool結果を
履歴に含めた16-tokenの応答を1回生成する。数分以上、数百GB規模のUnified Memoryと
Engram SSD読み出しを使い得るため、他のモデル実行と同時に開始しない。checkpointは
read-onlyで、Pythonはoffline検証orchestrationに限る。

ログ、request、HTTP JSON、SSE、identity、終了codeは
`artifacts/native-tool-api/run-日時-PID/`へ保存する。失敗時はそのdirectoryを保持し、
通常は同じcommandで新しいrunを作る。build/testまで成功済みで検証部分だけを再開する場合は
`DSV41_SKIP_BUILD=1 bash tools/benchmark/run_native_tool_api_check.sh`を使える。
この検証は短いtool API経路に限定し、公式logits oracle、性能、peak memory、長いtool出力、
256K runtime qualificationを満たしたとは扱わない。結果を読んで確認するまで通過としない。

実モデルrun `run-20260915-132520-46314`はexit 0で、identity 18件を照合した。286-token
promptから34 tokenの`get_weather` callを返し、非streamとSSEでfunction name、Tokyoの
arguments、`tool_calls` finish、usage（286+34=320）が一致した。SSEはusage専用chunk 1回と
`[DONE]` 1回を記録した。assistant tool callとtool結果を含む後続requestも200で本文を返し、
tool callを再発行しなかった。[tool検証レビュー](../artifacts/native-tool-api/reviewed-tool-result-20260915.json)。
この結果は単一toolの短い実モデル経路の確認であり、複数call、strict schema、性能・memory・
公式oracle・256K qualificationは未確認である。

## Reasoning effort（2026-09-15、lowの短い実モデル検証確認済み）

`run-20260915-101355-42364`をレビューし、exit 0とidentity全16件の一致を確認した。
thinking指定なし・reasoning_effort=lowの4-token生成は、非streamとusage true/falseの
SSEでreasoning `Hmm, the`、finish=`length`、usage=31+4=35が一致した。
切断用enabled requestは1/256 tokenでcancelされ、後続low requestも同じ出力とusageを
再現した。[レビュー記録](../artifacts/native-model-api/reviewed-reasoning-low-20260915.json)。
low以外の実モデル経路、EOS/本文遷移、公式oracle、性能・memory・256Kは未qualified。

OpenAI互換の`reasoning_effort`をrequest-local prompt設定へ接続した。固定recipeと同じく
`minimal/low`はlow（prompt score 50）、`medium/high`はhigh（75）、`xhigh`はxhigh
（75）、`max`はmax（100）へ正規化する。`none`はthinking指定がなければthinkingを
無効化する。明示的な`thinking.type`を優先し、enabled＋noneはhigh、disabled＋maxは
thinking無効となる。この優先順位・全mapping・rendered promptを固定recipeの公式request
変換と比較した。

HTTP request全体はrecipe adapter境界で公式型へ変換し、defaultのunconnected serverは
optional recipe dependencyなしでbuild/testできる。context admission、実prompt、
非stream/SSE parserは同じrequest-local変換結果を使う。
thinking budgetは未対応で、`thinking`内の未知fieldとして拒否する。

短い実モデル検証を行う場合は次を実行する。

```sh
DSV41_CHECK_THINKING=omitted DSV41_CHECK_REASONING_EFFORT=low \
DSV41_CHECK_MAX_TOKENS=4 bash tools/benchmark/run_native_model_api_check.sh
```

モデルロード1回、低effort promptで非stream/SSE usage true/false、切断と後続requestを
検査する。数分以上・数百GB規模のUnified Memory圧力があり得るため他のモデル実行と
同時に開始しない。checkpointはread-only。ログ/identity/失敗復旧は従来の同script契約に
従い、途中state resumeは行わない。実モデルで確認した範囲はlowの短いgreedy requestに
限定する。この成功を他effort、公式logits oracle、性能・memory・256Kの判定には使わない。

## SSE usageオプション（2026-09-15）

`run-20260915-100703-41960`をレビューし、exit 0とidentity全16件の一致を確認した。
thinking disabledの短いEOS完了で、include_usage=trueは通常chunkの`usage: null`、
finish後の`choices: []` usage専用chunk 1回、`[DONE]` 1回となった。usageは5+11=16。
falseは全12 eventでusage fieldを省略し、本文・finishは非stream/trueと一致した。
切断request 5は`cancelled committed=1 budget=256`、後続request 6はHTTP 200で同じ本文と
usageを再現した。[レビュー記録](../artifacts/native-model-api/reviewed-stream-usage-20260915.json)。
確認範囲は短いgreedy promptであり、低速client・長文・性能/memory・256Kは未qualified。

`stream_options: {"include_usage": true}`を接続した。trueの場合は通常のSSE chunkに
`usage: null`を付け、finish chunkの後に`choices: []`と全committed tokenのusageを持つ
専用chunkを1回返し、その後`[DONE]`で終える。省略/falseでは全chunkのusageを省略する。
非stream応答のusageは従来どおり返す。stream=falseでstream_optionsを指定すると400、
型違い/未知fieldはJSON抽出時に拒否する。

固定recipeはfinishとusageを同じchunkに出すため、SSE transportでsnapshotを分離する。
prompt、token execution、recipeの本文/reasoning parserは変更しない。
native/parserエラー時にはusage専用chunk・成功finish・`[DONE]`を生成しない。
EOSや未表示の末尾UTF-8もnative committed countで数える契約は維持する。

短いテストでtrue/falseのusage有無、専用chunkの順序とcount、非streamでの不正指定、
型違い/未知field拒否を追加した。変更後の短い実モデル結果は上記で確認済み。
既存の成功レビューは変更前のwire形式の証拠として保持する。

任意の実モデル再確認は同じ入口で行う。

```sh
DSV41_CHECK_THINKING=disabled DSV41_CHECK_MAX_TOKENS=128 DSV41_CHECK_REQUIRE_EOS=1 \
  bash tools/benchmark/run_native_model_api_check.sh
```

今回からusage true/falseのSSEを各1回比較するため、前回よりprefill/generationが1回増える。
modelロード1回、非stream計3回、正常SSE2回、切断用SSE1回。数分以上・数百GB規模の
memory圧力があり得る。checkpointはread-only。既存ログに
`completion-stream-no-usage.sse`も保存する。失敗時は同じcommandで新規runを作り再実行し、
途中state resumeは行わない。timeoutやidentity/結果確認手順は従来と同じ。
この変更や短いテストを性能・memory・256K qualificationへ昇格させない。

## Thinking切替（2026-09-15、短い実モデル検証確認済み）

`run-20260915-093922-41623`をレビューし、exit 0とidentity全16件の一致を確認した。
thinking disabled、128-token予算の`Hi`に対し本文
`Hi there! How can I help you today?`を返し、reasoning fieldなし、EOSによる`stop`、
usage=prompt 5 + completion 11 = total 16となった。非stream2回・SSE・切断後の
新規requestで出力とusageが一致した。途中のthinking enabled requestは1 tokenで
`cancelled`となり、その後disabled設定へ戻って同じ本文を再現した。
[レビュー記録](../artifacts/native-model-api/reviewed-thinking-disabled-20260915.json)。
この短いpromptでのEOS完了は公式oracle・長文・性能/memory qualificationを代替しない。

`thinking: {"type":"enabled"}` / `{"type":"disabled"}`を受け付ける。
省略時は従来どおりenabled。unknown type/fieldはJSON抽出時に拒否する。
requestごとにimmutable tokenizerを共有するencoderをcloneし、thinking設定を
context admission、実prompt encoding、非stream/SSEのrecipe parserへ同じ値で渡す。
共有encoderを変更しないので、切断試験などのenabled requestとdisabled requestの間で
設定が残らない。reasoning_effortとtoolsは後続節のとおり接続済み。stop等は未接続。

固定recipeの公式request変換とenabled/disabled両方のrendered promptを比較した。
checkpoint不要テストでdisabled出力がreasoningではなくcontentへ入り、EOSを表示せず
usageへ含めることを非stream/SSEで確認する。実モデルの確認範囲は上記の短いpromptに限定する。

```sh
DSV41_CHECK_THINKING=disabled DSV41_CHECK_MAX_TOKENS=128 DSV41_CHECK_REQUIRE_EOS=1 \
  bash tools/benchmark/run_native_model_api_check.sh
```

モデルロード1回、同じ`Hi` requestの非stream2回・SSE1回・切断後の非stream1回を
それぞれ最大128 tokenまで生成する。本文が非空、reasoningが空、EOSによるstop終了と
各応答の一致を必須にする。切断用の別requestだけはthinking enabled/予算256に固定し、
最初の内容deltaで切断する。これにより同じmodelで設定の切替と復帰も確認する。
128 token以内にEOSへ到達しなければ検証失敗として記録し、自動的に予算を拡大しない。
数分以上・数百GB規模のmemoryとEngram SSD読み出しがあり得るため他のモデル実行と
同時に開始しない。checkpointはread-only。Pythonはoffline検証orchestrationのみ。

ログ/応答は既存`artifacts/native-model-api/run-日時-PID/`へ保存し、request.json/result.jsonに
thinkingと生成予算を残す。失敗時はtest.log、server.log、応答JSON/SSEを確認する。
同じcommandで新規state・新規runを作り再実行する。途中state resumeは未提供。
既定timeoutは起動/各request 1800秒（`DSV41_CHECK_TIMEOUT`）。
EOS到達だけで公式oracle一致、性能・peak memory・256K qualificationを満たしたとは扱わない。

## SSE・切断時キャンセル（2026-09-15、短い実モデル検証確認済み）

`run-20260915-093146-41241`のexit 0とidentity全16件を照合した。非streamとSSEで
reasoning `We need answer to`、finish=`length`、usage=31+4=35が一致した。
SSEは4つの内容delta、finish/usage 1回、`[DONE]` 1回を記録した。
256-token予算のrequest 4を最初の内容deltaで切断し、native logで
`terminal=cancelled committed=1 budget=256`を確認した。後続request 5はHTTP 200で
同じ出力とusageを再現し、healthも200だった。
[SSE検証レビュー](../artifacts/native-model-api/reviewed-sse-result-20260915.json)。
実モデルのEOS/本文への遷移、prefill中・非stream切断、持続的な低速client、
性能・peak memory・256Kは引き続き未qualifiedである。

`native-model`の`stream=true`をSSEへ接続した。native owner threadのTOKENを容量8の
channelへ渡し、非streamと同じrecipe processorでreasoning/本文・EOSを処理する。
成功時はfinish/usageと`[DONE]`を返す。HTTPヘッダ送信後のnative/parser失敗は
`event: error`とJSON errorで明示し、成功finishや`[DONE]`を出さない。
usageは全committed tokenを数える。独自のtool/reasoning parserは追加しない。
`stream_options.include_usage`は接続済みで、trueではusage専用chunk、falseではusage省略を返す。

SSE body破棄はatomic cancel flagを立て、token receiverを閉じる。満杯channelで待つ
native callbackも解除され、false返却で協調停止する。非stream request futureにも
drop guardを追加した。新しいC ABI `dsv41_bridge_submit_cancellable`は同じatomic flagを
owner threadからgeneration境界でpollするため、最初のtoken通知前にもcancelを確認する。
既存submitはpollなしの互換wrapperとして維持する。FFIにはRust参照を保持させず、
context/flagの寿命は同期submit中に限定する。モデル自体のthread affinityは維持する。

キャンセルは実行中のGPU kernelやprefillそのものを中断せず、次のgeneration境界で適用する。
HTTP admission permitはnative submitとrequest state破棄が終わるまで保持する。
workerのterminal理由・committed数・予算をserver.logへ記録する（token内容は記録しない）。
channel以外にnative GenerationResultと検査用token列は最大生成予算に比例して保持する。
低速clientで任意長の未送信token queueが増殖する構成にはしない。長時間の低速client・
memory・socket切断検知の遅延は未qualifiedである。

短いテストで正常SSE、reasoning/本文・EOS/usage、エラー時の偽成功抑止、未pollのbodyを
破棄した場合のcancel、満杯channel解除、C ABIのprefill前cancelを確認した。
実モデルで確認した範囲は上記の短いSSEと最初の内容delta後の切断に限定する。

再検証は同じ入口を使う。

```sh
bash tools/benchmark/run_native_model_api_check.sh
```

更新後はモデルロード1回、非stream 4 token生成2回、SSE 4 token生成1回、
256-token予算のSSEを最初の内容deltaで切断し、その後4 tokenの新規requestを実行する。
SSEの本文/reasoning/finish/usageを非streamと比較し、切断後はBUSYだけをretryして復帰を
確認する。server.logの`cancelled`とcommitted<256も必須とし、単なる予算終了をcancel成功と
扱わない。数分以上・数百GB規模のmemory圧力があり得る。checkpointはread-only。
既存のログ/identity/timeout/失敗時の新規run手順に加え、completion-stream.sseと
disconnect.sse、after-disconnect.jsonを保存する。途中stateのresumeは未提供。
SSE/disconnectの実モデル結果を読んで確認するまでM3 exitへ昇格しない。

## Rust非stream HTTP接続（2026-09-15、初期接続の履歴）

`server`の`native-model` featureはMLX共有bridgeとrecipe parserへ接続する。
`DSV41_NATIVE_MODEL=1`、`DSV41_CHECKPOINT`、`DSV41_TOKENIZER`で起動し、
`DSV41_SUMMARY`/`DSV41_METADATA`/`DSV41_PROVENANCE`は従来artifactを既定値とする。
tokenizerのEOS IDをcheckpoint configと照合しnative stop tokenへ渡す。
設定不足・モデルロード失敗は起動失敗とし、shellへの暗黙fallbackは行わない。
従来の`native-bridge` feature/環境変数はshell検証用として残す。

初期対象はtext message、temperature、seed、max_tokensによる非stream生成だった。
promptと出力はrecipe `StreamProcessor`と`ChatCompletionResponse`で組み立て、本文とreasoningを
分離し、EOSを表示せず、usageに全committed token数を記録した。thinking、reasoning effort、
SSE、toolsはこの後に接続され、現在の契約は上の各節に記録している。stopは未接続である。

モデルのロード・submit・破棄は専用OS thread `dsv41-model`に固定する。
`spawn_blocking`は入力encoding・専用threadへの依頼/結果待ち・recipe応答処理を担当し、
MLX handleは移動しない。admission permitを完了まで保持し、競合requestは`503 runtime_busy`。
C ABIのrequest ID・通知index・終端・status・生成数を照合し、欠落や不整合、native例外は
HTTP 500にする。FFI callbackのpanicを捕捉しC++へunwindさせない。

この初期段階ではSSE、disconnect時のworker停止、backpressureは未接続だった。現在は上記の
bounded SSE/cancel経路へ置き換わっている。この履歴だけをM3 exitやRELEASEの証拠としない。
単体テストはrecipeのreasoning/本文分離、EOS表示抑止・usage、length、event不整合、
callback panic、HTTP errorとadmission解放を対象にし、実モデル推論を行わない。

初回の`run-20260915-084843-40205`はexit 1、両completionがHTTP 500で
`There is no Stream(gpu, 0) in current thread.`となった。identity全13件を修正前に照合した。
main threadで作成したMLX modelをblocking poolの別threadで実行していたため、
専用owner threadへ変更した。`NativeBridge`のunsafe Send/Syncも削除し、型としてthread間
移動を禁止した。owner threadの起動失敗は起動元へ返し、request処理でRust panicが起きた
場合は再利用せずthreadを終了する。通常のnative ERRORでは後続requestを処理可能。
worker終了時はchannelを閉じてjoinし、modelを作成thread上で破棄する。
[失敗レビュー](../artifacts/native-model-api/reviewed-failure-20260915.json)。
checkpoint不要の14テストが通過した。修正後の実モデルHTTP検証
`run-20260915-090725-40519`はexit 0、identity全14件が現物と一致した。
2 requestともHTTP 200、reasoning_content=`We need answer to`、contentは空、
finish_reason=`length`、usageはprompt=31 / completion=4 / total=35で一致した。
応答IDは別々で、healthは200、stream要求は501となった。
以前のGPU streamエラーはこのrunでは再現しなかった。
[成功レビュー](../artifacts/native-model-api/reviewed-result-20260915.json)。
これは4 tokenのreasoning出力までの検証で、本文への遷移、EOSによる完了、
SSE/disconnect、性能・memory・256Kは未qualifiedである。

### 実モデルHTTP検証（ユーザー実行）

```sh
bash tools/benchmark/run_native_model_api_check.sh
```

モデルロード1回と`Hi`に対するgreedy 4 tokenの独立requestを2回実行する。
recipeのthinking promptを使い、応答schema・reasoning/本文・finish reason・usage・再現性、
生成中のhealth応答とstream要求の明示拒否を確認する。C ABIとのtoken ID比較は含まない。
数分以上、数百GB規模のUnified Memory圧力とEngram SSD読出しがあり得るため、
他のモデル実行と同時に開始しない。checkpointはread-onlyである。

ログと`result.json`、全応答JSON、identity、revision/diff、設定を
`artifacts/native-model-api/run-日時-PID/`へ保存する。既定portは18082、
`DSV41_SMOKE_PORT`で変更可能。起動と各requestのtimeoutは既定1800秒、
`DSV41_CHECK_TIMEOUT`で変更できる。失敗時もログを保持し、起動したserverを終了する。
unit/build失敗は各log、モデル起動失敗は`server.log`、HTTP失敗は`test.log`/応答JSONを確認する。
offline Cargo cacheに依存がなければ依存取得後に同じscriptを再実行する。
途中stateのresumeは未提供で、再実行は新規state・新規runとする。
Pythonはこのユーザー実行検証のorchestrationにのみ使い、serverはRust/C++/Metalで動く。

## 2026-09-15: opt-in C ABI model bridge

`dsv41_runtime_bridge_mlx`共有libraryに`dsv41_bridge_create_model`を追加した。
checkpoint、M1 summary、Engram metadata/provenance、stop token ID列を明示して作成する。
metadataのSHA256をprovenanceに照合し、WeightCatalogとmodelをbridgeが所有する。
weights/backingは既存native referenceと同じread-only経路で、Python依存はない。
header/size検査とmetadata照合は全checkpoint payloadの再attestationを代替しない。

従来の`dsv41_bridge_create`とstatic `dsv41_runtime_bridge`は未接続shellを維持する。
`native-bridge`構成のRust serverはshellへ接続する。モデルHTTP接続は上記の
`native-model`構成で明示的に有効化する（実モデルAPI検証待ち）。
以下の過去のshell検証記録もその範囲で有効である。

接続したC ABIは同期submitでnative `TextGenerationReference`を実行し、TOKENのindexは
0始まり、FINISHED/ERRORのindexは通知済みtoken数になる。終了理由は`length`、`stop`、
`cancelled`。callbackのnonzero返却と別threadのcancelを協調停止へ変換する。
requestごとのKV/stateはgenerate終了時に破棄され、継続resumeはまだ提供しない。
各requestのRNGは指定seedから開始する。

invalid/busyはrequest ID=0でイベントを出さずに拒否する。受付済みrequestは終端イベントを
1回だけ返す。C++例外は`runtime_error` ERRORへ変換しC境界を越えさせない。
callbackは例外を投げてはならず、Rust側もpanicをFFIへ伝播させない実装が必要である。
終了済み/不明IDのcancelは`NOT_FOUND`。terminal送信の確定後のcancelも同様である。
create/submitはbridge library内でプロセス単位に直列化し、競合は待ち行列へ積まず`BUSY`
（createはNULLとdiagnostic）で拒否する。callbackからdestroyしてはならない。
destroyは全submit/cancelが終了した後に呼ぶ。MLX library外の並行model操作は未qualified。

checkpoint不要の`bridge_lifecycle` CTestでTOKEN/terminal accounting、callback cancel、
別thread cancel、reentry拒否、stop、model例外と後続requestを確認した。
既存`bridge_smoke`も通過し、MLX共有libraryと実モデル検証binaryのbuildを確認した。
これらを実モデルのC ABI生成成功とは扱わない。

### 実モデル検証（ユーザー実行）

`run-20260915-083741-39444`のreference/test logとexit 0を確認し、identityのSHA256全16件を
現物と照合した。直接nativeとC ABIの通常requestで`20 49 438 223`が一致した。
request 1は`length`/4 token、request 2は`cancelled`/1 token、request 3は`length`/4 tokenで、
キャンセル後の新規requestも同じtoken列を再現した。terminal回数、通知indexとcount、
request ID、完了済みrequestへのcancel拒否も実行binaryの検査を通過した。
[レビュー記録](../artifacts/bridge-model/reviewed-result-20260915.json)。
これはgreedyの短いC ABI full-model検証であり、Rust/HTTP接続・性能・256Kの判定ではない。

```sh
bash tools/benchmark/run_bridge_model_check.sh
```

直接native生成を別processで実行し、そのtoken列をC ABIの通常生成と比較する。
続けて最初の通知でcancelし、同一bridgeで新規requestを生成してtoken列・終了理由・
通知数・request IDを確認する。デフォルトはprompt `0 42 1000 42`、greedy、seed=0、
生成4 token。`TOKENS_FILE`、`MAX_NEW`（2以上）、`TEMPERATURE`、`SEED`を指定できる。
モデルロード2回、prefill計4回で数分以上、数百GB規模のUnified Memory圧力とEngram SSD
読み出しがあり得る。他のモデル検証と同時実行しない。checkpointは変更しない。

`artifacts/bridge-model/run-日時-PID/`にconfigure/build/reference/test log、終了コード、
revisionとtracked diff、実行command、入力/期待token列、binary/source/metadataのSHA256を
保存する。非0は失敗で、ログを確認して同じcommandを再実行すると新規runを作る。
途中stateのresumeは提供しない。結果は読み取り・identity照合後にレビュー記録へ保存する。
この比較も公式oracle、HTTP/SSE parser、disconnect/backpressure、性能・peak memory、
256K qualificationの判定には使わない。次はRustのモデル接続とrecipe出力parserを接続する。

M2のcanonical MLX reference接続を土台に、M3ではAPI protocolとC++ runtimeを
接続する。recipeはprompt/message encodingとstream parsingを所有し、C++は
token execution、state、sampling、committed token eventを所有する。

固定recipe sourceは次で準備・検証する。

```sh
bash tools/benchmark/prepare_recipe.sh
```

この処理は`third_party/deepseek-recipe`へblob-filtered checkoutを作り、revision・
tree・Cargo metadataを`artifacts/recipe/prepare-.../`へ保存する。取得にはネットワーク
と数百MB程度の一時領域を使う。既存checkoutは上書きせず、失敗時は`RECIPE_DIR`で
新しい出力先を指定して再実行する。Cargo依存の有効化はsource確認後に行う。

2026-09-15にrevision `8cadfede7063c896b944e7bae05daa3549ae97ea`、tree
`3ed5db20847e09a9f3d157660e27c8d2c3141fc3`を取得した。provenanceは
`artifacts/recipe/prepare-reviewed-20260915.json`へ保存し、checkout本体はvendorしない。

## 接続順序

1. Rust側でrecipe出力をtoken ID列へ固定し、request identityとtoken countを記録する。
2. `dsv41_request_t`へborrowed token列を渡し、C++ sessionをrequest単位で生成する。
3. `DSV41_EVENT_TOKEN`をcommitted tokenだけに限定し、finish/errorを必ず終端イベントにする。
4. callbackのnonzero返却、client disconnect、backend failureを協調cancelへ変換する。
5. SSE adapterをRust側に置き、event stringの寿命とusage/finish accountingをコピーして管理する。

## M3 exit条件

- plain textのprefill + incremental decodeがAPI経路でnative generationと同一tokenになる。
- committed token数、usage、finish reason、state positionがrequestごとに一致する。
- literal special tokenをprotocol層で誤拒否しない。
- cancel / disconnect / backpressure後にworkerとsession stateが残らない。
- backend errorがHTTP成功へ変換されず、SSE/non-stream双方で明示される。

2026-09-15: 接続準備としてnative `TextGenerationReference`へcommitted token callbackと
協調cancelを追加した。batchと逐次通知は共通の生成loopを使う。
[契約・実モデル検証手順](sampling-reference.md#2026-09-15-native生成通知と協調キャンセル)
を参照。C ABI shellの置換、recipe出力parser、HTTP/SSEへの接続は引き続き未完了。
ユーザー実行の`run-20260915-082652-38793`でgreedy 4 tokenのcallback有無の一致、
開始前・1 token後のキャンセル、キャンセル後の新規requestの再現を確認した。
exit 0とidentity全8件も照合済みで、[レビュー記録](../artifacts/generation-lifecycle/reviewed-result-20260915.json)
に保存した。これはnative生成のlifecycle検証であり、M3 exit判定ではない。

`include/dsv41/runtime_bridge.h`はこの契約のABIドラフトである。ABI実装とrecipe
依存を有効化するまではdeveloper serverをRELEASEと呼ばない。

`dsv41_runtime_bridge`は現在、ABI smoke用のunconnected shellを提供する。
create/submit/destroyとinvalid argument検査を実装し、submitは推論を実行せず
`runtime_unavailable` error eventを返す。native generationを返すbridgeへ置換する
まで、このshellの成功をAPI inference成功として扱わない。

Rust側には`server` crateの`native-bridge` featureで有効化する薄い所有権wrapperを
追加した。default buildは従来どおり純Rust stubで、feature有効時のみC ABIへ接続する。
token列のborrow、request ID、callback eventの文字列copy、cancel結果をRust型へ
変換する。現在のbridge shellはunavailableを返すため、これはABI/lifecycle検査で
あり、native generation接続の証拠ではない。

`dsv41-bridge-smoke`はC++からshellを呼び、ABI version拒否、invalid argument、
unavailable error event、request ID、cancel応答を検査する。MLXやcheckpointを
ロードしないため、短いCI契約検査として実行できる。

CTestの`bridge_smoke`にも登録し、`ctest --test-dir build-mlx -R '^bridge_smoke$'`
で単独実行できる。

Rust featureをlibraryへ接続する場合は、C++ libraryを先にbuildしてから次を実行する。

```sh
cmake --build build-mlx --target dsv41_runtime_bridge -j 4
DSV41_BRIDGE_LIB_DIR="$PWD/build-mlx" \
  cargo check --manifest-path server/Cargo.toml --features native-bridge
```

feature未指定の`cargo check`は外部libraryを要求しない。recipe token/event adapter
が接続されるまでは、feature buildもABI検査の範囲に留まる。

checkpoint同梱の公式encoding fixtureは次で生成できる。

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/export_recipe_fixtures.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --output artifacts/recipe/encoding-fixtures.json
```

これはchat、reasoning effort、tool、structured image、literal image marker拒否を
記録するoffline fixtureであり、production APIの成功判定には使わない。

fixture token IDsをnative generationへ通す場合は、次を実行する。

```sh
MAX_NEW=16 TEMPERATURE=0 SEED=0 \
  bash tools/benchmark/run_recipe_native_generation.sh chat
```

引数は`chat`、`thinking`、`tool`、`structured_image`から選べる。full backboneを
実行するため数分以上かかる可能性があり、run artifactとidentityを
`artifacts/text-generate/run-.../`へ保存する。これはrecipe renderingとnative
token pathの接続検査であり、公式生成oracleではない。

chat fixtureの実行結果は`artifacts/recipe/native-generation-chat-reviewed-20260915.json`
へ保存した。5 tokenの公式rendered promptから16 tokenをcommitし、`next_position=21`
となることを確認済みである。これはrecipe encodingとnative generationの接続証拠で
あり、Rust API bridgeや公式生成oracleの証拠ではない。

thinking fixtureも34 tokenのrendered promptから16 tokenをcommitし、
`next_position=50`となった。記録は
`artifacts/recipe/native-generation-thinking-reviewed-20260915.json`に保存した。

tool fixtureも45 tokenのDSML promptから16 tokenをcommitし、`next_position=61`
となった。記録は`artifacts/recipe/native-generation-tool-reviewed-20260915.json`
に保存した。

structured image fixtureはtext native pathが`encoder reference requires text token IDs`
でexit 1となった。画像pixel処理へ誤って流入しない明示的な境界であり、
`artifacts/recipe/native-generation-structured-image-reviewed-20260915.json`へ記録した。
vision projectorとimage spanはM6で別接続する。

fixtureの再現性は次で確認できる。

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/verify_recipe_fixtures.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --fixture artifacts/recipe/encoding-fixtures.json
```

sourceとtokenizerが変わるとexact比較が失敗する。

2026-09-15の再検証では5 fixtureがexact、encoding source SHA256は
`502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1`、tokenizer
SHA256は`c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`で一致した。

Rust adapterのfeature付きunit testもC++ shellへリンクして実行し、error eventの
kind/code、request ID、callback変換を確認済みである。

2026-09-15にC++ libraryを`build-mlx`から参照したfeature付き`cargo build`も成功した。
このbuildはserver handlerへbridgeを接続しておらず、実token生成の証拠ではない。

recipe core/encoding/protocol crateの固定checkoutがコンパイル・テスト可能か確認する場合は、次を実行する。

```sh
bash tools/benchmark/run_recipe_core_checks.sh
```

このチェックは`deepseek-recipe-core`、`deepseek-recipe-encoding`、`deepseek-recipe`
だけを対象にし、image/python bindingsやnative runtimeは含めない。Cargo依存の取得と
コンパイルに数分以上、数GB未満の一時領域を使う可能性がある。ログ、revision、tree、
終了コードは`artifacts/recipe/checks-<timestamp>-<pid>/`へ保存され、失敗時は
`test.log`を確認して同じcheckoutで再実行できる。別checkoutを試す場合は
`RECIPE_DIR=/path/to/deepseek-recipe bash tools/benchmark/run_recipe_core_checks.sh`
とする。終了コード0だけではnative API接続や生成oracleのqualificationを意味しない。

2026-09-15の実行では固定revision/treeに対してコンパイルとcore unit test 5件が成功した。
結果は`artifacts/recipe/checks-reviewed-20260915.json`に記録した。これはrecipeの
protocol部品がビルド可能であることの確認であり、native runtimeへの接続判定ではない。

JSON messageからrecipe `Conversation`への変換は`recipe-adapter` featureに隔離した。
境界テストは次で実行できる。

```sh
cargo test --manifest-path server/Cargo.toml --features recipe-adapter
```

このadapterはplain textとtext content arrayを扱い、literal image markerを保持する。
画像blockはvision実装へ渡す前に`ImageContentRequiresVision`で停止する。tokenizer、
native bridge、HTTP handlerへの接続は次段階で行う。

adapterには`encode_file`も用意し、呼び出し側が指定したtokenizer JSONでrecipeの
renderingとtokenizeを一括実行できる。tokenizerはrequestごとに読み込む前提ではなく、
HTTP application stateで所有してnative bridgeへtoken列をborrowする構成にする。

`RecipeEncoder`はtokenizerを`Arc`で一度だけ所有し、複数requestから共有できる。
生成requestの寿命中だけ返却`Vec<u32>`を保持し、そのsliceをbridge requestへborrowする。

serverを`--features recipe-adapter`で起動し`DSV41_TOKENIZER`にtokenizer JSONを指定すると、
application stateの初期化時に一度だけtokenizerをロードする。HTTP requestはこのencoderで
検証・encodingされ、失敗は`recipe_encoding_error` (400)として返る。encoding成功後も
bridgeが未接続なら従来どおり`runtime_unavailable` (501)となる。

`NativeRuntimeBackend`はrecipe encoding済みのtoken列、`max_tokens`、`temperature`、
`seed`をC ABIへ渡し、TOKEN eventを収集するfeature-gated実装である。現在のC++ shellは
unavailableを返すため、接続テストは501境界の確認に留まる。実モデルbridgeへ昇格する
前に、event終端・usage・cancelの統合テストを追加する。

developer serverでこの経路を有効にするには、C++ bridge libraryをbuildした上で次の
環境変数を指定する。

```sh
DSV41_NATIVE_BRIDGE=1 \
DSV41_TOKENIZER=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/tokenizer.json \
DSV41_BRIDGE_LIB_DIR="$PWD/build-mlx" \
cargo run --manifest-path server/Cargo.toml --features 'native-bridge recipe-adapter'
```

現時点ではshell bridgeがunavailableを返すため、healthは`native-bridge`、completionは
`501 runtime_unavailable`となる。tokenizer未指定またはfeature未指定時はunconnectedへ
フォールバックする。

既存のAPI smokeをnative構成で実行する場合は、次のようにfeatureと環境変数を渡す。

```sh
DSV41_CARGO_FEATURES='native-bridge recipe-adapter' \
DSV41_NATIVE_BRIDGE=1 \
DSV41_EXPECTED_RUNTIME=native-bridge \
DSV41_TOKENIZER=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/tokenizer.json \
DSV41_BRIDGE_LIB_DIR="$PWD/build-mlx" \
bash tools/benchmark/run_api_smoke.sh
```

結果は`artifacts/api-smoke/run-.../`に保存される。shell bridgeではchat statusが501で
あることを維持し、healthのruntimeだけ`native-bridge`になることを検査する。

2026-09-15にこの構成を実行し、healthが`native-bridge`、chat/optionsが501、model
未検出が404、入力制約が400となることを確認した。レビュー記録は
`artifacts/api-smoke/native-bridge-reviewed-20260915.json`に保存した。

配置確認を含むpreflightは次で実行できる。

```sh
bash tools/benchmark/run_native_bridge_preflight.sh
```

checkpoint/tokenizer/bridge libraryの存在を検査し、結果を
`artifacts/api-smoke/preflight-.../`へ保存する。checkpointは変更せず、Cargo buildは
起動しない。実runtime置換後も同じ入口でAPI lifecycleを再検証できる。

初回preflightは成功し、前提条件、native-bridge health、501/404/400のAPI契約を確認した。
記録は`artifacts/api-smoke/native-bridge-preflight-reviewed-20260915.json`に保存した。

C ABI shellもserverと同じく空prompt、`max_new_tokens=0`、非有限温度、負温度を
`DSV41_BRIDGE_INVALID_ARGUMENT`で拒否する。bridge smokeでこの境界を検査している。
生成上限`DSV41_BRIDGE_MAX_NEW_TOKENS=262144`も同じ境界で適用する。
recipe encoderが有効なserverでは、encoded prompt token数と`max_tokens`（未指定時16）の
合算も256K以下に制限し、超過は`context_length_exceeded` (400)とする。これはtotal-context
admissionであり、recipe adapter未使用のunconnected stubには適用されない。
C ABIでは`input_token_count + max_new_tokens`も同じ256K上限で検査し、API層を迂回した
requestでもtotal-context契約を維持する。

native-bridge smokeで短いpromptに`max_tokens=262144`を指定し、合算超過が
`400 context_length_exceeded`となることを確認した。結果は
`artifacts/api-smoke/total-context-reviewed-20260915.json`に保存した。
