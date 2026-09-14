# Qualification contract

状態: M1の[checkpoint atlas / integrity検証](checkpoint-atlas.md)が完了。本runtimeのnumerical / performance qualificationは未達。2026-09-13にユーザー報告のomlx v0.7.0.dev2を外部性能baselineとして採用した。同条件の比較runは未実行。[改訂計画](omlx-baseline.md)を適用する。

2026-09-14時点で、修正版native pathはencoder 0..19、decoder 20..39、Engram、final collapse/norm/headを接続し、実token IDからlogitsまでの継続・fork/reset・不正token契約を通過した。layer-0 MoEの独立CPU比較は、expert出力のBF16 cast境界とexpert ID昇順加算を合わせた後、59要素の局所差（最大 `0.0009765625`）まで縮小している。これはM2の構造進捗であり、公式oracle比較・sampling/generation・長文qualificationを含むM2 exit gateは未達のままとする。

## Milestones

| Phase | 成果物とexit gate |
| --- | --- |
| M0 — Repository contract | 5文書でarchitecture / correctness / memory / reference / performance契約と禁止事項を固定。GLMは知見inventoryのみ |
| M1 — Checkpoint atlas | 全tensorの所在・形式・owner・bytesを確定。Engram backingを分離し、512 GB予算を作る。missing / unknownと総数の不一致を解消 |
| M2 — End-to-end reference | C++ / MLX / Metalでtoken→encoder→decoder→logits。Engramを含むtext targetを公式oracleと比較。最小SSD reader、state比較、referenceを保持。omlxの直接load / state / 数値境界を対照し、部分probeの拡張よりfull path接続を優先 |
| M3 — Native execution graph | Python往復のないprefill / decode / replayを形成。固定omlxを第一reference、vLLMを補助として、CED scheduleとpacked KVのexactnessを検証。ここから性能最適化 |
| M4 — 32K→256K | 32K correctnessから64K / 128K / 200K比較点 / 256Kへ進み、長時間session後半のdecode TPTをhard gateとして測定 |
| M5 — Engram storage engine | OS page-cache基準のmmap / pread、working-set telemetryとpressure検証。独自cache / async prefetchは必要性が示された場合のみ。全state exactnessとM4再実行 |
| M6 — DSpark / API / Vision / RELEASE | native DSpark、Rust / Axum + deepseek-recipe API、C++ bridge、tool call、reasoning effort、streaming、image input。最終構成で256K qualificationを再実行してrelease |

M0のreference方針とrecipeの採用候補revisionは[architecture](architecture.md)に固定した。omlx実行環境の依存・設定の固定、補助vLLM / native MLXのrevision選定、recipeのbuild / 接続検証と依存lockは後続phaseのexit条件。未確認referenceを利用済みと表示しない。M4の一度のpassをM5 / M6変更後へ無条件に持ち越さない。

## Context ladderとworkload

K＝1,024 tokens。32K / 64K / 128K / 256Kはpromptと生成済みtokenを合わせた総context長で、generation budgetではない。API admissionはprompt tokens + requested output reserveがqualified上限以内か確認する。公式推奨の1M context / 256K以上のmax_tokensを未検証のserver設定として広告しない。

各段階で同一checkpoint / tokenizer、1 active session、固定tokenized corpusを用いる。coding-agent workloadには複数turn、code、長いtool output、追記prefillとdecodeの反復を含める。fixture hash、turn schedule、seed、prompt長、output長を事前固定する。

各context上限Lについて、少なくとも次の3種類を実行する。

1. L−4,096 tokensまでprefillし、teacher-forced continuation 4,096 tokensを実行してLへ到達する。後半2,048 tokensのlatencyとexactnessを記録する。
2. 同じ開始contextからgreedy生成を行い、実際のtoken / logitsを比較する。EOS等で測定長を満たさない場合は明記し、短いrunを長時間passの代わりにしない。
3. 小さいcontextからtool turnとgenerationを累積してLへ達するsessionを実行する。各段階のstate、追記prefill、first decode、tail TPT、reset後のmemory回収を確認する。

大きなpromptを一度だけloadする試験では累積sessionを代替できない。Engram cold / warm両条件で同じcorpusを使い、異なるlookup localityも含める。SWA / compression境界、candidate容量境界、cancel / resumeは[correctness](correctness.md)の対象とする。

## 測定境界

| Metric | 定義 |
| --- | --- |
| load / warm-up | checkpoint読込・検証・resident化・compileを別々に計測 |
| cold prefill wall | fresh session、Engram cache状態を記録し、GPU完了までを計測。OS page cacheがcoldか不明ならunknownとする |
| warm prefill wall | weights / kernels / Engramのwarming条件を固定したfresh sessionで計測。APCなし |
| TTFT | request受付から最初のoutput token利用可能まで。M6ではAPI経路も測定 |
| first decode latency | prefill→decode切替の最初のstepを独立に記録 |
| decode TPT | 各committed output tokenを利用可能にするまでのwall latency、ms/token。値が小さいほどよい |
| tail decode TPT | 上限付近の固定測定窓におけるmean / p50 / p95 / p99 / maxとraw時系列 |
| full session wall | 追記prefill、decode、replay、Engram stall、sampling、同期を含む総時間。外部tool実行待ちは別計上 |
| memory / I/O | physical footprint、peak allocation、graph / cache容量、swap増加、page faults、Engram hit / miss / bytes / latency |

TPTにCPU dispatch、GPU execution、必要な同期、Engram I/O待ち、sampling、state commitを含める。token/sやkernel GPU timeは補助指標。DSpark有効時はdraft token数で割らず、accepted / committed token数を使い、block emissionによる待ち時間も記録する。

M4ではnative generation boundary、M6ではそれに加えてloopback APIのtoken event境界を計測する。streamingのchunking / bufferingも報告し、全tokenを最後に返す方式でtail latencyを隠さない。

## Performance hard gates

correctnessの失敗は性能評価に優先する。全candidateは同じfull workloadでbaselineと比較する。測定環境にはhardware、OS、compiler、MLX / Metal revision、build flags、checkpoint、power / thermal条件、Engram cache予算を記録する。モデルdownload等の競合I/Oがあるrunはqualificationに使用しない。

- cold条件は最低3回、warmのbaseline / candidate比較は実行順を交互にして最低5組を記録する。除外runには理由を残す。
- promotion対象のfull-path wallまたはtail TPTの改善がrun間変動を超えることを要求し、他の主要指標にregressionがないことも確認する。microbenchmarkだけの改善、平均でtail悪化を隠す変更はrejectする。
- 256K session後半のTPTは必須gate。M3終了時にM4実行前の独立した契約変更で、絶対的なmean / p95 / p99上限、許容drift、memory / I/O上限、統計的判定法を数値で固定する。candidateの結果を見てpassになる値へ調整しない。
- 外部baselineの32K–200K報告値は[omlx baseline](omlx-baseline.md)に固定した。これは256Kのtail分布・絶対上限を与えないため、未測定の値を外挿しない。上記閾値が未設定のままM4やRELEASEをpassにすることは禁止する。
- bounded stateの容量増加、古いgraph保持、unexpected full-model copy、継続的なswap増加、I/O failure、NaN / Inf、state不一致は失敗とする。予算内の意図したcache成長は内訳と上限を示す。

late-session driftは同等のcontext長・cache条件のfresh runとも比較し、contextの伸長に伴う費用とsession経過に伴う劣化を区別する。稀なstallを集約値だけで消さず、raw traceに残す。

## omlxに対する性能昇格

内部optimizationは従来のlocal baseline gateを満たすこと。加えて、専用native runtimeとしての性能優位を主張するには、固定omlxとの同一入力・機能・cache・測定境界の比較でfull-path wallまたはtail TPTの改善がrun間変動を超え、他の主要指標にregressionがないことを必須とする。内部baselineへの改善だけでomlx超えとはしない。32K / 64K / 128K / 200Kの比較結果を保存し、256Kは両runtimeを新規測定する。omlxが256Kで未実行・失敗の場合もnativeの256K gateは維持し、比較不能を速度勝利に数えない。

## Release evidence

各runはruntime commit、全reference identity、build / environment、入力・schedule・seed、enabled features、raw timing、state比較、memory / I/O、判定閾値、結果を保存する。失敗・未実行・unsupportedはpassと別に表示する。

M6ではtext-onlyとimage入力、DSpark on / off、tool call、数値reasoning effort、streaming / non-streaming、cancel / resumeを検証する。protocolは公式encoding fixtureと照合し、OpenCodeから実際にAPIを通す。APIや画像処理を加えた最終構成で256K gateを満たすまでRELEASEと呼ばない。APCを後から追加する際もstate再利用のexactnessとfull-path性能を再qualificationする。

recipe採用に伴い、Rust toolchain / Cargo.lock / recipe revisionとnative bridge ABIをrun identityへ含める。mock serverの起動はAPI接続成功に数えない。実token IDsの増分decode、JSON / SSEの整合性、未対応option、stop sequence、切断・cancel・backpressure・backend failureを検証し、native committed token数とAPI usageを別計数する。requestからSSEまでのTTFT / tail latencyとnativeのみのTPTを併記する。

## 長時間検証の運用

ユーザー指定に従い、数分以上かかる見込みの検証はユーザー実行用スクリプトとして渡す。実行手順、ログ / 結果path、想定負荷、再開方法を準備し、assistantが長時間監視し続ける形にしない。結果を回収するまでは未実行 / 未判定として保持し、その間は独立した実装・短いcorrectness検査を進める。
