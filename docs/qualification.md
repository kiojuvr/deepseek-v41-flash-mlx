# Qualification contract

状態: M0。qualification結果、測定値、performance baselineはまだない。

## Milestones

| Phase | 成果物とexit gate |
| --- | --- |
| M0 — Repository contract | 5文書でarchitecture / correctness / memory / reference / performance契約と禁止事項を固定。GLMは知見inventoryのみ |
| M1 — Checkpoint atlas | 全tensorの所在・形式・owner・bytesを確定。Engram backingを分離し、512 GB予算を作る。missing / unknownと総数の不一致を解消 |
| M2 — End-to-end reference | C++ / MLX / Metalでtoken→encoder→decoder→logits。Engramを含むtext targetを公式oracleと比較。最小SSD reader、state比較、referenceを保持 |
| M3 — Native execution graph | Python往復のないprefill / decode / replayを形成。vLLMの対応referenceを固定し、CED scheduleとpacked KVのexactnessを検証。ここから性能最適化 |
| M4 — 32K→256K | 32K correctnessから64K / 128K / 256Kへ進み、長時間session後半のdecode TPTをhard gateとして測定 |
| M5 — Engram storage engine | bounded page cache、mmap、async prefetch、working-set telemetry。cold / warm full-pathの改善と全state exactness。M4を再実行 |
| M6 — DSpark / API / Vision / RELEASE | native DSpark、OpenAI互換API、tool call、reasoning effort、streaming、image input。最終構成で256K qualificationを再実行してrelease |

M0のreference方針は固定済みだが、vLLM / MLX / recipeの依存revision選定は後続phaseのexit条件。未確認referenceを利用済みと表示しない。M4の一度のpassをM5 / M6変更後へ無条件に持ち越さない。

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
- M0では要求TPTの数値もbaselineもないため絶対値を捏造しない。上記閾値が未設定のままM4やRELEASEをpassにすることは禁止する。
- bounded stateの容量増加、古いgraph保持、unexpected full-model copy、継続的なswap増加、I/O failure、NaN / Inf、state不一致は失敗とする。予算内の意図したcache成長は内訳と上限を示す。

late-session driftは同等のcontext長・cache条件のfresh runとも比較し、contextの伸長に伴う費用とsession経過に伴う劣化を区別する。稀なstallを集約値だけで消さず、raw traceに残す。

## Release evidence

各runはruntime commit、全reference identity、build / environment、入力・schedule・seed、enabled features、raw timing、state比較、memory / I/O、判定閾値、結果を保存する。失敗・未実行・unsupportedはpassと別に表示する。

M6ではtext-onlyとimage入力、DSpark on / off、tool call、数値reasoning effort、streaming / non-streaming、cancel / resumeを検証する。protocolは公式encoding fixtureと照合し、OpenCodeから実際にAPIを通す。APIや画像処理を加えた最終構成で256K gateを満たすまでRELEASEと呼ばない。APCを後から追加する際もstate再利用のexactnessとfull-path性能を再qualificationする。
