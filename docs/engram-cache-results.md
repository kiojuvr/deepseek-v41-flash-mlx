# Engram cache比較: 2026-09-12

ユーザー実行の6 processes / 18 tracesを回収し、correctnessと結果の整合性を確認した。**read-only mmap + OS page cacheをM2のbaselineとして維持する。** 独自の永続row cacheやprefetchを追加する根拠は今回の結果にはない。SSD miss時のlatency、backbone同居、長contextの性能判定は未実施。

測定実装はcommit `5ebd3d7`。binary、source manifest、fixtures、M1 summary、runnerのhashを照合し、completion markerとraw JSONのdigestを確認した。全36,864入力のraw timingからmean / percentiles / maxを再計算し、`comparison.json`とも一致した。全runで公式oracle検査が成功し、同一traceのchecksumはmmap / preadと繰り返しの全組合せで一致した。

[集計・判定・digest](../artifacts/engram/cache-review-20260912.json)と、[元のJSON / logs一式](../artifacts/engram/cache-run-20260912.tar.gz)を保存した。archiveは元ファイル20個を保持し、展開内容のdigestも確認済み。元の `artifacts/engram/cache-runs/` は変更していない。

## Lookupの結果

各traceは2,048 input tokens。数値は3 processesのmeanの平均と、各processで得たp99の範囲。単位はms / input token。測定範囲はC++ hash + gather + CPU BF16で、モデルdecode TPTではない。

| 条件 | mmap mean | pread mean | mmap p99範囲 | pread p99範囲 |
| --- | ---: | ---: | ---: | ---: |
| coding・各processの初回 | 0.06360 | 0.13734 | 0.18429–0.18888 | 0.17917–0.18046 |
| random token IDs | 0.13520 | 0.16890 | 0.14817–0.15363 | 0.17842–0.18063 |
| coding・random後の再実行 | 0.04927 | 0.13982 | 0.06233–0.06333 | 0.17392–0.17425 |

coding再実行のmmap lookupはpreadの約35.2%の時間で、mmap自身の初回より約22.5%短い。3 processesの再実行meanは0.04919–0.04934 msと近かった。一方、coding初回のp99はmmapの方が大きい。平均だけから全条件で有利とは評価しない。

preadには各read前後のstat確認があり、mmapにはgather前後の確認がある。差にはこのsyscall費用も含まれる。cache policyだけの性能差を分離した実験ではない。

## Cacheとmemoryの読み方

全18 tracesでprocessの`diskio_bytesread`、`pageins`、`major_faults`の増分は0だった。OS cacheが温まった条件と整合するが、device-level I/Oは測っていないため、物理SSD readが完全に0だったという証明ではない。今回はSSD miss latencyや実測cache hit率を報告できない。

mmapでは各processの最初のcodingで約27,970回、randomで約182,000回のminor faultsが発生し、coding再実行では0になった。新しいprocessでも同傾向が再現している。最初のmmap accessにはmappingの確立費用が含まれると考えられ、初回と再実行の差をSSD cold / warm差へ読み替えない。

| trace | logical row lookups | unique rows | unique file pages | addressed page bytes |
| --- | ---: | ---: | ---: | ---: |
| coding | 98,304 | 15,120 | 30,146 | 471.031 MiB |
| random IDs | 98,304 | 98,300 | 185,764 | 2,902.562 MiB |

合成coding入力では行の再利用が明瞭で、random入力はほぼ全行が異なる。coding再実行はrandomの後もminor faultsなしで通った。今回の小さなworking setを保持できたことを支持する結果で、64 GiB予算や実agentの再利用率を検証したものではない。

mmapのrandom後のprocess RSSは約3,395.6–3,395.8 MiB、`phys_footprint`は約148.1–148.3 MiBだった。preadのRSSは約76.7–76.8 MiB、footprintは約53.5–53.7 MiB。これらの会計値は大きく異なり、footprintだけをEngramのRAM消費としたり、RSSへ加算したりしない。preadのprocess RSSが小さいことも、OS cacheにpagesが存在しないことを意味しない。backboneと同居させる際はsystem memory / pressureも計測する。

## 全体wallと次の作業

completion markerに記録されたprobe全体のwall timeはmmapで1.310–1.325秒、preadで1.239–1.242秒。全6 processesの合計は7.678秒だった。ここには起動、header / fixture検査、公式oracle照合、checksum / page集計、出力まで含まれる。どの区間が差を生んだかは現計測では分離できない。lookupの短縮を、probe全体やモデル全体の高速化として扱わない。

このrunでは実際の所要時間は短かった。今後も予想所要時間に応じて運用し、数分以上かかる見込みの検証はユーザー実行にする。今回の結果を得るための再実行は不要。

次はM2のEngram projection / gate / convolution / residualと、token→logitsのlocal reference接続を優先する。mmapとpreadのexactな読み出しを維持し、lookup単独の追加最適化は先行しない。backboneをloadしたfull pathができた段階で、memory pressure、実入力、SSD missを含むstall、長context後半のTPTを測る。本結果によるperformance promotion、M4 / M5完了判定は行わない。
