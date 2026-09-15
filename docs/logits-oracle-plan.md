# Full backboneの次段階: logits oracle比較

2026-09-14。decoder / backboneのユーザーrun記録を再確認し、ログhash、終了code 0、binary / fixture identityが現物と一致した。構造的full pathとlocal整合性の通過として保持する。次の優先順位は独立oracleとの比較 → 差の切り分け・解消 → sampling / generation。

## Oracleの区別

固定checkpointの`inference/model.py`は`kernel.py`をimportし、FP8 / FP4 GEMM、activation量子化、sparse attention、mHC kernelを使う。`kernel.py`はTileLangをimportし、shared memory / GPU kernelを定義する。`device=cpu`へ変えるだけで公式minimal inferenceのfull logitsが得られるとは扱わない。

比較の種類を明記する。

- **公式kernel実行**: 対応GPU上で固定公式minimal inferenceと公式kernelを動かした値。現環境での実行は未確認。
- **公式式のCPU転記**: PyTorch CPUでkernelの式・cast境界を明示した独立実装。意味論の切り分けに使うが、CUDA reductionやtieのoracleとは呼ばない。既存CPU fixtureはこの分類を維持する。
- **固定oMLX対照**: Apple Silicon上で比較可能な実動reference。公式精度の証拠を補助するが、既知のactivation floor差等をnativeへ無条件に移植しない。

## 最初の比較単位

固定token列 `[0,42,1000,42]`、plain text、teacher forcingで比較する。samplingや生成tokenの分岐を持ち込まない。nativeの各tokenのlogitsと、encoder最終hidden / pre-mix、decoder各layer、final collapse / normを保存できるtrace境界を作る。

CPU full modelをいきなり全展開するより先に、固定oMLXと公式式の小さいfixtureを対照としてtraceを準備する。logitsの最初の差からlayer境界を遡り、最初に違うactivation、routing IDs / score / weight、KV bytes / scalesを同じ入力で再比較する。routing tieは回数だけでなくlayer・token・境界score・候補IDを記録する必要がある。

数値報告にはdtype / shape、bit mismatch数、最大絶対差、非有限値、argmax一致を含める。argmax一致だけでexactness通過とはしない。CPU対照の差、oMLX対照の差、local referenceとoptimized pathの差を別に保存する。

## Main KVの記述訂正

現`metal/attention/kv_quant.metal`はmain KVについてE2M1の8水準を選び、2要素を1 byteへpackする。E4M3はgroup-16 scaleの形式。`src/cache/global_kv.cpp`はpacked bytesとscaleを保存し、`main_rows`はE2M1 × E4M3を復元する。

したがって「E4M3の値をFP4へ置換すること」を次の既知修正として扱う根拠は、現コードにはない。未完了なのは公式packed bytes / scale丸め / 復元の独立oracle比較と、既に記録したscale overflow領域。数値の不一致が特定される前に量子化形式を変更しない。

## 残る作業

native trace exporterと比較harness、固定oMLXの外部hook harnessを実装した（[harness文書](logits-oracle-harness.md)）。固定token列 `[0,42,1000,42]` で最初の比較を実施し、[結果](logits-oracle-results.md)を保存した。最初の分岐は `encoder.layer0.attn_in` で、nativeの`mx::rsqrt`が原因と特定し`1/sqrt`へ修正した。次の分岐はmHC projectionのFP32 reduction順で、公式CPU転記とのbit一致は原理的に難しい。samplingへ進む前に、この差を許容するかの契約判断が必要。長時間になるnative / oMLX / CPU実行は、identity・ログ・結果・再開手順を備えたユーザー実行スクリプトとして準備する。

on-demand expert loadは現referenceの資源節約経路として扱い、当初のresident MoE production契約や性能判定へ自動昇格させない。既知のRMSNorm / mHC / Engram差、routing tieの最小ID方針、index tie停止は引き継ぐ。samplingはこの比較に必要なく、oracle差を解決したことにはならない。

## 引き継ぎ後の優先方針（2026-09-14）

公式式CPU referenceをlogitsまで拡張する方針を維持する。oMLX servingのencoding validationはAPI / protocolの問題として分離し、offline tensor比較はHTTP経由にしない。ユーザー報告のimage marker拒否はこの作業では未再現・未修正。

比較harnessのsub-boundary順序とoracle側非有限値の検出を修正し、logical dtype不一致・element bit mismatch数・最初のbit不一致も記録する。5件の短い回帰検査が通過。既存レポートは旧比較器の履歴として保持する。

最初の独立境界として、`trace_cpu_rope.py`で固定公式model.pyの`precompute_freqs_cis` / `apply_rotary_emb`をAST抽出し、そのままPyTorch CPUで実行した。M1 verificationで公式source hashを検証し、入力・出力・script hashをmanifestに保存する。nativeのRoPE前値を固定入力とする局所比較であり、full CPU pathではない。

60 tokenのlayer 0 Qは7要素のbit不一致、max_abs 0.015625、KVはbit一致。結果は`artifacts/logits-trace/compare-cpu-rope-takeover-20260914.json`。oMLX compiled RoPEの14要素差とは別の比較で、nativeのCUDA一致／不一致を判定しない。CPU full backboneとrouting tieの独立検証は未完了。

次の拡張としてlayer 0 Attention全体のCPU traceを実装した。固定native `attn_in`から10境界を独立計算し、FP8 projectionの32-channel block和、BF16 cast、SWA slot順序、sinkを明示する。実checkpoint実行は未実施。ユーザー実行手順と資源・失敗/再実行手順は[harness文書](logits-oracle-harness.md)に記載した。結果を読んで最初の分岐を切り分けた後、mHC/MoEと後続layerへ広げる。M2の許容差や昇格契約は変更しない。
