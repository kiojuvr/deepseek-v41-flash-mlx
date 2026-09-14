# M2 canonical MLX reference review

2026-09-15。M2の着地点を、PyTorch/CUDA公式実装とのbit一致ではなく、公式
checkpointの意味論を保ったcanonical MLX referenceとしてレビューする。

## 完了した範囲

- 実token IDからencoder 0..19、decoder 20..39、Engram、collapse/norm/head、
  logitsまでのfull path。
- chunk/tokenwise、continuation、fork、reset、不正token拒否。
- layer-0 MoEのCPU FP4対照。nativeのBF16 cast境界とexpert ID昇順加算を反映後、
  59要素差（最大 `0.0009765625`）まで縮小。per-route contributionは27要素差。
- greedy / temperature generationによる16 committed token、state position継続。
- API層とruntime層を分離する`RuntimeBackend`とC ABI event contract。

## M2の判定

現native pathをcanonical MLX referenceとして扱う。公式CPU転記、固定oMLX、公式
CUDA実行（実行可能な場合）は差分診断の別証拠であり、M2のbackend exactnessへ
混ぜない。canonical referenceを変更する場合は、入力・state・境界tensor・logits
の再検証を行う。

## M2外に残す項目

- 32Kから256Kまでのteacher-forced / cumulative session qualification。
- API bridgeへのrecipe接続、SSE、cancel/backpressure、usage/finish accounting。
- DSpark、vision、image/text protocol。
- oMLXとの性能比較とRELEASE判定。

Evidence index: `artifacts/m2/reviewed-result-20260915.json`。
