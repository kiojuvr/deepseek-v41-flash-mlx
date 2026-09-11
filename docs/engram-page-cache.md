# Engram: OS page-cache baseline

状態: 2026-09-12、M2の部分referenceを実装。hash → checkpoint row → BF16のcorrectness確認済み。token → logits、Engram全演算、backbone同居時の性能、256K qualificationは未完了。

## 方針と根拠

Engramの202,758,032,400 bytes（188.833 GiB）は公式shardに残し、read-only mmapを初期経路とする。比較用に通常のbuffered preadを保持する。OSがfile-backed pagesを再利用・evictし、runtimeは要求された行だけを小さな所有bufferへ集める。独自の永続row cache、全表prefetch、mlock、F_NOCACHEは導入しない。

調査した参考実装:

- [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp/tree/a334fc01ea60d84a3be2990b2da624d0d46818fd): [mmap実装](https://github.com/spiritbuun/buun-llama-cpp/blob/a334fc01ea60d84a3be2990b2da624d0d46818fd/src/llama-mmap.cpp)はlazy領域を一括prefetchから除外する。MoE / CPU・CUDAの構成が対象で、Engram / Apple Siliconの性能証明ではない。
- [Deltafin](https://github.com/kiojuvr/deltafin/tree/c33be69cbfc5baeb8543ca0ce523e99885daf596): 同revisionのM8 / M9調査では通常のbuffered positional readとOS cacheを使い、匿名expert cacheの重複を避ける。異なる入力の後にanchorを再実行して自然な再利用を測る方法を参考にする。expert slabとEngram rowの粒度は異なる。

いずれからもコードをコピーしていない。GLM runtimeのコードも使っていない。

Engramは同じ正規化済みn-gramから同じ行を引くため、繰り返す語彙・識別子・protocolによる再利用が期待できる。一方、hashは隣接行の局所性を保証しない。小粒度lookupがSSD I/Oを小さくするとも限らない。OS cacheをbaselineとして採用する根拠と、Engramでの優位性の実証を区別する。

## 公式layoutと実装範囲

canonical sourceはM1で検証したrevision `dba1be0a40aa45a94ad051997016db3960a90277` の `inference/engram.py` と `inference/model.py`。使用したconfig / tokenizerを含むSHA-256は[fixture provenance](../artifacts/engram/fixture-provenance.json)に記録する。

| 項目 | 固定snapshotの値 |
| --- | --- |
| Engram layer | 1, 14 |
| 各layerのrow数 | 384,006,168 / 384,016,682 |
| n-gram / heads | 2・3・4 gram、それぞれ8 heads |
| 1 input tokenあたり | 24 rows × 2 layers = 48 rows |
| 1 rowの保存形式 | E4M3 256 bytes + E8M0 scale 8 bytes（32要素 / scale） |
| logical lookup bytes / token | 12,672 bytes |
| 実機OS page size | 16,384 bytes |

weightとscaleは離れたtensor領域にある。各領域が1 pageに収まっても、再利用のない48行は最大約1.5 MiB分のpagesへアクセスし得る。page境界をまたぐ場合はさらに増える。probeは絶対file offsetから境界跨ぎを含めたunique page数を数える。ただしaddressed page bytesは物理I/O量でもresident量でもない。

- `WeightCatalog`: M1 summaryを要求し、使用shardのheader digest・file size・open時のidentityを照合する。全payloadの再hashは行わないため、M1は過去時点の完全性証拠である。実行中にcheckpointを変更しない。stat確認は変更検出の補助であり、同時truncateによるmmapのSIGBUSを防ぐ保証ではない。
- `EngramHashState`: 正規化map・prime・multiplierは公式Pythonでoffline生成し、実行中はC++だけでhashする。batch 1、3 tokenの履歴、maskによる履歴遮断、連続position、resetとstate copyを扱う。最大chunkは32,768 tokens。session / speculative commitへの統合は今後行う。
- `EngramStore`: 最大4,096 rows / gather（packed約1.03 MiB）。mmap / preadの両方が同じowned packed bytesを返す。mmapのviewはmappingとfdの寿命を共有し、catalog破棄後も有効。
- `engram_lookup_mlx`: packed bytesをMLX所有arrayへコピーし、Metalで公式のFP32 multiply → BF16と一致するbit列へ復元する。整数演算でsubnormalも保持する。full backingをGPU bufferにせず、GPU consumerの入力寿命をgraphに持たせる。

Engramのprojection、gate、short convolution、残差への接続はまだない。このMetal kernelはreference接続に必要な実装で、speedupによるpromotion対象ではない。64 GiBのworking-set予算はOS cacheへの計画上の余裕であり、runtimeが設定できるhard capではない。

## 確認済みの証拠

[CPU mmap結果](../artifacts/engram/cpu-verify.json)、[CPU pread結果](../artifacts/engram/cpu-pread-verify.json)、[Metal結果](../artifacts/engram/metal-verify.json):

- 2 traces × 2,048 tokens × 48 = 196,608 row IDsが公式と一致。C++の一括・tokenwise双方を照合し、fixture生成時は公式の複数chunk分割も照合した。
- 全65,536組のE4M3 / E8M0 bit patternについて、C++とMetalのBF16 bitsが公式 `ParallelEngramEmbedding` のCPU実行と一致した。NaNはcanonical bit表現を含めて比較した。
- checkpointからの1,158行（重複を含む選択数）のBF16 bitsが同じ公式classと一致した。mmap / pread双方で検証した。
- fixture不要のCTestでhash / mask / reset・state copy、境界拒否、mapping寿命、file変更検出を検査。runnerはfake probeでprepare-only、再開、入力 / 結果変更拒否、失敗後retry、lockを検査した。ReleaseとASan / UBSanで通過。

traceは合成coding風入力とseed固定random token IDsであり、実agent sessionの測定ではない。8 token / traceの短いsmokeでtelemetry出力まで確認したが、速度の結論には使わない。[実行provenance](../artifacts/engram/run-provenance.json)にbuild / source / resultのidentityを固定する。

## ユーザー実行: 自然なcache再利用の比較

2026-09-12にユーザー実行の6 runを回収・確認済み。[結果と判定](engram-cache-results.md)を参照。追加の実行は現在不要。以下は初回実行と再開の手順で、既存の出力がある場合は `--resume` または新しい `--output` を使う。

repo rootから実行する。現workspaceではbinaryとfixturesを準備済み。

```sh
python3 tools/benchmark/run_engram_cache.py
```

既存結果を残して再開する場合:

```sh
python3 tools/benchmark/run_engram_cache.py --resume
```

初回は3 rounds、計6 processesを直列実行する。各processはcorrectness確認の後、coding → random IDs → 同じcodingを各2,048 tokens処理する。mmap / preadの順序はroundごとに反転する。測定部分は合計36,864 input tokens。所要時間はSSDと既存cache状態に依存する。

読み取り対象はEngramのみ。backbone約286.408 GiBはまだloadしない。mmapでは約188.833 GiBの仮想address spaceを予約するが、全量をRAMへ読み込まない。2 tracesで数GiB程度の異なるfile pagesに触れ得る。OS read-aheadに伴うI/O増加もあり、実際のresident / I/Oは結果を見て判断する。checkpointを書き換えず、全体cache purgeも実行しない。他のmodel実行や大きなI/Oは避ける。

出力先は `artifacts/engram/cache-runs/`:

| ファイル | 内容 |
| --- | --- |
| `plan.json` | binary / fixtures / summary / runnerのhash、実行順 |
| `00-mmap.log` 等 | 各processの標準出力・エラー |
| `00-mmap.json` 等 | tokenごとのlookup時間、percentiles、unique rows / pages、process memory / I/O / faults |
| `00-mmap.done.json` 等 | 成功結果のdigest、command、process全体wall time |
| `comparison.json` | trace checksum照合と各runの集計 |

Ctrl-Cで子processを停止し、完了runを保存する。`--resume`は成功結果のdigestを確認してskipし、未完了runを再実行する。途中失敗では `.log` を確認する。build / fixture等を変更した場合は、新しい `--output` を指定する。強制終了で `runner.lock` が残った場合は、記載PIDと子probeの停止を確認してからlockを取り除く。中断・再実行によるcache warmingは消せないため、その経緯も結果評価に含める。

`--prepare-only`は入力確認とplan作成だけを行う。その後の実行には同じ引数と `--resume` を付ける。Pythonは検証orchestrationのみで、測定対象のhash / gather / dequantizeにPythonは入らない。

計測時間はC++ hash + gather + CPU BF16のlookup wall。checksumとpage集計は時間外。現在のpread経路は各readの前後でstat確認するため、mmapとの差にはsyscall / identity確認の費用も含まれる。結果は実装経路全体の比較であり、OS方式だけの比較ではない。`--mlx`はcorrectness確認を追加するflagで、storage比較をGPU性能測定へ変えるものではない。

cacheはprocess間で引き継がれる。oracle用の実rowも測定前に読み取る。global cold、cache hit rate、device bandwidthは主張しない。`proc_pid_rusage`のdisk I/Oはprocess会計値で、device-level I/Oの完全な代替ではない。`comparison.json`のpassedは検査と実行の成功で、性能優位やM4 / M5合格を意味しない。

今回の結果を踏まえ、mmap baselineを維持してM2のfull pathへ統合する。backbone・workspaceと同居した状態でのOS cacheの残り方、実coding-agent入力、長context後半のstall / TPTは後続qualificationで測る。

## 再生成・build

依存なしのC++部分（テストにPython標準libraryを使用）:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

fixturesはoffline oracle用Python環境で再生成する。今回確認した環境はPython 3.12.11、PyTorch 2.13.0、NumPy 2.5.1、tokenizers 0.22.2、SymPy 1.14.0。checkpoint自身の公式sourceをimportし、model classのみAST抽出して実行する。モデル全体やTileLangをloadしない。大きな生成物はgit対象外、provenanceのみ保持する。

```sh
/Volumes/SDXC-512/deltafin/.venv/bin/python tools/reference/export_engram_fixture.py \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --verification artifacts/checkpoint/verification.json \
  --output artifacts/engram
```

MetalはMLX 0.32.2のC++ packageを指定してbuildする。ここでは既存環境にインストール済みの公式MLXのheaders / dylib / metallibを利用した。GLMコードへの依存やruntimeでのPython起動はない。別配置の場合は `MLX_DIR` を対応するCMake packageへ変更する。

```sh
cmake -S . -B build-mlx -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 -DDSV41_ENABLE_MLX=ON \
  -DMLX_DIR=/Volumes/SDXC-512/glm53-flash-mlx/.venv/lib/python3.13/site-packages/mlx/share/cmake/MLX
cmake --build build-mlx -j 8
build-mlx/dsv41-engram-probe \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --m1-summary artifacts/checkpoint/summary.json --fixtures artifacts/engram \
  --output artifacts/engram/metal-verify.json --verify-only --mlx
```

このprebuilt MLXには旧Xcode framework search pathが含まれ、link時に未存在directoryのwarningが出る。実機でbuild・GPU実行は通過したが、独立したMLX dependency配置とtoolchain固定は今後整理する。sandboxでMetal deviceが見えない場合は、通常の端末で実行する。
