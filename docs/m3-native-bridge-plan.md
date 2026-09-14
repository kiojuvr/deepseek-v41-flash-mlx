# M3 native bridge plan

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

Rust adapterのfeature付きunit testもC++ shellへリンクして実行し、error eventの
kind/code、request ID、callback変換を確認済みである。

2026-09-15にC++ libraryを`build-mlx`から参照したfeature付き`cargo build`も成功した。
このbuildはserver handlerへbridgeを接続しておらず、実token生成の証拠ではない。
