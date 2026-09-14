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
