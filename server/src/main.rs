use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{env, net::SocketAddr, sync::Arc};
mod backend;
mod native;
#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
mod owner_worker;
#[cfg(feature = "recipe-adapter")]
mod recipe_adapter;
mod request_cancel;
#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
mod streaming;
use backend::{GenerationOptions, RuntimeBackend, UnconnectedBackend};

const MAX_QUALIFIED_OUTPUT_TOKENS: u32 = 262_144;
#[cfg(feature = "recipe-adapter")]
const MAX_QUALIFIED_CONTEXT_TOKENS: usize = 262_144;

#[derive(Clone)]
struct AppState {
    model: Arc<String>,
    backend: Arc<dyn RuntimeBackend>,
    admission: Arc<tokio::sync::Semaphore>,
    #[cfg(feature = "recipe-adapter")]
    recipe: Option<Arc<recipe_adapter::RecipeEncoder>>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum ThinkingType {
    Enabled,
    Disabled,
}
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Thinking {
    #[serde(rename = "type")]
    kind: ThinkingType,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    thinking: Option<Thinking>,
    #[serde(default)]
    reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(flatten)]
    unsupported: std::collections::BTreeMap<String, Value>,
}

#[derive(Serialize)]
struct Model {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({"status":"ok","runtime":state.backend.name(),"release":false}))
}

async fn models(State(state): State<AppState>) -> impl IntoResponse {
    Json(
        json!({"object":"list","data":[Model{id:(*state.model).clone(),object:"model",owned_by:"deepseek"}]}),
    )
}

async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> axum::response::Response {
    if req.model != *state.model {
        return (
            StatusCode::NOT_FOUND,
            Json(
                json!({"error":{"message":"unknown model","type":"invalid_request_error","code":"model_not_found"}}),
            ),
        ).into_response();
    }
    if req.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error":{"message":"messages must not be empty","type":"invalid_request_error"}}),
            ),
        ).into_response();
    }
    if !req.unsupported.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error":{"message":format!("unsupported request fields: {}", req.unsupported.keys().cloned().collect::<Vec<_>>().join(", ")),"type":"invalid_request_error","code":"unsupported_parameter"}}),
            ),
        ).into_response();
    }
    let raw_request = match serde_json::to_value(&req) {
        Ok(value) => value,
        Err(error) => {
            return backend_error(backend::BackendError::Invalid(error.to_string()))
                .into_response();
        }
    };
    if req.temperature.is_some_and(|v| !v.is_finite() || v < 0.0) {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error":{"message":"temperature must be finite and nonnegative","type":"invalid_request_error"}}),
            ),
        ).into_response();
    }
    if req.max_tokens == Some(0) {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error":{"message":"max_tokens must be greater than zero","type":"invalid_request_error"}}),
            ),
        ).into_response();
    }
    if req
        .max_tokens
        .is_some_and(|v| v > MAX_QUALIFIED_OUTPUT_TOKENS)
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error":{"message":"max_tokens exceeds the current 256K admission limit","type":"invalid_request_error"}}),
            ),
        ).into_response();
    }
    if req.stream_options.is_some() && !req.stream {
        return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":"stream_options requires stream=true","type":"invalid_request_error","code":"invalid_stream_options"}}))).into_response();
    }
    let options = GenerationOptions {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        seed: req.seed,
        include_usage: req.stream_options.as_ref().is_some_and(|o| o.include_usage),
    };
    #[cfg(feature = "recipe-adapter")]
    if let Some(encoder) = &state.recipe {
        let prompt_tokens = match encoder.prepare(&raw_request) {
            Ok(plan) => plan.tokens.len(),
            Err(error) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(
                        json!({"error":{"message":error,"type":"invalid_request_error","code":"recipe_encoding_error"}}),
                    ),
                ).into_response();
            }
        };
        let requested = req.max_tokens.unwrap_or(16) as usize;
        if prompt_tokens.saturating_add(requested) > MAX_QUALIFIED_CONTEXT_TOKENS {
            return (
                StatusCode::BAD_REQUEST,
                Json(
                    json!({"error":{"message":"prompt plus max_tokens exceeds the current 256K context admission limit","type":"invalid_request_error","code":"context_length_exceeded"}}),
                ),
            ).into_response();
        }
    }
    let permit = match state.admission.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(
                    json!({"error":{"message":"native runtime is busy","type":"server_error","code":"runtime_busy"}}),
                ),
            ).into_response();
        }
    };
    if req.stream && state.backend.name() == "native-model" {
        return match state.backend.stream(raw_request, options, permit) {
            Ok(response) => response,
            Err(error) => backend_error(error).into_response(),
        };
    }
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _cancel_on_drop = request_cancel::CancelOnDrop(cancel.clone());
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        state
            .backend
            .complete(&raw_request, req.stream, options, cancel)
    })
    .await
    .unwrap_or_else(|e| {
        Err(backend::BackendError::Runtime(format!(
            "native worker failed: {e}"
        )))
    });
    match result {
        Ok(value) => (StatusCode::OK, Json(value)),
        Err(error) => backend_error(error),
    }
    .into_response()
}
fn backend_error(error: backend::BackendError) -> (StatusCode, Json<Value>) {
    match error {
        backend::BackendError::UnsupportedStream => (
            StatusCode::NOT_IMPLEMENTED,
            Json(
                json!({"error":{"message":"native SSE is not connected","type":"server_error","code":"streaming_unavailable"}}),
            ),
        ),
        backend::BackendError::Busy => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(
                json!({"error":{"message":"native runtime is busy","type":"server_error","code":"runtime_busy"}}),
            ),
        ),
        backend::BackendError::Invalid(message) => (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error":{"message":message,"type":"invalid_request_error","code":"native_invalid_request"}}),
            ),
        ),
        backend::BackendError::Runtime(message) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":{"message":message,"type":"server_error","code":"runtime_error"}})),
        ),
        backend::BackendError::Unavailable => (
            StatusCode::NOT_IMPLEMENTED,
            Json(
                json!({"error":{"message":"native runtime backend is not connected","type":"server_error","code":"runtime_unavailable"}}),
            ),
        ),
    }
}

#[tokio::main]
async fn main() {
    #[cfg(not(feature = "native-model"))]
    assert!(
        env::var("DSV41_NATIVE_MODEL").as_deref() != Ok("1"),
        "DSV41_NATIVE_MODEL requires the native-model feature"
    );
    let bind = env::var("DSV41_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let model = env::var("DSV41_MODEL").unwrap_or_else(|_| "DeepSeek-V4.1-Flash".into());
    #[cfg(feature = "recipe-adapter")]
    let recipe = env::var("DSV41_TOKENIZER")
        .ok()
        .map(|path| {
            recipe_adapter::RecipeEncoder::from_file(&path).expect("failed to load DSV41_TOKENIZER")
        })
        .map(Arc::new);
    #[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
    let backend: Arc<dyn RuntimeBackend> = if env::var("DSV41_NATIVE_MODEL").as_deref() == Ok("1") {
        #[cfg(feature = "native-model")]
        {
            let encoder = recipe
                .as_ref()
                .expect("native model requires DSV41_TOKENIZER");
            let checkpoint =
                env::var("DSV41_CHECKPOINT").expect("native model requires DSV41_CHECKPOINT");
            let summary = env::var("DSV41_SUMMARY")
                .unwrap_or_else(|_| "artifacts/checkpoint/summary.json".into());
            let metadata = env::var("DSV41_METADATA")
                .unwrap_or_else(|_| "artifacts/engram/metadata.json".into());
            let provenance = env::var("DSV41_PROVENANCE")
                .unwrap_or_else(|_| "artifacts/engram/fixture-provenance.json".into());
            let eos = encoder
                .eos_token()
                .expect("native model requires EOS token");
            let config: Value = serde_json::from_slice(
                &std::fs::read(std::path::Path::new(&checkpoint).join("config.json"))
                    .expect("checkpoint config missing"),
            )
            .expect("invalid checkpoint config");
            assert_eq!(
                config["eos_token_id"].as_u64(),
                Some(eos as u64),
                "checkpoint and tokenizer EOS differ"
            );
            eprintln!("Loading native model bridge");
            Arc::new(
                backend::NativeRuntimeBackend::new(
                    move || {
                        native::NativeBridge::load(
                            &checkpoint,
                            &summary,
                            &metadata,
                            &provenance,
                            &[eos],
                        )
                    },
                    (**encoder).clone(),
                    model.clone(),
                    true,
                )
                .expect("native model initialization failed"),
            )
        }
        #[cfg(not(feature = "native-model"))]
        panic!("DSV41_NATIVE_MODEL requires the native-model feature");
    } else if env::var("DSV41_NATIVE_BRIDGE").as_deref() == Ok("1") {
        recipe
            .as_ref()
            .map(|encoder| {
                Arc::new(
                    backend::NativeRuntimeBackend::new(
                        || {
                            native::NativeBridge::connect()
                                .ok_or_else(|| "bridge shell creation failed".into())
                        },
                        (**encoder).clone(),
                        model.clone(),
                        false,
                    )
                    .expect("bridge owner initialization failed"),
                ) as Arc<dyn RuntimeBackend>
            })
            .unwrap_or_else(|| Arc::new(UnconnectedBackend))
    } else {
        Arc::new(UnconnectedBackend)
    };
    #[cfg(not(all(feature = "native-bridge", feature = "recipe-adapter")))]
    let backend: Arc<dyn RuntimeBackend> = Arc::new(UnconnectedBackend);
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(AppState {
            model: Arc::new(model),
            backend,
            admission: Arc::new(tokio::sync::Semaphore::new(1)),
            #[cfg(feature = "recipe-adapter")]
            recipe,
        });
    let listener =
        tokio::net::TcpListener::bind(bind.parse::<SocketAddr>().expect("invalid DSV41_BIND"))
            .await
            .expect("bind failed");
    axum::serve(listener, app).await.expect("server failed");
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Failing;
    impl RuntimeBackend for Failing {
        fn name(&self) -> &'static str {
            "test"
        }
        fn complete(
            &self,
            _: &Value,
            _: bool,
            _: GenerationOptions,
            _: Arc<std::sync::atomic::AtomicBool>,
        ) -> Result<Value, backend::BackendError> {
            Err(backend::BackendError::Runtime(
                "injected model failure".into(),
            ))
        }
    }
    fn state() -> AppState {
        AppState {
            model: Arc::new("test".into()),
            backend: Arc::new(Failing),
            admission: Arc::new(tokio::sync::Semaphore::new(1)),
            #[cfg(feature = "recipe-adapter")]
            recipe: None,
        }
    }
    fn request() -> ChatRequest {
        ChatRequest {
            model: "test".into(),
            messages: vec![json!({"role":"user","content":"Hi"})],
            stream: false,
            max_tokens: Some(4),
            temperature: Some(0.0),
            seed: Some(0),
            stop: None,
            thinking: None,
            reasoning_effort: None,
            stream_options: None,
            tools: None,
            tool_choice: None,
            unsupported: Default::default(),
        }
    }
    #[tokio::test]
    async fn native_error_remains_http_error_and_releases_admission() {
        let state = state();
        let response = chat(State(state.clone()), Json(request())).await;
        let status = response.status();
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["code"], "runtime_error");
        assert_eq!(state.admission.available_permits(), 1);
    }
    #[tokio::test]
    async fn busy_request_is_rejected_before_worker_launch() {
        let state = state();
        let _permit = state.admission.clone().acquire_owned().await.unwrap();
        let response = chat(State(state), Json(request())).await;
        let status = response.status();
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["code"], "runtime_busy");
    }
    #[tokio::test]
    async fn unsupported_parameters_are_not_silently_ignored() {
        let mut req = request();
        req.unsupported.insert("stop".into(), json!("END"));
        let response = chat(State(state()), Json(req)).await;
        let status = response.status();
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "unsupported_parameter");
    }
    #[test]
    fn thinking_setting_is_typed_and_rejects_unknown_values() {
        let base = json!({"model":"test","messages":[{"role":"user","content":"Hi"}],"thinking":{"type":"disabled"}});
        let req: ChatRequest = serde_json::from_value(base.clone()).unwrap();
        assert!(matches!(req.thinking.unwrap().kind, ThinkingType::Disabled));
        for value in [
            json!({"type":"automatic"}),
            json!({"type":"enabled","budget":4}),
            json!(false),
        ] {
            let mut request = base.clone();
            request["thinking"] = value;
            assert!(serde_json::from_value::<ChatRequest>(request).is_err());
        }
    }
    #[tokio::test]
    async fn stream_options_require_streaming() {
        let mut req = request();
        req.stream_options = Some(StreamOptions {
            include_usage: true,
        });
        let response = chat(State(state()), Json(req)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "invalid_stream_options");
    }
    #[test]
    fn stream_options_reject_unknown_fields_and_wrong_types() {
        for option in [json!({"include_usage":"true"}), json!({"unknown":true})] {
            let value = json!({"model":"test","messages":[],"stream":true,"stream_options":option});
            assert!(serde_json::from_value::<ChatRequest>(value).is_err());
        }
        let value = json!({"model":"test","messages":[],"stream":true,"stream_options":{}});
        let req: ChatRequest = serde_json::from_value(value).unwrap();
        assert!(!req.stream_options.unwrap().include_usage);
    }
}
