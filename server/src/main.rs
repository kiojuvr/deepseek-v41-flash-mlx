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
#[cfg(feature = "recipe-adapter")]
mod recipe_adapter;
use backend::{GenerationOptions, RuntimeBackend, UnconnectedBackend};

const MAX_QUALIFIED_OUTPUT_TOKENS: u32 = 262_144;
#[cfg(feature = "recipe-adapter")]
const MAX_QUALIFIED_CONTEXT_TOKENS: usize = 262_144;

#[derive(Clone)]
struct AppState {
    model: Arc<String>,
    backend: Arc<dyn RuntimeBackend>,
    #[cfg(feature = "recipe-adapter")]
    recipe: Option<Arc<recipe_adapter::RecipeEncoder>>,
}

#[derive(Deserialize)]
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
) -> (StatusCode, Json<Value>) {
    if req.model != *state.model {
        return (
            StatusCode::NOT_FOUND,
            Json(
                json!({"error":{"message":"unknown model","type":"invalid_request_error","code":"model_not_found"}}),
            ),
        );
    }
    if req.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error":{"message":"messages must not be empty","type":"invalid_request_error"}}),
            ),
        );
    }
    if req.temperature.is_some_and(|v| !v.is_finite() || v < 0.0) {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error":{"message":"temperature must be finite and nonnegative","type":"invalid_request_error"}}),
            ),
        );
    }
    if req.max_tokens == Some(0) {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error":{"message":"max_tokens must be greater than zero","type":"invalid_request_error"}}),
            ),
        );
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
        );
    }
    let options = GenerationOptions {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        seed: req.seed,
    };
    #[cfg(feature = "recipe-adapter")]
    if let Some(encoder) = &state.recipe {
        let prompt_tokens = match encoder.encode(&req.messages) {
            Ok(tokens) => tokens.len(),
            Err(error) => return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":error,"type":"invalid_request_error","code":"recipe_encoding_error"}}))),
        };
        let requested = req.max_tokens.unwrap_or(16) as usize;
        if prompt_tokens.saturating_add(requested) > MAX_QUALIFIED_CONTEXT_TOKENS {
            return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":"prompt plus max_tokens exceeds the current 256K context admission limit","type":"invalid_request_error","code":"context_length_exceeded"}})));
        }
    }
    match state.backend.complete(&req.messages, req.stream, options) {
        Ok(value) => (StatusCode::OK, Json(value)),
        Err(backend::BackendError::Unavailable) => (
            StatusCode::NOT_IMPLEMENTED,
            Json(
                json!({"error":{"message":"native runtime backend is not connected","type":"server_error","code":"runtime_unavailable"}}),
            ),
        ),
    }
}

#[tokio::main]
async fn main() {
    let bind = env::var("DSV41_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let model = env::var("DSV41_MODEL").unwrap_or_else(|_| "DeepSeek-V4.1-Flash".into());
    #[cfg(feature = "recipe-adapter")]
    let recipe = env::var("DSV41_TOKENIZER")
        .ok()
        .map(|path| recipe_adapter::RecipeEncoder::from_file(&path).expect("failed to load DSV41_TOKENIZER"))
        .map(Arc::new);
    #[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
    let backend: Arc<dyn RuntimeBackend> = if env::var("DSV41_NATIVE_BRIDGE").as_deref() == Ok("1") {
        recipe.as_ref()
            .and_then(|encoder| native::NativeBridge::connect().map(|bridge| Arc::new(backend::NativeRuntimeBackend::new(bridge, (**encoder).clone())) as Arc<dyn RuntimeBackend>))
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
            #[cfg(feature = "recipe-adapter")]
            recipe,
        });
    let listener =
        tokio::net::TcpListener::bind(bind.parse::<SocketAddr>().expect("invalid DSV41_BIND"))
            .await
            .expect("bind failed");
    axum::serve(listener, app).await.expect("server failed");
}
