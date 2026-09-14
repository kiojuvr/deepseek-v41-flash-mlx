use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{env, net::SocketAddr, sync::Arc};

#[derive(Clone)]
struct AppState { model: Arc<String> }

#[derive(Deserialize)]
struct ChatRequest { model: String, messages: Vec<Value>, #[serde(default)] stream: bool }

#[derive(Serialize)]
struct Model { id: String, object: &'static str, owned_by: &'static str }

async fn health() -> impl IntoResponse { Json(json!({"status":"ok","runtime":"unconnected","release":false})) }

async fn models(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({"object":"list","data":[Model{id:(*state.model).clone(),object:"model",owned_by:"deepseek"}]}))
}

async fn chat(State(state): State<AppState>, Json(req): Json<ChatRequest>) -> (StatusCode, Json<Value>) {
    if req.model != *state.model {
        return (StatusCode::NOT_FOUND, Json(json!({"error":{"message":"unknown model","type":"invalid_request_error","code":"model_not_found"}})));
    }
    if req.messages.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":"messages must not be empty","type":"invalid_request_error"}})));
    }
    // A stub must never pretend that inference happened. This endpoint is useful
    // for client/protocol probes before the C++ bridge is available.
    let _stream = req.stream;
    (StatusCode::NOT_IMPLEMENTED, Json(json!({"error":{"message":"native runtime backend is not connected","type":"server_error","code":"runtime_unavailable"}})))
}

#[tokio::main]
async fn main() {
    let bind = env::var("DSV41_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let model = env::var("DSV41_MODEL").unwrap_or_else(|_| "DeepSeek-V4.1-Flash".into());
    let app = Router::new().route("/health", get(health)).route("/v1/models", get(models)).route("/v1/chat/completions", post(chat)).with_state(AppState { model: Arc::new(model) });
    let listener = tokio::net::TcpListener::bind(bind.parse::<SocketAddr>().expect("invalid DSV41_BIND")).await.expect("bind failed");
    axum::serve(listener, app).await.expect("server failed");
}
