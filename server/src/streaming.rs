//! SSE transport around the pinned recipe stream processor.
use crate::{
    backend::BackendError,
    recipe_adapter::{RecipeEncoder, RecipePlan},
    request_cancel::CancelOnDrop,
};
use axum::response::{
    IntoResponse, Response, Sse,
    sse::{Event, KeepAlive},
};
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage};
use std::convert::Infallible;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::StreamExt;

pub struct StreamEnd {
    pub reason: String,
    pub committed: usize,
}
#[derive(Default, Clone)]
struct Status {
    error: Option<String>,
    finished: bool,
    committed: usize,
}
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub fn response(
    encoder: RecipeEncoder,
    plan: RecipePlan,
    model: String,
    prompt: usize,
    eos: u32,
    mut tokens: mpsc::Receiver<u32>,
    done: oneshot::Receiver<Result<StreamEnd, BackendError>>,
    cancelled: Arc<AtomicBool>,
    include_usage: bool,
) -> Response {
    let id = format!(
        "chatcmpl-stream-{}-{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    let processor = encoder.processor(&plan, &model, &id);
    let status = Arc::new(Mutex::new(Status::default()));
    let input_status = status.clone();
    let input = async_stream::stream! {
        yield InferenceChunk::Ready { system_fingerprint: None, prompt_usage: PromptUsage { prompt_tokens: prompt, prompt_cache_hit_tokens: 0 } };
        let mut count = 0;
        let mut saw_eos = false;
        while let Some(token_id) = tokens.recv().await {
            count += 1;
            input_status.lock().unwrap().committed = count;
            if saw_eos {
                input_status.lock().unwrap().error = Some("token received after EOS".into());
                return;
            }
            if token_id == eos {
                saw_eos = true;
                yield InferenceChunk::Text { content: String::new(), content_tokens: 1 };
            } else { yield InferenceChunk::Token { token_id }; }
        }
        match done.await {
            Ok(Ok(end)) if end.committed == count && (!saw_eos || end.reason == "stop") => {
                let reason = match end.reason.as_str() {
                    "length" => InferenceFinishReason::Length,
                    "stop" => InferenceFinishReason::Stop,
                    _ => { input_status.lock().unwrap().error = Some("invalid finish reason".into()); return; }
                };
                { let mut status = input_status.lock().unwrap(); status.committed = count; status.finished = true; }
                yield InferenceChunk::Finish { finish_reason: reason };
            }
            Ok(Ok(_)) => input_status.lock().unwrap().error = Some("stream completion accounting mismatch".into()),
            Ok(Err(error)) => input_status.lock().unwrap().error = Some(format!("{error:?}")),
            Err(_) => input_status.lock().unwrap().error = Some("native worker ended without a terminal result".into()),
        }
    };
    // Capture the guard before the body is first polled: even an unpolled body
    // must cancel and close the receiver when the transport discards it.
    let guard = CancelOnDrop(cancelled.clone());
    let output = async_stream::stream! {
        let _guard = guard;
        let mut failure = None;
        {
            let parsed = processor.process(input);
            tokio::pin!(parsed);
            while let Some(chunk) = parsed.next().await {
                let snapshot = status.lock().unwrap().clone();
                if let Some(error) = snapshot.error { failure = Some(error); break; }
                let mut value = match chunk.and_then(|chunk| serde_json::to_value(chunk).map_err(|e| deepseek_recipe::stream::StreamError::Decode { detail: e.to_string() })) {
                    Ok(value) => value,
                    Err(error) => { failure = Some(error.to_string()); break; }
                };
                let usage = value.get("usage").filter(|v| !v.is_null()).cloned();
                let local_stop = !snapshot.finished
                    && value
                        .pointer("/choices/0/finish_reason")
                        .and_then(|v| v.as_str())
                        == Some("stop");
                if include_usage { value["usage"] = serde_json::Value::Null; }
                else { value.as_object_mut().unwrap().remove("usage"); }
                // Recipe owns parsing/deltas. Transport splits its final usage
                // snapshot into a choices=[] chunk requested by the client.
                let mut final_usage = usage.filter(|_| include_usage).map(|mut usage| {
                    usage["completion_tokens"] = snapshot.committed.into();
                    usage["total_tokens"] = (prompt + snapshot.committed).into();
                    let mut chunk = value.clone();
                    chunk["choices"] = serde_json::json!([]);
                    chunk["usage"] = usage;
                    chunk
                });
                match Event::default().json_data(value) {
                    Ok(event) => yield Ok::<Event, Infallible>(event),
                    Err(error) => { failure = Some(error.to_string()); break; }
                }
                if let Some(usage) = final_usage.take() {
                    match Event::default().json_data(usage) {
                        Ok(event) => yield Ok(event),
                        Err(error) => { failure = Some(error.to_string()); break; }
                    }
                }
                if local_stop {
                    // Recipe matched a request-local decoded stop. It has already
                    // suppressed the matched text; stop native generation at the
                    // next cooperative boundary and accept this parser terminal.
                    cancelled.store(true, Ordering::Release);
                    status.lock().unwrap().finished = true;
                }
            }
        } // Drop parser/input before reporting failure, unblocking native sends.
        let snapshot = status.lock().unwrap().clone();
        if let Some(error) = failure.or(snapshot.error).or_else(|| (!snapshot.finished).then(|| "missing native terminal result".into())) {
            cancelled.store(true, Ordering::Release);
            let value = serde_json::json!({"error":{"message":error,"type":"server_error","code":"runtime_error"}});
            yield Ok(Event::default().event("error").data(value.to_string()));
        } else {
            yield Ok(Event::default().data("[DONE]"));
        }
    };
    Sse::new(output)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    const TOOL_OUTPUT: &str = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"location\" string=\"true\">Tokyo</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";

    fn encoder() -> RecipeEncoder {
        let vocab = [
            ("[UNK]", 0),
            ("<｜end▁of▁sentence｜>", 1),
            ("reason", 2),
            ("</think>", 3),
            ("answer", 4),
            (TOOL_OUTPUT, 5),
            ("STOP", 6),
            ("tail", 7),
        ]
        .into_iter()
        .map(|(s, i)| (s.to_owned(), i))
        .collect();
        RecipeEncoder::new(tokenizers::Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        ))
    }

    fn plan(encoder: &RecipeEncoder, thinking: bool) -> RecipePlan {
        encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": if thinking { "enabled" } else { "disabled" }}
            }))
            .unwrap()
    }

    fn tool_plan(encoder: &RecipeEncoder) -> RecipePlan {
        encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
                "thinking": {"type": "disabled"},
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "parameters": {"type": "object"}
                }}],
                "tool_choice": "required"
            }))
            .unwrap()
    }
    fn stop_plan(encoder: &RecipeEncoder) -> RecipePlan {
        encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "disabled"},
                "stop": "answerSTOP"
            }))
            .unwrap()
    }
    #[tokio::test]
    async fn normal_stream_matches_recipe_and_finishes_once() {
        let (tx, rx) = mpsc::channel(8);
        let (done, end) = oneshot::channel();
        for token in [2, 3, 4, 1] {
            tx.send(token).await.unwrap();
        }
        drop(tx);
        done.send(Ok(StreamEnd {
            reason: "stop".into(),
            committed: 4,
        }))
        .ok();
        let encoder = encoder();
        let response = response(
            encoder.clone(),
            plan(&encoder, true),
            "model".into(),
            5,
            1,
            rx,
            end,
            Arc::new(AtomicBool::new(false)),
            true,
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert_eq!(text.matches("[DONE]").count(), 1);
        assert!(text.contains("reasoning_content") && text.contains("answer"));
        assert!(text.contains("\"completion_tokens\":4") && text.contains("\"total_tokens\":9"));
        assert!(!text.contains("end▁of▁sentence") && !text.contains("event: error"));
    }
    #[tokio::test]
    async fn usage_option_controls_separate_final_snapshot() {
        for include_usage in [false, true] {
            let (tx, rx) = mpsc::channel(8);
            let (done, end) = oneshot::channel();
            tx.send(4).await.unwrap();
            drop(tx);
            done.send(Ok(StreamEnd {
                reason: "length".into(),
                committed: 1,
            }))
            .ok();
            let encoder = encoder();
            let response = response(
                encoder.clone(),
                plan(&encoder, false),
                "model".into(),
                5,
                1,
                rx,
                end,
                Arc::new(AtomicBool::new(false)),
                include_usage,
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            let events: Vec<serde_json::Value> = text
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|line| *line != "[DONE]")
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert_eq!(text.matches("[DONE]").count(), 1);
            if include_usage {
                let last = events.last().unwrap();
                assert_eq!(last["choices"], serde_json::json!([]));
                assert_eq!(last["usage"]["completion_tokens"], 1);
                assert_eq!(last["usage"]["total_tokens"], 6);
                for event in &events[..events.len() - 1] {
                    assert_eq!(event.get("usage"), Some(&serde_json::Value::Null));
                }
                assert_eq!(
                    events[events.len() - 2]["choices"][0]["finish_reason"],
                    "length"
                );
            } else {
                assert!(events.iter().all(|event| event.get("usage").is_none()));
                assert_eq!(
                    events.last().unwrap()["choices"][0]["finish_reason"],
                    "length"
                );
            }
        }
    }

    #[tokio::test]
    async fn disabled_thinking_sse_returns_answer_content() {
        let (tx, rx) = mpsc::channel(8);
        let (done, end) = oneshot::channel();
        tx.send(4).await.unwrap();
        tx.send(1).await.unwrap();
        drop(tx);
        done.send(Ok(StreamEnd {
            reason: "stop".into(),
            committed: 2,
        }))
        .ok();
        let encoder = encoder();
        let response = response(
            encoder.clone(),
            plan(&encoder, false),
            "model".into(),
            5,
            1,
            rx,
            end,
            Arc::new(AtomicBool::new(false)),
            true,
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("\"content\":\"answer\""));
        assert!(!text.contains("reasoning_content"));
        assert!(text.contains("\"completion_tokens\":2"));
        assert_eq!(text.matches("[DONE]").count(), 1);
    }

    #[tokio::test]
    async fn local_stop_finishes_sse_and_requests_native_cancellation() {
        let (tx, rx) = mpsc::channel(8);
        let (_done, end) = oneshot::channel();
        tx.send(4).await.unwrap();
        tx.send(6).await.unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let encoder = encoder();
        let response = response(
            encoder.clone(),
            stop_plan(&encoder),
            "model".into(),
            5,
            1,
            rx,
            end,
            cancelled.clone(),
            true,
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(!text.contains("\"content\":\"answer\""));
        assert!(!text.contains("STOP") && !text.contains("event: error"));
        assert!(text.contains("\"finish_reason\":\"stop\""));
        assert!(text.contains("\"completion_tokens\":2"));
        assert_eq!(text.matches("[DONE]").count(), 1);
        assert!(cancelled.load(Ordering::Acquire));
        assert!(tx.send(7).await.is_err());
    }

    #[tokio::test]
    async fn official_parser_streams_tool_call_and_tool_finish() {
        let (tx, rx) = mpsc::channel(8);
        let (done, end) = oneshot::channel();
        tx.send(5).await.unwrap();
        tx.send(1).await.unwrap();
        drop(tx);
        done.send(Ok(StreamEnd {
            reason: "stop".into(),
            committed: 2,
        }))
        .ok();
        let encoder = encoder();
        let response = response(
            encoder.clone(),
            tool_plan(&encoder),
            "model".into(),
            8,
            1,
            rx,
            end,
            Arc::new(AtomicBool::new(false)),
            true,
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("\"name\":\"get_weather\""));
        let events: Vec<serde_json::Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| *line != "[DONE]")
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let arguments = events
            .iter()
            .flat_map(|event| event["choices"].as_array().into_iter().flatten())
            .flat_map(|choice| {
                choice["delta"]["tool_calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
            })
            .filter_map(|call| call["function"]["arguments"].as_str())
            .collect::<String>();
        assert_eq!(arguments, "{\"location\": \"Tokyo\"}");
        assert!(text.contains("\"finish_reason\":\"tool_calls\""));
        assert!(text.contains("\"completion_tokens\":2"));
        assert_eq!(text.matches("[DONE]").count(), 1);
    }

    #[tokio::test]
    async fn native_failure_never_emits_success_finish_or_done() {
        let (tx, rx) = mpsc::channel(8);
        let (done, end) = oneshot::channel();
        tx.send(2).await.unwrap();
        drop(tx);
        done.send(Err(BackendError::Runtime("injected".into())))
            .ok();
        let encoder = encoder();
        let response = response(
            encoder.clone(),
            plan(&encoder, true),
            "model".into(),
            5,
            1,
            rx,
            end,
            Arc::new(AtomicBool::new(false)),
            true,
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("event: error") && text.contains("injected"));
        assert!(!text.contains("[DONE]") && !text.contains("\"finish_reason\":\""));
    }
    #[tokio::test]
    async fn unpolled_body_drop_cancels_and_unblocks_full_channel() {
        let (tx, rx) = mpsc::channel(1);
        let (_done, end) = oneshot::channel();
        tx.send(2).await.unwrap();
        assert!(matches!(
            tx.try_send(3),
            Err(mpsc::error::TrySendError::Full(_))
        ));
        let cancelled = Arc::new(AtomicBool::new(false));
        let encoder = encoder();
        let response = response(
            encoder.clone(),
            plan(&encoder, true),
            "model".into(),
            5,
            1,
            rx,
            end,
            cancelled.clone(),
            true,
        );
        drop(response);
        assert!(cancelled.load(Ordering::Acquire));
        assert!(tx.send(3).await.is_err());
    }
}
