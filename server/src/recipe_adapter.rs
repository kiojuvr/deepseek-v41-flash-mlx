//! Protocol to pinned DeepSeek recipe boundary.
//!
//! Recipe owns prompt encoding and output parsing; native owns token execution.

use deepseek_recipe_core::{conversation::Conversation, messages::InputMessage};
use deepseek_recipe_encoding::{PromptEncoding, v4::dsv41::DeepseekV41Encoding};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug)]
pub struct RecipePlan {
    pub tokens: Vec<u32>,
    parsing: deepseek_recipe::stream::state_machine::ParsingOptions,
    thinking: bool,
    has_stop_sequences: bool,
}

pub struct IncrementalResponse {
    pub value: Value,
    pub locally_stopped: bool,
    pub consumed: usize,
}

pub struct IncrementalSource {
    pub prompt_tokens: usize,
    pub eos: u32,
    pub tokens: tokio::sync::mpsc::Receiver<u32>,
    pub done: tokio::sync::oneshot::Receiver<Result<(String, usize), String>>,
    pub cancelled: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterError {
    MissingRole,
    InvalidRole(String),
    InvalidContent,
    ImageContentRequiresVision,
}

fn content(value: &Value) -> Result<String, AdapterError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                let Some(kind) = part.get("type").and_then(Value::as_str) else {
                    return Err(AdapterError::InvalidContent);
                };
                match kind {
                    "text" => out.push_str(
                        part.get("text")
                            .and_then(Value::as_str)
                            .ok_or(AdapterError::InvalidContent)?,
                    ),
                    "image_url" | "input_image" => {
                        return Err(AdapterError::ImageContentRequiresVision);
                    }
                    _ => return Err(AdapterError::InvalidContent),
                }
            }
            Ok(out)
        }
        _ => Err(AdapterError::InvalidContent),
    }
}

/// Convert OpenAI-style messages into the pinned recipe conversation model.
pub fn conversation(messages: &[Value]) -> Result<Conversation, AdapterError> {
    let mut out = Conversation::default();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or(AdapterError::MissingRole)?;
        let text = content(message.get("content").ok_or(AdapterError::InvalidContent)?)?;
        out.messages.push(match role {
            "system" => InputMessage::System { content: text },
            "user" => InputMessage::User {
                content: text,
                image_sources: vec![],
            },
            "assistant" => InputMessage::Assistant {
                content: text,
                reasoning_content: None,
                tool_calls: None,
            },
            "tool" => InputMessage::Tool {
                content: text,
                image_sources: vec![],
                tool_call_id: message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            },
            other => return Err(AdapterError::InvalidRole(other.to_owned())),
        });
    }
    Ok(out)
}

/// Render and tokenize a conversation with a caller-owned tokenizer file.
/// Loading is explicit so the HTTP layer can keep tokenizer ownership in its
/// application state and the native bridge only receives borrowed token IDs.
#[allow(dead_code)]
pub fn encode_file(messages: &[Value], tokenizer_path: &str) -> Result<Vec<u32>, String> {
    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;
    RecipeEncoder::new(tokenizer).encode(messages)
}

/// Reusable, immutable recipe encoder intended for application state.
#[derive(Clone)]
pub struct RecipeEncoder {
    tokenizer: Arc<tokenizers::Tokenizer>,
}

impl RecipeEncoder {
    pub fn has_stop_sequences(plan: &RecipePlan) -> bool {
        plan.has_stop_sequences
    }
    pub fn prepare(&self, raw: &Value) -> Result<RecipePlan, String> {
        use deepseek_recipe::openai::chat_completion::request::ChatCompletionRequest;
        use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
        let request: ChatCompletionRequest =
            serde_json::from_value(raw.clone()).map_err(|error| format!("protocol: {error}"))?;
        let converted = request
            .convert(ConversionOptions::default())
            .map_err(|error| format!("protocol: {error}"))?;
        for message in &converted.conversation.messages {
            let images = match message {
                InputMessage::User { image_sources, .. }
                | InputMessage::Tool { image_sources, .. } => image_sources.len(),
                _ => 0,
            };
            if images != 0 {
                return Err("protocol: image content requires vision runtime".into());
            }
        }
        if converted
            .conversation
            .tools
            .iter()
            .any(|tool| tool.strict == Some(true))
        {
            return Err("protocol: strict tool schemas are not supported".into());
        }
        let thinking = converted.conversation.thinking_mode;
        let has_stop_sequences = converted
            .parsing_options
            .stop_sequences
            .iter()
            .any(|sequence| !sequence.is_empty());
        let tokens = DeepseekV41Encoding::new()
            .with_tokenizer(self.tokenizer.clone())
            .encode(&converted.conversation)
            .map_err(|error| error.to_string())?;
        Ok(RecipePlan {
            tokens,
            parsing: converted.parsing_options,
            thinking,
            has_stop_sequences,
        })
    }

    pub fn processor(
        &self,
        plan: &RecipePlan,
        model: &str,
        id: &str,
    ) -> deepseek_recipe::stream::StreamProcessor<
        deepseek_recipe::openai::chat_completion::response::ChatCompletionChunkGenerator,
    > {
        use deepseek_recipe::openai::chat_completion::response::ChatCompletionChunkGenerator;
        use deepseek_recipe::stream::StreamProcessor;
        let generator =
            ChatCompletionChunkGenerator::new(id.into(), model.into(), true, plan.thinking);
        StreamProcessor::new(generator, plan.parsing.clone()).with_tokenizer(self.tokenizer.clone())
    }
    pub fn eos_token(&self) -> Result<u32, String> {
        self.tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .ok_or_else(|| "tokenizer has no EOS token".into())
    }

    pub async fn response(
        &self,
        plan: RecipePlan,
        model: &str,
        id: &str,
        prompt_tokens: usize,
        tokens: &[u32],
        reason: &str,
    ) -> Result<Value, String> {
        use deepseek_recipe::openai::chat_completion::response::ChatCompletionResponse;
        use deepseek_recipe::response::ProtocolResponse;
        use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage};
        use deepseek_recipe::util::append_delta::AppendDelta;
        use tokio_stream::StreamExt;
        let finish_reason = match reason {
            "length" => InferenceFinishReason::Length,
            "stop" => InferenceFinishReason::Stop,
            _ => return Err("invalid generation finish reason".into()),
        };
        let eos = self.eos_token()?;
        let mut chunks = vec![InferenceChunk::Ready {
            system_fingerprint: None,
            prompt_usage: PromptUsage {
                prompt_tokens,
                prompt_cache_hit_tokens: 0,
            },
        }];
        for (index, &token_id) in tokens.iter().enumerate() {
            if token_id == eos {
                if index + 1 != tokens.len() || reason != "stop" {
                    return Err("unexpected EOS in committed output".into());
                }
                // EOS is committed and counted, but not displayed as assistant text.
                chunks.push(InferenceChunk::Text {
                    content: String::new(),
                    content_tokens: 1,
                });
            } else {
                chunks.push(InferenceChunk::Token { token_id });
            }
        }
        let processor = self.processor(&plan, model, id);
        chunks.push(InferenceChunk::Finish { finish_reason });
        let output = processor.process(tokio_stream::iter(chunks));
        tokio::pin!(output);
        let mut response = ChatCompletionResponse::new(
            id.into(),
            model.into(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_secs(),
            prompt_tokens,
            0,
        );
        while let Some(chunk) = output.next().await {
            response.append(chunk.map_err(|e| e.to_string())?);
        }
        let mut response = serde_json::to_value(response).map_err(|e| e.to_string())?;
        // Without local stops every committed native ID belongs to the request,
        // including EOS and an incomplete trailing UTF-8 sequence. With a local
        // stop, recipe usage ends at the decoded chunk that matched it; native
        // IDs speculatively produced after that point are not client-visible.
        if !plan.has_stop_sequences {
            response["usage"]["completion_tokens"] = tokens.len().into();
            response["usage"]["total_tokens"] = (prompt_tokens + tokens.len()).into();
        }
        Ok(response)
    }

    pub async fn incremental_response(
        &self,
        plan: RecipePlan,
        model: &str,
        id: &str,
        source: IncrementalSource,
    ) -> Result<IncrementalResponse, String> {
        use deepseek_recipe::openai::chat_completion::response::{
            ChatCompletionFinishReason, ChatCompletionResponse,
        };
        use deepseek_recipe::response::ProtocolResponse;
        use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage};
        use deepseek_recipe::util::append_delta::AppendDelta;
        use std::sync::atomic::Ordering;
        use tokio_stream::StreamExt;

        let IncrementalSource {
            prompt_tokens,
            eos,
            mut tokens,
            done,
            cancelled,
        } = source;

        #[derive(Default)]
        struct InputStatus {
            error: Option<String>,
            finished: bool,
            consumed: usize,
        }
        let status = Arc::new(std::sync::Mutex::new(InputStatus::default()));
        let input_status = status.clone();
        let input = async_stream::stream! {
            yield InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens, prompt_cache_hit_tokens: 0 },
            };
            let mut saw_eos = false;
            while let Some(token_id) = tokens.recv().await {
                {
                    let mut status = input_status.lock().unwrap();
                    status.consumed += 1;
                    if saw_eos {
                        status.error = Some("token received after EOS".into());
                        return;
                    }
                }
                if token_id == eos {
                    saw_eos = true;
                    yield InferenceChunk::Text { content: String::new(), content_tokens: 1 };
                } else {
                    yield InferenceChunk::Token { token_id };
                }
            }
            match done.await {
                Ok(Ok((reason, committed))) if committed == input_status.lock().unwrap().consumed
                    && (!saw_eos || reason == "stop") => {
                    let reason = match reason.as_str() {
                        "length" => InferenceFinishReason::Length,
                        "stop" => InferenceFinishReason::Stop,
                        _ => {
                            input_status.lock().unwrap().error = Some("invalid finish reason".into());
                            return;
                        }
                    };
                    input_status.lock().unwrap().finished = true;
                    yield InferenceChunk::Finish { finish_reason: reason };
                }
                Ok(Ok(_)) => input_status.lock().unwrap().error = Some("completion accounting mismatch".into()),
                Ok(Err(error)) => input_status.lock().unwrap().error = Some(error),
                Err(_) => input_status.lock().unwrap().error = Some("native worker ended without a terminal result".into()),
            }
        };
        let processor = self.processor(&plan, model, id);
        let output = processor.process(input);
        tokio::pin!(output);
        let mut response = ChatCompletionResponse::new(
            id.into(),
            model.into(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_secs(),
            prompt_tokens,
            0,
        );
        let mut locally_stopped = false;
        while let Some(chunk) = output.next().await {
            let chunk = chunk.map_err(|error| error.to_string())?;
            let snapshot = status.lock().unwrap();
            if let Some(error) = &snapshot.error {
                return Err(error.clone());
            }
            let local_stop = !snapshot.finished
                && chunk.choices.iter().any(|choice| {
                    matches!(choice.finish_reason, Some(ChatCompletionFinishReason::Stop))
                });
            drop(snapshot);
            response.append(chunk);
            if local_stop {
                locally_stopped = true;
                cancelled.store(true, Ordering::Release);
            }
        }
        let snapshot = status.lock().unwrap();
        if let Some(error) = &snapshot.error {
            return Err(error.clone());
        }
        if !locally_stopped && !snapshot.finished {
            return Err("missing native terminal result".into());
        }
        let consumed = snapshot.consumed;
        drop(snapshot);
        let mut value = serde_json::to_value(response).map_err(|error| error.to_string())?;
        if !locally_stopped {
            value["usage"]["completion_tokens"] = consumed.into();
            value["usage"]["total_tokens"] = (prompt_tokens + consumed).into();
        }
        Ok(IncrementalResponse {
            value,
            locally_stopped,
            consumed,
        })
    }
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
        }
    }

    #[allow(dead_code)]
    pub fn from_file(path: &str) -> Result<Self, String> {
        tokenizers::Tokenizer::from_file(path)
            .map(Self::new)
            .map_err(|e| format!("tokenizer: {e}"))
    }

    pub fn encode(&self, messages: &[Value]) -> Result<Vec<u32>, String> {
        let c = conversation(messages).map_err(|e| format!("protocol: {e:?}"))?;
        DeepseekV41Encoding::new()
            .with_tokenizer(self.tokenizer.clone())
            .encode(&c)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_recipe_encoding::{PromptEncoding, v4::dsv41::DeepseekV41Encoding};

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
            ("東", 8),
            ("京", 9),
        ]
        .into_iter()
        .map(|(word, id)| (word.to_owned(), id))
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

    #[tokio::test]
    async fn recipe_separates_reasoning_and_counts_committed_eos() {
        let encoder = encoder();
        let value = encoder
            .response(
                plan(&encoder, true),
                "model",
                "id",
                5,
                &[2, 3, 4, 1],
                "stop",
            )
            .await
            .unwrap();
        assert_eq!(
            value["choices"][0]["message"]["reasoning_content"],
            "reason"
        );
        assert_eq!(value["choices"][0]["message"]["content"], "answer");
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert_eq!(value["usage"]["completion_tokens"], 4);
        assert_eq!(value["usage"]["total_tokens"], 9);
        assert!(!value.to_string().contains("end▁of▁sentence"));
    }
    #[tokio::test]
    async fn thinking_disabled_routes_answer_to_content() {
        let encoder = encoder();
        let value = encoder
            .response(plan(&encoder, false), "model", "id", 5, &[4, 1], "stop")
            .await
            .unwrap();
        assert_eq!(value["choices"][0]["message"]["content"], "answer");
        assert!(
            value["choices"][0]["message"]
                .get("reasoning_content")
                .is_none()
        );
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn request_stop_suppresses_match_and_excludes_speculative_tail_usage() {
        let encoder = encoder();
        let plan = encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "disabled"},
                "stop": "STOP"
            }))
            .unwrap();
        let value = encoder
            .response(plan, "model", "id", 5, &[4, 6, 7], "length")
            .await
            .unwrap();
        assert_eq!(value["choices"][0]["message"]["content"], "answer");
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert_eq!(value["usage"]["completion_tokens"], 2);
        assert_eq!(value["usage"]["total_tokens"], 7);
        let content = value["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(!content.contains("STOP") && !content.contains("tail"));
    }

    #[tokio::test]
    async fn incremental_nonstream_stop_cancels_at_consumed_match() {
        let encoder = encoder();
        let plan = encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "disabled"},
                "stop": "STOP"
            }))
            .unwrap();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let (_done, finished) = tokio::sync::oneshot::channel();
        tx.send(4).await.unwrap();
        tx.send(6).await.unwrap();
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let result = encoder
            .incremental_response(
                plan,
                "model",
                "id",
                IncrementalSource {
                    prompt_tokens: 5,
                    eos: 1,
                    tokens: rx,
                    done: finished,
                    cancelled: cancelled.clone(),
                },
            )
            .await
            .unwrap();
        assert!(result.locally_stopped && result.consumed == 2);
        assert!(cancelled.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(result.value["choices"][0]["message"]["content"], "answer");
        assert_eq!(result.value["choices"][0]["finish_reason"], "stop");
        assert_eq!(result.value["usage"]["completion_tokens"], 2);
        assert!(tx.send(7).await.is_err());
    }

    #[tokio::test]
    async fn incremental_nonstream_without_match_uses_native_terminal() {
        let encoder = encoder();
        let plan = encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "disabled"},
                "stop": "STOP"
            }))
            .unwrap();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let (done, finished) = tokio::sync::oneshot::channel();
        tx.send(4).await.unwrap();
        drop(tx);
        done.send(Ok(("length".into(), 1))).unwrap();
        let result = encoder
            .incremental_response(
                plan,
                "model",
                "id",
                IncrementalSource {
                    prompt_tokens: 5,
                    eos: 1,
                    tokens: rx,
                    done: finished,
                    cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                },
            )
            .await
            .unwrap();
        assert!(!result.locally_stopped && result.consumed == 1);
        assert_eq!(result.value["choices"][0]["finish_reason"], "length");
        assert_eq!(result.value["usage"]["completion_tokens"], 1);
    }

    #[test]
    fn request_stop_shape_is_validated_by_pinned_recipe() {
        let encoder = encoder();
        for stop in [
            serde_json::json!(7),
            serde_json::json!([
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q"
            ]),
        ] {
            let request = serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stop": stop
            });
            assert!(encoder.prepare(&request).is_err());
        }
    }

    #[tokio::test]
    async fn stop_arrays_cover_unicode_and_overlapping_prefixes() {
        let encoder = encoder();
        let unicode = encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "disabled"},
                "stop": ["unused", "東京"]
            }))
            .unwrap();
        let value = encoder
            .response(unicode, "model", "unicode", 5, &[8, 9, 7], "length")
            .await
            .unwrap();
        assert_eq!(value["choices"][0]["message"]["content"], "");
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert_eq!(value["usage"]["completion_tokens"], 2);

        let overlap = encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "disabled"},
                "stop": ["answer", "answerSTOP"]
            }))
            .unwrap();
        let value = encoder
            .response(overlap, "model", "overlap", 5, &[4, 6], "length")
            .await
            .unwrap();
        assert_eq!(value["choices"][0]["message"]["content"], "");
        assert_eq!(value["usage"]["completion_tokens"], 1);
    }

    #[test]
    fn explicit_thinking_matches_pinned_recipe_request_conversion() {
        use deepseek_recipe::openai::chat_completion::request::ChatCompletionRequest;
        use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
        for enabled in [false, true] {
            let messages = vec![serde_json::json!({"role":"user","content":"Hi"})];
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model":"model", "messages":messages,
                "thinking":{"type":if enabled {"enabled"} else {"disabled"}}
            }))
            .unwrap();
            let converted = request.convert(ConversionOptions::default()).unwrap();
            let mut local = conversation(&messages).unwrap();
            local.thinking_mode = enabled;
            let encoding = DeepseekV41Encoding::new();
            assert_eq!(
                encoding.render_conversation(&local).prompt,
                encoding.render_conversation(&converted.conversation).prompt
            );
            assert_eq!(
                converted.parsing_options.reasoning_initial_stage.is_some(),
                enabled
            );
        }
    }

    #[test]
    fn reasoning_effort_prompts_match_pinned_recipe_conversion() {
        use deepseek_recipe::openai::chat_completion::request::ChatCompletionRequest;
        use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
        use deepseek_recipe_core::conversation::ReasoningEffort;
        for (name, local) in [
            ("minimal", Some(ReasoningEffort::Low)),
            ("medium", Some(ReasoningEffort::High)),
            ("xhigh", Some(ReasoningEffort::Xhigh)),
            ("max", Some(ReasoningEffort::Max)),
            ("none", None),
        ] {
            let raw = serde_json::json!({
                "model":"model", "messages":[{"role":"user","content":"Hi"}],
                "reasoning_effort":name
            });
            let converted = serde_json::from_value::<ChatCompletionRequest>(raw)
                .unwrap()
                .convert(ConversionOptions::default())
                .unwrap();
            let enabled = local.is_some();
            let mut conversation =
                conversation(&[serde_json::json!({"role":"user","content":"Hi"})]).unwrap();
            conversation.thinking_mode = enabled;
            conversation.reasoning_effort = local;
            let encoding = DeepseekV41Encoding::new();
            assert_eq!(
                encoding.render_conversation(&conversation).prompt,
                encoding.render_conversation(&converted.conversation).prompt
            );
        }
    }

    #[test]
    fn tools_and_required_choice_configure_prompt_and_parser() {
        let encoder = encoder();
        let plan = encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
                "thinking": {"type": "disabled"},
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"]
                        }
                    }
                }],
                "tool_choice": "required"
            }))
            .unwrap();
        assert!(!plan.tokens.is_empty());
        assert!(plan.parsing.parse_tool_calls);
        assert!(plan.parsing.tool_call_initial_stage);
        assert!(!plan.thinking);
    }

    #[test]
    fn tool_choice_none_disables_tool_prompt_and_parser() {
        let encoder = encoder();
        let plan = encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Answer normally."}],
                "thinking": {"type": "disabled"},
                "tools": [{"type": "function", "function": {
                    "name": "ping", "parameters": {"type": "object"}
                }}],
                "tool_choice": "none"
            }))
            .unwrap();
        assert!(!plan.parsing.parse_tool_calls);
        assert!(!plan.parsing.tool_call_initial_stage);
        assert!(!plan.tokens.is_empty());
    }

    #[test]
    fn named_tool_choice_selects_only_the_named_definition() {
        use deepseek_recipe::openai::chat_completion::request::ChatCompletionRequest;
        use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
        let raw = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "Ping."}],
            "thinking": {"type": "disabled"},
            "tools": [
                {"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "other", "parameters": {"type": "object"}}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "ping"}}
        });
        let converted = serde_json::from_value::<ChatCompletionRequest>(raw.clone())
            .unwrap()
            .convert(ConversionOptions::default())
            .unwrap();
        assert_eq!(converted.conversation.tools.len(), 1);
        assert_eq!(converted.conversation.tools[0].name, "ping");
        let plan = encoder().prepare(&raw).unwrap();
        assert!(plan.parsing.parse_tool_calls);
        assert!(plan.parsing.tool_call_initial_stage);
    }

    #[test]
    fn malformed_tool_choices_are_rejected_before_encoding() {
        let cases = [
            serde_json::json!({
                "model": "model", "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{"type": "web_search", "function": {"name": "ping"}}]
            }),
            serde_json::json!({
                "model": "model", "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{"type": "function", "function": {"name": "ping"}}],
                "tool_choice": {"type": "function", "function": {"name": "missing"}}
            }),
            serde_json::json!({
                "model": "model", "messages": [{"role": "user", "content": "Hi"}],
                "tools": [
                    {"type": "function", "function": {"name": "ping"}},
                    {"type": "function", "function": {"name": "ping"}}
                ]
            }),
        ];
        for raw in cases {
            let error = encoder().prepare(&raw).unwrap_err();
            assert!(error.starts_with("protocol:"), "{error}");
        }
    }

    #[test]
    fn historical_tool_call_and_result_use_official_conversion() {
        let plan = encoder()
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [
                    {"role": "user", "content": "Weather in Tokyo?"},
                    {"role": "assistant", "content": null, "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\":\"Tokyo\"}"}
                    }]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
                ],
                "thinking": {"type": "disabled"},
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"}
                    }
                }]
            }))
            .unwrap();
        assert!(!plan.tokens.is_empty());
        assert!(plan.parsing.parse_tool_calls);
        assert!(!plan.parsing.tool_call_initial_stage);
    }

    #[test]
    fn required_tool_choice_with_thinking_is_rejected_by_recipe() {
        let error = encoder()
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "enabled"},
                "tools": [{"type": "function", "function": {
                    "name": "ping", "parameters": {"type": "object"}
                }}],
                "tool_choice": "required"
            }))
            .unwrap_err();
        assert!(error.starts_with("protocol:"));
    }

    #[test]
    fn strict_tool_schema_is_explicitly_rejected() {
        let error = encoder()
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "disabled"},
                "tools": [{"type": "function", "function": {
                    "name": "ping", "parameters": {"type": "object"}, "strict": true
                }}]
            }))
            .unwrap_err();
        assert_eq!(error, "protocol: strict tool schemas are not supported");
    }

    #[tokio::test]
    async fn official_parser_builds_nonstream_tool_call_response() {
        let encoder = encoder();
        let plan = encoder
            .prepare(&serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
                "thinking": {"type": "disabled"},
                "tools": [{"type": "function", "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"}
                }}],
                "tool_choice": "required"
            }))
            .unwrap();
        let value = encoder
            .response(plan, "model", "id", 8, &[5, 1], "stop")
            .await
            .unwrap();
        let choice = &value["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(choice["message"]["tool_calls"][0]["type"], "function");
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"location\": \"Tokyo\"}"
        );
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn length_and_invalid_eos() {
        let encoder = encoder();
        let value = encoder
            .response(plan(&encoder, true), "model", "id", 5, &[2], "length")
            .await
            .unwrap();
        assert_eq!(value["choices"][0]["finish_reason"], "length");
        assert_eq!(value["usage"]["completion_tokens"], 1);
        assert!(
            encoder
                .response(plan(&encoder, true), "model", "id", 5, &[1, 4], "stop")
                .await
                .is_err()
        );
        assert!(
            encoder
                .response(plan(&encoder, true), "model", "id", 5, &[1], "length")
                .await
                .is_err()
        );
    }

    #[test]
    fn literal_image_marker_is_plain_text() {
        let c = conversation(&[
            serde_json::json!({"role":"user", "content":"literal <｜deepseek_image｜>"}),
        ])
        .unwrap();
        let rendered = DeepseekV41Encoding::new().render_conversation(&c);
        assert!(rendered.prompt.contains("<｜deepseek_image｜>"));
    }

    #[test]
    fn structured_image_stops_at_vision_boundary() {
        let e = conversation(&[serde_json::json!({"role":"user", "content":[{"type":"image_url","image_url":{"url":"x"}}]})]).unwrap_err();
        assert_eq!(e, AdapterError::ImageContentRequiresVision);
    }
}
