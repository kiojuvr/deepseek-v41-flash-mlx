//! Protocol to pinned DeepSeek recipe boundary.
//!
//! This module deliberately performs no tokenization or inference. It converts
//! OpenAI-shaped message values into the recipe's typed conversation so the
//! server and native runtime share one prompt semantic owner.

use deepseek_recipe_core::{conversation::Conversation, messages::InputMessage};
use deepseek_recipe_encoding::{PromptEncoding, v4::dsv41::DeepseekV41Encoding};
use serde_json::Value;

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
                    "text" => out.push_str(part.get("text").and_then(Value::as_str).ok_or(AdapterError::InvalidContent)?),
                    "image_url" | "input_image" => return Err(AdapterError::ImageContentRequiresVision),
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
        let role = message.get("role").and_then(Value::as_str).ok_or(AdapterError::MissingRole)?;
        let text = content(message.get("content").ok_or(AdapterError::InvalidContent)?)?;
        out.messages.push(match role {
            "system" => InputMessage::System { content: text },
            "user" => InputMessage::User { content: text, image_sources: vec![] },
            "assistant" => InputMessage::Assistant { content: text, reasoning_content: None, tool_calls: None },
            "tool" => InputMessage::Tool { content: text, image_sources: vec![], tool_call_id: message.get("tool_call_id").and_then(Value::as_str).unwrap_or_default().to_owned() },
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
    let conversation = conversation(messages).map_err(|e| format!("protocol: {e:?}"))?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("tokenizer: {e}"))?;
    DeepseekV41Encoding::new()
        .with_tokenizer(tokenizer)
        .encode(&conversation)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_recipe_encoding::{v4::dsv41::DeepseekV41Encoding, PromptEncoding};

    #[test]
    fn literal_image_marker_is_plain_text() {
        let c = conversation(&[serde_json::json!({"role":"user", "content":"literal <｜deepseek_image｜>"})]).unwrap();
        let rendered = DeepseekV41Encoding::new().render_conversation(&c);
        assert!(rendered.prompt.contains("<｜deepseek_image｜>"));
    }

    #[test]
    fn structured_image_stops_at_vision_boundary() {
        let e = conversation(&[serde_json::json!({"role":"user", "content":[{"type":"image_url","image_url":{"url":"x"}}]})]).unwrap_err();
        assert_eq!(e, AdapterError::ImageContentRequiresVision);
    }
}
