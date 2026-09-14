use serde_json::Value;

#[derive(Clone, Copy, Debug)]
pub struct GenerationOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
}

/// Inference boundary owned by the runtime bridge, independent of HTTP.
///
/// The developer server deliberately starts with an unconnected backend so
/// protocol validation can be exercised before native generation is wired in.
pub trait RuntimeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn complete(
        &self,
        _messages: &[Value],
        _stream: bool,
        _options: GenerationOptions,
    ) -> Result<Value, BackendError>;
}

#[derive(Debug)]
pub enum BackendError {
    Unavailable,
}

pub struct UnconnectedBackend;

impl RuntimeBackend for UnconnectedBackend {
    fn name(&self) -> &'static str {
        "unconnected"
    }

    fn complete(
        &self,
        _messages: &[Value],
        _stream: bool,
        options: GenerationOptions,
    ) -> Result<Value, BackendError> {
        let _ = (options.max_tokens, options.temperature, options.seed);
        Err(BackendError::Unavailable)
    }
}
