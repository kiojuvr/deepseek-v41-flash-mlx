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

#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
#[allow(dead_code)]
pub struct NativeRuntimeBackend {
    bridge: crate::native::NativeBridge,
    encoder: crate::recipe_adapter::RecipeEncoder,
}

#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
#[allow(dead_code)]
impl NativeRuntimeBackend {
    pub fn new(bridge: crate::native::NativeBridge, encoder: crate::recipe_adapter::RecipeEncoder) -> Self {
        Self { bridge, encoder }
    }
}

#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
#[allow(dead_code)]
impl RuntimeBackend for NativeRuntimeBackend {
    fn name(&self) -> &'static str { "native-bridge" }

    fn complete(&self, messages: &[Value], stream: bool, options: GenerationOptions) -> Result<Value, BackendError> {
        let tokens = self.encoder.encode(messages).map_err(|_| BackendError::Unavailable)?;
        let max = options.max_tokens.unwrap_or(16);
        let temp = options.temperature.unwrap_or(0.0);
        let seed = options.seed.unwrap_or(0);
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
        let sink = events.clone();
        self.bridge.submit(&tokens, max, temp, seed, move |event| {
            if event.kind == crate::native::EVENT_TOKEN { sink.lock().unwrap().push(event.token_id); }
            true
        }).map_err(|_| BackendError::Unavailable)?;
        let ids = events.lock().unwrap().clone();
        Ok(serde_json::json!({"object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":if stream {"stop"} else {"stop"}}],"usage":{"prompt_tokens":tokens.len(),"completion_tokens":ids.len(),"total_tokens":tokens.len()+ids.len()}}))
    }
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
