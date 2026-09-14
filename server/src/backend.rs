use serde_json::Value;

/// Inference boundary owned by the runtime bridge, independent of HTTP.
///
/// The developer server deliberately starts with an unconnected backend so
/// protocol validation can be exercised before native generation is wired in.
pub trait RuntimeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn complete(&self, _messages: &[Value], _stream: bool) -> Result<Value, BackendError>;
}

#[derive(Debug)]
pub enum BackendError {
    Unavailable,
}

pub struct UnconnectedBackend;

impl RuntimeBackend for UnconnectedBackend {
    fn name(&self) -> &'static str { "unconnected" }

    fn complete(&self, _messages: &[Value], _stream: bool) -> Result<Value, BackendError> {
        Err(BackendError::Unavailable)
    }
}
