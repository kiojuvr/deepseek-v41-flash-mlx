use serde_json::Value;
#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, atomic::AtomicBool};

#[cfg_attr(
    not(all(feature = "native-bridge", feature = "recipe-adapter")),
    allow(dead_code)
)]
#[derive(Clone, Copy, Debug)]
pub struct GenerationOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
    pub include_usage: bool,
}

pub trait RuntimeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn complete(
        &self,
        request: &Value,
        stream: bool,
        options: GenerationOptions,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Value, BackendError>;
    fn stream(
        self: Arc<Self>,
        _request: Value,
        _options: GenerationOptions,
        _permit: tokio::sync::OwnedSemaphorePermit,
    ) -> Result<axum::response::Response, BackendError> {
        Err(BackendError::Unavailable)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum BackendError {
    Unavailable,
    UnsupportedStream,
    Busy,
    Invalid(String),
    Runtime(String),
}

#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
pub struct NativeRuntimeBackend {
    worker: crate::owner_worker::OwnerWorker<NativeJob, (Result<u64, i32>, CommittedEvents)>,
    encoder: crate::recipe_adapter::RecipeEncoder,
    model: String,
    loaded: bool,
}

#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
static NEXT_NONSTREAM_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
impl NativeRuntimeBackend {
    pub fn new<F>(
        create: F,
        encoder: crate::recipe_adapter::RecipeEncoder,
        model: String,
        loaded: bool,
    ) -> Result<Self, String>
    where
        F: FnOnce() -> Result<crate::native::NativeBridge, String> + Send + 'static,
    {
        let worker = crate::owner_worker::OwnerWorker::start(create, |bridge, job: NativeJob| {
            let max = job.options.max_tokens.unwrap_or(16);
            let temperature = job.options.temperature.unwrap_or(0.0);
            let seed = job.options.seed.unwrap_or(0);
            let events = std::sync::Arc::new(std::sync::Mutex::new(CommittedEvents::default()));
            let sink = events.clone();
            let cancelled = job.cancelled.clone();
            let output = job.output;
            let status = bridge.submit_cancellable(
                &job.tokens,
                max,
                temperature,
                seed,
                job.cancelled,
                move |event| {
                    let token =
                        (event.kind == crate::native::EVENT_TOKEN).then_some(event.token_id);
                    let valid = match sink.lock() {
                        Ok(mut log) => log.push(event, max as usize),
                        Err(_) => false,
                    };
                    if !valid {
                        return false;
                    }
                    if let (Some(token), Some(output)) = (token, &output)
                        && (cancelled.load(Ordering::Acquire)
                            || output.blocking_send(token).is_err())
                    {
                        cancelled.store(true, Ordering::Release);
                        return false;
                    }
                    true
                },
            );
            let log = match events.lock() {
                Ok(mut log) => std::mem::take(&mut *log),
                Err(_) => CommittedEvents {
                    invalid: true,
                    ..Default::default()
                },
            };
            if let Some(terminal) = &log.terminal {
                eprintln!(
                    "native request={} terminal={} committed={} budget={} temperature={} seed={}",
                    terminal.request_id,
                    terminal.finish_reason.as_deref().unwrap_or("error"),
                    log.tokens.len(),
                    max,
                    temperature,
                    seed
                );
            }
            (status, log)
        })?;
        Ok(Self {
            worker,
            encoder,
            model,
            loaded,
        })
    }
}

#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
struct NativeJob {
    tokens: Vec<u32>,
    options: GenerationOptions,
    cancelled: Arc<AtomicBool>,
    output: Option<tokio::sync::mpsc::Sender<u32>>,
}

#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
#[derive(Default)]
struct CommittedEvents {
    tokens: Vec<u32>,
    id: Option<u64>,
    terminal: Option<crate::native::EventView>,
    invalid: bool,
}
#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
impl CommittedEvents {
    fn push(&mut self, event: crate::native::EventView, max: usize) -> bool {
        use crate::native::*;
        if self.invalid
            || self.terminal.is_some()
            || event.request_id == 0
            || self.id.is_some_and(|id| id != event.request_id)
            || event.committed_index != self.tokens.len() as u64
        {
            self.invalid = true;
            return false;
        }
        self.id = Some(event.request_id);
        match event.kind {
            EVENT_TOKEN if self.tokens.len() < max && event.token_id < 129280 => {
                self.tokens.push(event.token_id)
            }
            EVENT_FINISHED | EVENT_ERROR => self.terminal = Some(event),
            _ => {
                self.invalid = true;
                return false;
            }
        }
        true
    }
    fn finish(&self, status: Result<u64, i32>, max: usize) -> Result<&str, BackendError> {
        use crate::native::*;
        if self.invalid {
            return Err(BackendError::Runtime(
                "invalid native event sequence".into(),
            ));
        }
        if status == Err(-3) {
            return Err(BackendError::Busy);
        }
        if status == Err(-1) {
            return Err(BackendError::Invalid("native request rejected".into()));
        }
        let terminal = self
            .terminal
            .as_ref()
            .ok_or_else(|| BackendError::Runtime("native request has no terminal event".into()))?;
        if terminal.kind == EVENT_ERROR {
            if terminal.error_code.as_deref() == Some("runtime_unavailable") && status == Err(-2) {
                return Err(BackendError::Unavailable);
            }
            return Err(BackendError::Runtime(
                terminal
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "native runtime error".into()),
            ));
        }
        if status != Ok(terminal.request_id) {
            return Err(BackendError::Runtime(
                "native status and terminal event disagree".into(),
            ));
        }
        match terminal.finish_reason.as_deref() {
            Some("length") if self.tokens.len() == max => Ok("length"),
            Some("stop") => Ok("stop"),
            Some("cancelled") => Err(BackendError::Runtime("generation cancelled".into())),
            _ => Err(BackendError::Runtime(
                "invalid native finish reason or token count".into(),
            )),
        }
    }

    fn finish_local_stop(
        &self,
        status: Result<u64, i32>,
        consumed: usize,
        max: usize,
    ) -> Result<(), BackendError> {
        use crate::native::EVENT_FINISHED;
        if self.invalid || consumed == 0 || self.tokens.len() < consumed {
            return Err(BackendError::Runtime(
                "invalid native event sequence for local stop".into(),
            ));
        }
        let terminal = self
            .terminal
            .as_ref()
            .ok_or_else(|| BackendError::Runtime("native request has no terminal event".into()))?;
        let valid_reason = match terminal.finish_reason.as_deref() {
            Some("cancelled" | "stop") => true,
            Some("length") => self.tokens.len() == max,
            _ => false,
        };
        if status != Ok(terminal.request_id) || terminal.kind != EVENT_FINISHED || !valid_reason {
            return Err(BackendError::Runtime(
                "invalid native terminal after local stop".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(all(feature = "native-bridge", feature = "recipe-adapter"))]
impl RuntimeBackend for NativeRuntimeBackend {
    fn name(&self) -> &'static str {
        if self.loaded {
            "native-model"
        } else {
            "native-bridge"
        }
    }

    fn complete(
        &self,
        request: &Value,
        stream: bool,
        options: GenerationOptions,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Value, BackendError> {
        if stream && self.loaded {
            return Err(BackendError::UnsupportedStream);
        }
        let plan = self
            .encoder
            .prepare(request)
            .map_err(BackendError::Invalid)?;
        let tokens = plan.tokens.clone();
        let max = options.max_tokens.unwrap_or(16);
        let prompt_tokens = tokens.len();
        if self.loaded && crate::recipe_adapter::RecipeEncoder::has_stop_sequences(&plan) {
            let eos = self.encoder.eos_token().map_err(BackendError::Invalid)?;
            let (output, input) = tokio::sync::mpsc::channel(8);
            let (done, finished) = tokio::sync::oneshot::channel();
            let job = NativeJob {
                tokens,
                options,
                cancelled: cancelled.clone(),
                output: Some(output),
            };
            let id = format!(
                "chatcmpl-live-{}-{}",
                std::process::id(),
                NEXT_NONSTREAM_ID.fetch_add(1, Ordering::Relaxed)
            );
            return std::thread::scope(|scope| {
                let native = scope.spawn(|| {
                    let outcome = self.worker.call(job);
                    let summary = match &outcome {
                        Ok((status, log)) => log
                            .finish(*status, max as usize)
                            .map(|reason| (reason.to_owned(), log.tokens.len()))
                            .map_err(|error| format!("{error:?}")),
                        Err(error) => Err(error.clone()),
                    };
                    let _ = done.send(summary);
                    outcome
                });
                let parsed =
                    tokio::runtime::Handle::current().block_on(self.encoder.incremental_response(
                        plan,
                        &self.model,
                        &id,
                        crate::recipe_adapter::IncrementalSource {
                            prompt_tokens,
                            eos,
                            tokens: input,
                            done: finished,
                            cancelled: cancelled.clone(),
                        },
                    ));
                if parsed.is_err() {
                    cancelled.store(true, Ordering::Release);
                }
                let native = native
                    .join()
                    .map_err(|_| BackendError::Runtime("native request thread panicked".into()))?
                    .map_err(BackendError::Runtime)?;
                let parsed = parsed.map_err(BackendError::Runtime)?;
                if parsed.locally_stopped {
                    native
                        .1
                        .finish_local_stop(native.0, parsed.consumed, max as usize)?;
                } else {
                    native.1.finish(native.0, max as usize)?;
                }
                Ok(parsed.value)
            });
        }
        let (status, log) = self
            .worker
            .call(NativeJob {
                tokens,
                options,
                cancelled,
                output: None,
            })
            .map_err(BackendError::Runtime)?;
        let reason = log.finish(status, max as usize)?;
        let id = format!("chatcmpl-{}-{}", std::process::id(), log.id.unwrap_or(0));
        // complete is run exclusively on spawn_blocking, never on an async worker.
        tokio::runtime::Handle::current()
            .block_on(self.encoder.response(
                plan,
                &self.model,
                &id,
                prompt_tokens,
                &log.tokens,
                reason,
            ))
            .map_err(BackendError::Runtime)
    }
    fn stream(
        self: Arc<Self>,
        request: Value,
        options: GenerationOptions,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) -> Result<axum::response::Response, BackendError> {
        if !self.loaded {
            return Err(BackendError::Unavailable);
        }
        let plan = self
            .encoder
            .prepare(&request)
            .map_err(BackendError::Invalid)?;
        let tokens = plan.tokens.clone();
        let eos = self.encoder.eos_token().map_err(BackendError::Invalid)?;
        let prompt = tokens.len();
        let (output, input) = tokio::sync::mpsc::channel(8);
        let (done, finished) = tokio::sync::oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));

        let model = self.model.clone();
        let job = NativeJob {
            tokens,
            options,
            cancelled: cancelled.clone(),
            output: Some(output),
        };
        let worker = self.clone();
        let encoder = self.encoder.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let result = worker
                .worker
                .call(job)
                .map_err(BackendError::Runtime)
                .and_then(|(status, log)| {
                    let reason = log
                        .finish(status, options.max_tokens.unwrap_or(16) as usize)?
                        .to_owned();
                    Ok(crate::streaming::StreamEnd {
                        reason,
                        committed: log.tokens.len(),
                    })
                });
            let _ = done.send(result);
        });
        Ok(crate::streaming::response(
            encoder,
            plan,
            model,
            prompt,
            eos,
            input,
            finished,
            cancelled,
            options.include_usage,
        ))
    }
}

pub struct UnconnectedBackend;
impl RuntimeBackend for UnconnectedBackend {
    fn name(&self) -> &'static str {
        "unconnected"
    }
    fn complete(
        &self,
        _: &Value,
        _: bool,
        options: GenerationOptions,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Value, BackendError> {
        let _ = (
            options.max_tokens,
            options.temperature,
            options.seed,
            cancelled,
        );
        Err(BackendError::Unavailable)
    }
}

#[cfg(all(test, feature = "native-bridge", feature = "recipe-adapter"))]
mod tests {
    use super::*;
    use crate::native::*;
    fn event(kind: u32, index: u64, reason: Option<&str>) -> EventView {
        EventView {
            kind,
            request_id: 7,
            committed_index: index,
            token_id: 42,
            finish_reason: reason.map(str::to_owned),
            error_code: None,
            error_message: None,
        }
    }
    #[test]
    fn terminal_accounting_and_failures() {
        let mut log = CommittedEvents::default();
        assert!(log.push(event(EVENT_TOKEN, 0, None), 1));
        assert!(log.finish(Ok(7), 1).is_err());
        assert!(log.push(event(EVENT_FINISHED, 1, Some("length")), 1));
        assert_eq!(log.finish(Ok(7), 1).unwrap(), "length");
        assert!(log.finish(Err(-4), 1).is_err());
        assert!(log.finish(Ok(8), 1).is_err());
        assert!(!log.push(event(EVENT_TOKEN, 1, None), 2));
        assert!(log.finish(Ok(7), 1).is_err());
    }
    #[test]
    fn errors_and_out_of_order_tokens_never_succeed() {
        let mut log = CommittedEvents::default();
        assert!(!log.push(event(EVENT_TOKEN, 1, None), 2));
        assert!(log.finish(Ok(7), 2).is_err());
        let mut log = CommittedEvents::default();
        assert!(log.push(event(EVENT_ERROR, 0, None), 2));
        assert!(log.finish(Ok(7), 2).is_err());
    }

    #[test]
    fn local_stop_requires_cancelled_terminal_after_consumed_tokens() {
        let mut log = CommittedEvents::default();
        assert!(log.push(event(EVENT_TOKEN, 0, None), 4));
        assert!(log.push(event(EVENT_TOKEN, 1, None), 4));
        assert!(log.push(event(EVENT_FINISHED, 2, Some("cancelled")), 4));
        assert!(log.finish_local_stop(Ok(7), 2, 4).is_ok());
        assert!(log.finish_local_stop(Ok(7), 3, 4).is_err());
        let mut raced = CommittedEvents::default();
        assert!(raced.push(event(EVENT_TOKEN, 0, None), 4));
        assert!(raced.push(event(EVENT_FINISHED, 1, Some("stop")), 4));
        assert!(raced.finish_local_stop(Ok(7), 1, 4).is_ok());
    }
}
