#![cfg(feature = "native-bridge")]
#![allow(dead_code)]
use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
const ABI_VERSION: u32 = 1;
#[repr(C)]
struct Bridge {
    _private: [u8; 0],
}
#[repr(C)]
struct Request {
    abi_version: u32,
    input_tokens: *const u32,
    input_token_count: usize,
    max_new_tokens: u32,
    temperature: f32,
    seed: u64,
}
#[cfg(feature = "native-model")]
#[repr(C)]
struct ModelConfig {
    abi_version: u32,
    checkpoint: *const c_char,
    summary: *const c_char,
    metadata: *const c_char,
    provenance: *const c_char,
    stop_tokens: *const u32,
    stop_token_count: usize,
}
#[repr(C)]
struct Event {
    kind: u32,
    request_id: u64,
    committed_index: u64,
    token_id: u32,
    finish_reason: *const c_char,
    error_code: *const c_char,
    error_message: *const c_char,
}
unsafe extern "C" {
    fn dsv41_bridge_create(abi_version: u32) -> *mut Bridge;
    #[cfg(feature = "native-model")]
    fn dsv41_bridge_create_model(
        config: *const ModelConfig,
        error: *mut c_char,
        capacity: usize,
    ) -> *mut Bridge;
    fn dsv41_bridge_destroy(bridge: *mut Bridge);
    fn dsv41_bridge_submit_cancellable(
        bridge: *mut Bridge,
        request: *const Request,
        callback: extern "C" fn(*const Event, *mut c_void) -> i32,
        context: *mut c_void,
        poll: extern "C" fn(*mut c_void) -> i32,
        request_id: *mut u64,
    ) -> i32;
    fn dsv41_bridge_cancel(bridge: *mut Bridge, request_id: u64) -> i32;
}
pub const EVENT_TOKEN: u32 = 1;
pub const EVENT_FINISHED: u32 = 2;
pub const EVENT_ERROR: u32 = 3;
pub struct EventView {
    pub kind: u32,
    pub request_id: u64,
    pub committed_index: u64,
    pub token_id: u32,
    pub finish_reason: Option<String>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}
fn text(p: *const c_char) -> Option<String> {
    if p.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }
}
extern "C" fn event_callback(e: *const Event, c: *mut c_void) -> i32 {
    if e.is_null() || c.is_null() {
        return 1;
    }
    let state = unsafe { &mut *(c as *mut CallbackState) };
    if state.panicked {
        return 1;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        let e = unsafe { &*e };
        let v = EventView {
            kind: e.kind,
            request_id: e.request_id,
            committed_index: e.committed_index,
            token_id: e.token_id,
            finish_reason: text(e.finish_reason),
            error_code: text(e.error_code),
            error_message: text(e.error_message),
        };
        if (state.callback)(v) { 0 } else { 1 }
    })) {
        Ok(status) => status,
        Err(_) => {
            state.panicked = true;
            1
        }
    }
}
extern "C" fn cancel_poll(c: *mut c_void) -> i32 {
    let state = unsafe { &*(c as *const CallbackState) };
    i32::from(state.panicked || state.cancelled.load(std::sync::atomic::Ordering::Acquire))
}
struct CallbackState {
    callback: Box<dyn FnMut(EventView) -> bool>,
    panicked: bool,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
pub struct NativeBridge {
    raw: *mut Bridge,
}
// The MLX model is thread-affine. Construction, submit and Drop stay on its
// owner thread; do not implement Send/Sync for this raw-handle wrapper.
impl NativeBridge {
    #[cfg(feature = "native-model")]
    pub fn load(
        checkpoint: &str,
        summary: &str,
        metadata: &str,
        provenance: &str,
        stops: &[u32],
    ) -> Result<Self, String> {
        use std::ffi::CString;
        let paths = [checkpoint, summary, metadata, provenance].map(CString::new);
        let [checkpoint, summary, metadata, provenance] = paths;
        let checkpoint = checkpoint.map_err(|e| e.to_string())?;
        let summary = summary.map_err(|e| e.to_string())?;
        let metadata = metadata.map_err(|e| e.to_string())?;
        let provenance = provenance.map_err(|e| e.to_string())?;
        let config = ModelConfig {
            abi_version: ABI_VERSION,
            checkpoint: checkpoint.as_ptr(),
            summary: summary.as_ptr(),
            metadata: metadata.as_ptr(),
            provenance: provenance.as_ptr(),
            stop_tokens: stops.as_ptr(),
            stop_token_count: stops.len(),
        };
        let mut error = [0 as c_char; 1024];
        let raw = unsafe { dsv41_bridge_create_model(&config, error.as_mut_ptr(), error.len()) };
        if raw.is_null() {
            Err(text(error.as_ptr()).unwrap_or_else(|| "model initialization failed".into()))
        } else {
            Ok(Self { raw })
        }
    }
    pub fn connect() -> Option<Self> {
        let p = unsafe { dsv41_bridge_create(ABI_VERSION) };
        (!p.is_null()).then_some(Self { raw: p })
    }
    pub fn submit<F>(
        &self,
        tokens: &[u32],
        max: u32,
        temp: f32,
        seed: u64,
        callback: F,
    ) -> Result<u64, i32>
    where
        F: FnMut(EventView) -> bool + 'static,
    {
        self.submit_cancellable(
            tokens,
            max,
            temp,
            seed,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            callback,
        )
    }
    pub fn submit_cancellable<F>(
        &self,
        tokens: &[u32],
        max: u32,
        temp: f32,
        seed: u64,
        cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
        callback: F,
    ) -> Result<u64, i32>
    where
        F: FnMut(EventView) -> bool + 'static,
    {
        let r = Request {
            abi_version: ABI_VERSION,
            input_tokens: tokens.as_ptr(),
            input_token_count: tokens.len(),
            max_new_tokens: max,
            temperature: temp,
            seed,
        };
        let mut f = CallbackState {
            callback: Box::new(callback),
            panicked: false,
            cancelled,
        };
        let mut id = 0;
        let s = unsafe {
            dsv41_bridge_submit_cancellable(
                self.raw,
                &r,
                event_callback,
                (&mut f as *mut CallbackState).cast(),
                cancel_poll,
                &mut id,
            )
        };
        if f.panicked {
            return Err(-4);
        }
        if s == 0 { Ok(id) } else { Err(s) }
    }
    pub fn cancel(&self, id: u64) -> Result<(), i32> {
        let s = unsafe { dsv41_bridge_cancel(self.raw, id) };
        if s == 0 { Ok(()) } else { Err(s) }
    }
}
impl Drop for NativeBridge {
    fn drop(&mut self) {
        unsafe { dsv41_bridge_destroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_panic_is_contained_at_ffi_boundary() {
        let mut state = CallbackState {
            callback: Box::new(|_| panic!("test sink panic")),
            panicked: false,
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        let event = Event {
            kind: EVENT_TOKEN,
            request_id: 1,
            committed_index: 0,
            token_id: 42,
            finish_reason: std::ptr::null(),
            error_code: std::ptr::null(),
            error_message: std::ptr::null(),
        };
        assert_eq!(
            event_callback(&event, (&mut state as *mut CallbackState).cast()),
            1
        );
        assert!(state.panicked);
        assert_eq!(
            event_callback(&event, (&mut state as *mut CallbackState).cast()),
            1
        );
    }

    #[test]
    fn unconnected_submit_emits_error_event() {
        let bridge = NativeBridge::connect().expect("bridge shell must create");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_cb = seen.clone();
        let result = bridge.submit(&[0, 42, 1000, 42], 4, 0.0, 7, move |event| {
            seen_cb.lock().unwrap().push(event);
            true
        });
        assert_eq!(result, Err(-2));
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].kind, EVENT_ERROR);
        assert_eq!(seen[0].error_code.as_deref(), Some("runtime_unavailable"));
        assert_eq!(seen[0].request_id, 1);
    }
}
