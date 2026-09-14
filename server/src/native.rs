#![cfg(feature = "native-bridge")]
use std::ffi::{CStr, c_char, c_void};
const ABI_VERSION: u32 = 1;
#[repr(C)]
struct Bridge;
#[repr(C)]
struct Request {
    abi_version: u32,
    input_tokens: *const u32,
    input_token_count: usize,
    max_new_tokens: u32,
    temperature: f32,
    seed: u64,
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
    fn dsv41_bridge_destroy(bridge: *mut Bridge);
    fn dsv41_bridge_submit(
        bridge: *mut Bridge,
        request: *const Request,
        callback: extern "C" fn(*const Event, *mut c_void) -> i32,
        context: *mut c_void,
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
    let f = unsafe { &mut *(c as *mut Box<dyn FnMut(EventView) -> bool>) };
    if f(v) { 0 } else { 1 }
}
pub struct NativeBridge {
    raw: *mut Bridge,
}
unsafe impl Send for NativeBridge {}
unsafe impl Sync for NativeBridge {}
impl NativeBridge {
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
        let r = Request {
            abi_version: ABI_VERSION,
            input_tokens: tokens.as_ptr(),
            input_token_count: tokens.len(),
            max_new_tokens: max,
            temperature: temp,
            seed,
        };
        let mut f: Box<dyn FnMut(EventView) -> bool> = Box::new(callback);
        let mut id = 0;
        let s = unsafe {
            dsv41_bridge_submit(
                self.raw,
                &r,
                event_callback,
                (&mut f as *mut _).cast(),
                &mut id,
            )
        };
        drop(f);
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
