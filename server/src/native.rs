#![cfg(feature = "native-bridge")]

use std::ffi::c_void;

const ABI_VERSION: u32 = 1;

#[repr(C)]
struct Bridge;

unsafe extern "C" {
    fn dsv41_bridge_create(abi_version: u32) -> *mut Bridge;
    fn dsv41_bridge_destroy(bridge: *mut Bridge);
}

/// Small ownership-safe wrapper. Generation methods are added only after the
/// recipe token/event adapter is fixed; this type currently validates create/drop.
pub struct NativeBridge {
    raw: *mut Bridge,
}

unsafe impl Send for NativeBridge {}
unsafe impl Sync for NativeBridge {}

impl NativeBridge {
    pub fn connect() -> Option<Self> {
        let raw = unsafe { dsv41_bridge_create(ABI_VERSION) };
        (!raw.is_null()).then_some(Self { raw })
    }

    pub fn as_context(&self) -> *mut c_void {
        self.raw.cast()
    }
}

impl Drop for NativeBridge {
    fn drop(&mut self) {
        unsafe { dsv41_bridge_destroy(self.raw) }
    }
}
