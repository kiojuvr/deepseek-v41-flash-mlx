fn main() {
    if std::env::var_os("CARGO_FEATURE_NATIVE_BRIDGE").is_some() {
        let dir = std::env::var("DSV41_BRIDGE_LIB_DIR")
            .expect("native-bridge requires DSV41_BRIDGE_LIB_DIR");
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=static=dsv41_runtime_bridge");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rerun-if-env-changed=DSV41_BRIDGE_LIB_DIR");
    }
}
