fn main() {
    if std::env::var_os("CARGO_FEATURE_NATIVE_BRIDGE").is_some() {
        let dir = std::env::var("DSV41_BRIDGE_LIB_DIR")
            .expect("native-bridge requires DSV41_BRIDGE_LIB_DIR");
        println!("cargo:rustc-link-search=native={dir}");
        if std::env::var_os("CARGO_FEATURE_NATIVE_MODEL").is_some() {
            println!("cargo:rustc-link-lib=dylib=dsv41_runtime_bridge_mlx");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
            println!("cargo:rerun-if-changed={dir}/libdsv41_runtime_bridge_mlx.dylib");
        } else {
            println!("cargo:rustc-link-lib=static=dsv41_runtime_bridge");
            println!("cargo:rerun-if-changed={dir}/libdsv41_runtime_bridge.a");
        }
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rerun-if-env-changed=DSV41_BRIDGE_LIB_DIR");
    }
}
