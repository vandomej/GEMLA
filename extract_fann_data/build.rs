fn main() {
    // Replace this with the path to the directory containing `fann.lib`
    let lib_dir = "F://vandomej/Downloads/vcpkg/packages/fann_x64-windows/lib";

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=static=fann");
    // Use `dylib=fann` instead of `static=fann` if you're linking dynamically

    // If there are any additional directories where the compiler can find header files, you can specify them like this:
    // println!("cargo:include={}", path_to_include_directory);
}
