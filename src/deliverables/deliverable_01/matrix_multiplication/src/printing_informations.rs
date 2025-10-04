use std::fs::File;
use std::io::Write;

pub fn print_and_write(file: &mut File, message: &str) {
    println!("{}", message);
    writeln!(file, "{}", message).expect("Failed to write to file");
}

pub fn print_cpu_results(
    file: &mut File,
    size: i32,
    sequential_metrics: (f32, f32, f32),
    rayon_metrics: (f32, f32, f32),
    tokio_metrics: (f32, f32, f32),
    std_thread_metrics: (f32, f32, f32)
) {
    let message = format!("Matrix Size: {}x{}", size, size);
    print_and_write(file, &message);

    let message = format!(
        "Sequential Matrix Multiplication - Total Time: {:.6} seconds, Average Time: {:.6} seconds, Standard Deviation: {:.6} seconds",
        sequential_metrics.0, sequential_metrics.1, sequential_metrics.2
    );
    print_and_write(file, &message);

    let message = format!(
        "Rayon Matrix Multiplication - Total Time: {:.6} seconds, Average Time: {:.6} seconds, Standard Deviation: {:.6} seconds",
        rayon_metrics.0, rayon_metrics.1, rayon_metrics.2
    );
    print_and_write(file, &message);

    let message = format!(
        "Tokio Matrix Multiplication - Total Time: {:.6} seconds, Average Time: {:.6} seconds, Standard Deviation: {:.6} seconds",
        tokio_metrics.0, tokio_metrics.1, tokio_metrics.2
    );
    print_and_write(file, &message);

    let message = format!(
        "Std::thread Matrix Multiplication - Total Time: {:.6} seconds, Average Time: {:.6} seconds, Standard Deviation: {:.6} seconds",
        std_thread_metrics.0, std_thread_metrics.1, std_thread_metrics.2
    );
    print_and_write(file, &message);

    print_and_write(file, "");
}

pub fn print_xpu_results(
    file: &mut File,
    size: i32,
    total_metrics: (f32, f32, f32),
    kernel_metrics: (f32, f32, f32),
    last_index: bool
) {
    let message = format!("Matrix Size: {}x{}", size, size);
    print_and_write(file, &message);

    let message = format!(
        "XPU OpenCL Total Time (includes transfers) - Total Time: {:.6} seconds, Average Time: {:.6} seconds, Standard Deviation: {:.6} seconds",
        total_metrics.0, total_metrics.1, total_metrics.2
    );
    print_and_write(file, &message);

    let message = format!(
        "XPU OpenCL Kernel Time (kernel only) - Total Time: {:.6} seconds, Average Time: {:.6} seconds, Standard Deviation: {:.6} seconds",
        kernel_metrics.0, kernel_metrics.1, kernel_metrics.2
    );
    print_and_write(file, &message);

    if !last_index {
		print_and_write(file, "");
	}
}
