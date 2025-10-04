mod matrix_operations;
mod printing_informations;

use matrix_operations::{
	MultiplicationMethod,
	calculate_metrics,
	generate_matrices,
	matrix_multiplication
};

use printing_informations::{
	print_and_write,
	print_cpu_results,
	print_xpu_results
};

use std::fs::File;

static REPETITIONS: i32 = 10;
static MINIMUM_VALUE: i32 = 1;
static MAXIMUM_VALUE: i32 = 10;

fn main() {
    let cpu_matrix_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let xpu_matrix_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    let mut sequential_duration_vectors: Vec<Vec<f32>> = vec![Vec::new(); cpu_matrix_sizes.len()];
    let mut rayon_duration_vectors: Vec<Vec<f32>> = vec![Vec::new(); cpu_matrix_sizes.len()];
    let mut tokio_duration_vectors: Vec<Vec<f32>> = vec![Vec::new(); cpu_matrix_sizes.len()];
    let mut std_thread_duration_vectors: Vec<Vec<f32>> = vec![Vec::new(); cpu_matrix_sizes.len()];

    let mut xpu_total_duration_vectors: Vec<Vec<f32>> = vec![Vec::new(); xpu_matrix_sizes.len()];
    let mut xpu_kernel_duration_vectors: Vec<Vec<f32>> = vec![Vec::new(); xpu_matrix_sizes.len()];

    for repetition in 0..REPETITIONS {
        println!();
        println!("--- // --- // ---");
        println!("Repetition: {}", repetition + 1);
        println!();

        // CPU implementations
        for (index, &size) in cpu_matrix_sizes.iter().enumerate() {
            println!("CPU - Matrix Size: {}x{}", size, size);

            // Generating two random matrices
            let (matrix_a, matrix_b) = generate_matrices(size, MINIMUM_VALUE, MAXIMUM_VALUE);

            // Sequential matrix multiplication
            let sequential_result = matrix_multiplication(&matrix_a, &matrix_b, MultiplicationMethod::Sequential);
            sequential_duration_vectors[index].push(sequential_result.duration);

            // Rayon parallel matrix multiplication
            let rayon_result = matrix_multiplication(&matrix_a, &matrix_b, MultiplicationMethod::Rayon);
            rayon_duration_vectors[index].push(rayon_result.duration);

            // Tokio parallel matrix multiplication
            let tokio_result = matrix_multiplication(&matrix_a, &matrix_b, MultiplicationMethod::Tokio);
            tokio_duration_vectors[index].push(tokio_result.duration);

            // Standard thread parallel matrix multiplication
            let std_thread_result = matrix_multiplication(&matrix_a, &matrix_b, MultiplicationMethod::StdThread);
            std_thread_duration_vectors[index].push(std_thread_result.duration);
        }

        println!();

        // XPU implementation
        for (index, &size) in xpu_matrix_sizes.iter().enumerate() {
            println!("XPU - Matrix Size: {}x{}", size, size);

            // Generating two random matrices
            let (matrix_a, matrix_b) = generate_matrices(size, MINIMUM_VALUE, MAXIMUM_VALUE);

            // XPU OpenCL parallel matrix multiplication
            let xpu_result = matrix_multiplication(&matrix_a, &matrix_b, MultiplicationMethod::XpuOpenCL);
            xpu_total_duration_vectors[index].push(xpu_result.duration);
            xpu_kernel_duration_vectors[index].push(xpu_result.kernel_duration);
        }

        println!();
    }

    // Create results file
    let mut file = File::create("results.txt").expect("Failed to create results file!");

    println!();
    print_and_write(&mut file, "--- CPU IMPLEMENTATIONS ---");
    print_and_write(&mut file, "");

    for (index, &size) in cpu_matrix_sizes.iter().enumerate() {
        let sequential_metrics = calculate_metrics(&sequential_duration_vectors[index]);
        let rayon_metrics = calculate_metrics(&rayon_duration_vectors[index]);
        let tokio_metrics = calculate_metrics(&tokio_duration_vectors[index]);
        let std_thread_metrics = calculate_metrics(&std_thread_duration_vectors[index]);

        print_cpu_results(
            &mut file,
            size,
            sequential_metrics,
            rayon_metrics,
            tokio_metrics,
            std_thread_metrics
        );
    }

    println!();
    print_and_write(&mut file, "");
    print_and_write(&mut file, "--- XPU IMPLEMENTATION ---");
    print_and_write(&mut file, "");

    for (index, &size) in xpu_matrix_sizes.iter().enumerate() {
        let total_metrics = calculate_metrics(&xpu_total_duration_vectors[index]);
        let kernel_metrics = calculate_metrics(&xpu_kernel_duration_vectors[index]);
        let last_index = index == (xpu_matrix_sizes.len() - 1);

        print_xpu_results(
            &mut file,
            size,
            total_metrics,
            kernel_metrics,
            last_index
        );
    }

    println!("Results have been written to results.txt!");
}
