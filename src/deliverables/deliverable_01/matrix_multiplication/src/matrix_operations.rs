use ocl::{
	Buffer,
	ProQue,
	flags
};

use once_cell::sync::OnceCell;
use tokio::runtime::Runtime;
use std::time::Instant;
use rayon::prelude::*;
use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub enum MultiplicationMethod {
    Sequential,
    Rayon,
    Tokio,
    StdThread,
    XpuOpenCL
}

pub struct MatrixResult {
    pub result: Vec<Vec<i32>>,
    pub duration: f32,
    pub kernel_duration: f32
}

pub fn generate_matrices(size: i32, minimum_value: i32, maximum_value: i32) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let mut rng = rand::rng();

    let matrix_a: Vec<Vec<i32>> = (0..size)
        .map(|_| (0..size).map(|_| rng.random_range(minimum_value..maximum_value)).collect())
        .collect();

    let matrix_b: Vec<Vec<i32>> = (0..size)
        .map(|_| (0..size).map(|_| rng.random_range(minimum_value..maximum_value)).collect())
        .collect();

    (matrix_a, matrix_b)
}

pub fn matrix_multiplication(
    matrix_a: &Vec<Vec<i32>>,
    matrix_b: &Vec<Vec<i32>>,
    method: MultiplicationMethod
) -> MatrixResult {
    // Common validation logic
    let rows_matrix_a = matrix_a.len();
    let columns_matrix_a = matrix_a[0].len();
    let rows_matrix_b = matrix_b.len();
    let columns_matrix_b = matrix_b[0].len();

    if columns_matrix_a != rows_matrix_b {
        panic!("Matrix dimensions do not match for multiplication!");
    }

    // XPU method returns kernel_duration separately
    if let MultiplicationMethod::XpuOpenCL = method {
        let (result, kernel_duration, total_duration) = xpu_opencl_implementation(
            matrix_a,
            matrix_b,
            rows_matrix_a,
            columns_matrix_a,
            columns_matrix_b
        );
        return MatrixResult {
            result,
            duration: total_duration,
            kernel_duration
        };
    }

    // CPU methods measure total duration only
    let start = Instant::now();

    let result = match method {
        MultiplicationMethod::Sequential =>
            sequential_implementation(matrix_a, matrix_b, rows_matrix_a, columns_matrix_a, columns_matrix_b),
        MultiplicationMethod::Rayon =>
            rayon_implementation(matrix_a, matrix_b, rows_matrix_a, columns_matrix_a, columns_matrix_b),
        MultiplicationMethod::Tokio =>
            tokio_implementation(matrix_a, matrix_b, rows_matrix_a, columns_matrix_a, columns_matrix_b),
        MultiplicationMethod::StdThread =>
            std_thread_implementation(matrix_a, matrix_b, rows_matrix_a, columns_matrix_a, columns_matrix_b),
        MultiplicationMethod::XpuOpenCL =>
            unreachable!()
    };

    let duration = start.elapsed().as_secs_f32();

    MatrixResult {
        result,
        duration,
        kernel_duration: duration
    }
}

fn sequential_implementation(
    matrix_a: &Vec<Vec<i32>>,
    matrix_b: &Vec<Vec<i32>>,
    rows_matrix_a: usize,
    columns_matrix_a: usize,
    columns_matrix_b: usize
) -> Vec<Vec<i32>> {
    let mut result = vec![vec![0; columns_matrix_b]; rows_matrix_a];

    for i in 0..rows_matrix_a {
        for j in 0..columns_matrix_b {
            for k in 0..columns_matrix_a {
                result[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }

    result
}

fn rayon_implementation(
    matrix_a: &Vec<Vec<i32>>,
    matrix_b: &Vec<Vec<i32>>,
    rows_matrix_a: usize,
    columns_matrix_a: usize,
    columns_matrix_b: usize
) -> Vec<Vec<i32>> {
    let mut result = vec![vec![0; columns_matrix_b]; rows_matrix_a];

    result.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in 0..columns_matrix_b {
            for k in 0..columns_matrix_a {
                row[j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    });

    result
}

fn tokio_implementation(
    matrix_a: &Vec<Vec<i32>>,
    matrix_b: &Vec<Vec<i32>>,
    rows_matrix_a: usize,
    columns_matrix_a: usize,
    columns_matrix_b: usize
) -> Vec<Vec<i32>> {
    let mut result = vec![vec![0; columns_matrix_b]; rows_matrix_a];

    Runtime::new().unwrap().block_on(async {
        let mut handles = vec![];

        for i in 0..rows_matrix_a {
            let row_a = matrix_a[i].clone();
            let matrix_b = matrix_b.clone();
            let handle = tokio::spawn(async move {
                let mut row_result = vec![0; columns_matrix_b];
                for j in 0..columns_matrix_b {
                    for k in 0..columns_matrix_a {
                        row_result[j] += row_a[k] * matrix_b[k][j];
                    }
                }
                (i, row_result)
            });
            handles.push(handle);
        }

        for handle in handles {
            let (i, row_result) = handle.await.unwrap();
            result[i] = row_result;
        }
    });

    result
}

fn std_thread_implementation(
    matrix_a: &Vec<Vec<i32>>,
    matrix_b: &Vec<Vec<i32>>,
    rows_matrix_a: usize,
    columns_matrix_a: usize,
    columns_matrix_b: usize
) -> Vec<Vec<i32>> {
    let mut result = vec![vec![0; columns_matrix_b]; rows_matrix_a];
    let mut handles = vec![];

    for i in 0..rows_matrix_a {
        let row_a = matrix_a[i].clone();
        let matrix_b = matrix_b.clone();
        let handle = std::thread::spawn(move || {
            let mut row_result = vec![0; columns_matrix_b];
            for j in 0..columns_matrix_b {
                for k in 0..columns_matrix_a {
                    row_result[j] += row_a[k] * matrix_b[k][j];
                }
            }
            (i, row_result)
        });
        handles.push(handle);
    }

    for handle in handles {
        let (i, row_result) = handle.join().unwrap();
        result[i] = row_result;
    }

    result
}

// XPU OpenCL implementation with tiled optimization
const TILE_SIZE: usize = 16;

static GLOBAL_PROQUE: OnceCell<ProQue> = OnceCell::new();

fn get_proque() -> &'static ProQue {
    GLOBAL_PROQUE.get_or_init(|| {
        let kernel_source = r#"
        #define TS {TS}
        __kernel void matmul_tiled(
            const int N,
            const int P,
            __global const float* A,
            __global const float* B,
            __global float* C,
            __local float* As,
            __local float* Bs
        ) {
            int row = get_global_id(0);
            int col = get_global_id(1);

            float sum = 0.0f;
            int num_tiles = (N + TS - 1) / TS;

            int local_row = get_local_id(0);
            int local_col = get_local_id(1);

            for (int t = 0; t < num_tiles; ++t) {
                int a_col = t * TS + local_col;
                int b_row = t * TS + local_row;

                // Load A tile into local memory
                if (row < get_global_size(0) && a_col < N)
                    As[local_row * TS + local_col] = A[row * N + a_col];
                else
                    As[local_row * TS + local_col] = 0.0f;

                // Load B tile into local memory
                if (b_row < N && col < get_global_size(1))
                    Bs[local_row * TS + local_col] = B[b_row * P + col];
                else
                    Bs[local_row * TS + local_col] = 0.0f;

                barrier(CLK_LOCAL_MEM_FENCE);

                for (int k = 0; k < TS; ++k) {
                    sum += As[local_row * TS + k] * Bs[k * TS + local_col];
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (row < get_global_size(0) && col < get_global_size(1))
                C[row * P + col] = sum;
        }
        "#;

        let source = kernel_source.replace("{TS}", &TILE_SIZE.to_string());

        ProQue::builder()
            .src(source)
            .build()
            .expect("Failed to build ProQue for OpenCL. Check drivers and ICD configuration.")
    })
}

fn flatten_matrix_to_f32(matrix: &Vec<Vec<i32>>) -> Vec<f32> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut flat = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        for j in 0..cols {
            flat.push(matrix[i][j] as f32);
        }
    }

    flat
}

fn unflatten_f32_to_matrix(flat: Vec<f32>, rows: usize, cols: usize) -> Vec<Vec<i32>> {
    let mut matrix = vec![vec![0i32; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            matrix[i][j] = flat[i * cols + j].round() as i32;
        }
    }

    matrix
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    ((value + multiple - 1) / multiple) * multiple
}

fn xpu_opencl_implementation(
    matrix_a: &Vec<Vec<i32>>,
    matrix_b: &Vec<Vec<i32>>,
    rows_matrix_a: usize,
    columns_matrix_a: usize,
    columns_matrix_b: usize
) -> (Vec<Vec<i32>>, f32, f32) {
    let pro_que = get_proque();

    // Convert matrices to flat f32 arrays
    let flat_a = flatten_matrix_to_f32(matrix_a);
    let flat_b = flatten_matrix_to_f32(matrix_b);

    let rows = rows_matrix_a;
    let cols = columns_matrix_b;
    let n = columns_matrix_a;

    // Round up dimensions to tile size for proper alignment
    let global_rows = round_up_to_multiple(rows, TILE_SIZE);
    let global_cols = round_up_to_multiple(cols, TILE_SIZE);

    let queue = pro_que.queue();

    // Start measuring total time
    let start_total = Instant::now();

    // Create buffers and copy data to device
    let buffer_a = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(flat_a.len())
        .copy_host_slice(&flat_a)
        .build()
        .expect("Failed to create buffer A");

    let buffer_b = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(flat_b.len())
        .copy_host_slice(&flat_b)
        .build()
        .expect("Failed to create buffer B");

    let buffer_c = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_WRITE_ONLY)
        .len(global_rows * global_cols)
        .build()
        .expect("Failed to create buffer C");

    // Build kernel with local memory allocation
    let kernel = pro_que.kernel_builder("matmul_tiled")
        .arg(&(n as i32))
        .arg(&(cols as i32))
        .arg(&buffer_a)
        .arg(&buffer_b)
        .arg(&buffer_c)
        .arg_local::<f32>(TILE_SIZE * TILE_SIZE)
        .arg_local::<f32>(TILE_SIZE * TILE_SIZE)
        .build()
        .expect("Failed to build kernel");

    // Warm-up execution to avoid initial compilation overhead
    unsafe {
        kernel.cmd()
            .global_work_size((global_rows, global_cols))
            .local_work_size((TILE_SIZE, TILE_SIZE))
            .enq()
            .expect("Warm-up kernel execution failed");
    }
    queue.finish().expect("Warm-up queue finish failed");

    // Measure kernel execution time only
    let start_kernel = Instant::now();
    unsafe {
        kernel.cmd()
            .global_work_size((global_rows, global_cols))
            .local_work_size((TILE_SIZE, TILE_SIZE))
            .enq()
            .expect("Kernel execution failed");
    }
    queue.finish().expect("Failed to finish queue after kernel execution");
    let kernel_duration = start_kernel.elapsed().as_secs_f32();

    // Read result from device to host
    let mut flat_c = vec![0f32; global_rows * global_cols];
    buffer_c.read(&mut flat_c).enq().expect("Failed to read buffer C");
    queue.finish().expect("Failed to finish queue after reading");

    let total_duration = start_total.elapsed().as_secs_f32();

    // Extract actual result (trim padding)
    let mut trimmed = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            trimmed[r * cols + c] = flat_c[r * global_cols + c];
        }
    }

    let result = unflatten_f32_to_matrix(trimmed, rows, cols);

    (result, kernel_duration, total_duration)
}

pub fn calculate_metrics(durations: &Vec<f32>) -> (f32, f32, f32) {
    let sum: f32 = durations.iter().sum();
    let average = sum / durations.len() as f32;
    let variance: f32 = durations.iter().map(|value| {
        let diff = average - *value;
        diff * diff
    }).sum::<f32>() / durations.len() as f32;
    let standard_deviation = variance.sqrt();

    (sum, average, standard_deviation)
}
