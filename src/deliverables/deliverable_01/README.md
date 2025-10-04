# Deliverable 01: Matrix Multiplication Parallelization

## Overview

This project implements and benchmarks different approaches to matrix multiplication, comparing CPU-based parallelization strategies with GPU acceleration using Intel Iris Xe through OpenCL. The implementation explores the performance characteristics of sequential and parallel algorithms across varying matrix sizes.

## Implementation Details

### Project Structure

The project is organized into three main modules:

- `main.rs`: Orchestrates the execution flow, benchmark loop, and results aggregation
- `matrix_operations.rs`: Contains all matrix multiplication implementations and utility functions
- `printing_informations.rs`: Handles output formatting for both terminal and file output

### Matrix Multiplication Methods

#### CPU Implementations

**Sequential Implementation**
The baseline implementation uses three nested loops following the standard matrix multiplication algorithm: C[i][j] = Σ(A[i][k] × B[k][j]). This serves as the reference point for measuring parallelization speedup.

**Rayon Implementation**
Utilizes Rayon's work-stealing thread pool to parallelize row computations. The `par_iter_mut()` iterator distributes rows across available CPU cores, allowing multiple rows of the result matrix to be computed simultaneously. Each thread processes complete rows independently, minimizing synchronization overhead.

**Tokio Implementation**
Implements task-based parallelism using Tokio's async runtime. Each row is spawned as an independent async task. While Tokio is designed for I/O-bound operations, this implementation demonstrates task scheduling overhead in CPU-bound workloads. The blocking operation `block_on()` waits for all tasks to complete before returning results.

**Standard Thread Implementation**
Uses Rust's `std::thread` to spawn one thread per matrix row. Each thread computes a complete row and returns the index-result pair. This approach demonstrates manual thread management and the associated creation/joining overhead.

#### XPU Implementation (OpenCL)

The GPU implementation uses a tiled matrix multiplication kernel optimized for Intel Iris Xe architecture:

**Tiling Strategy**
The kernel divides matrices into 16×16 tiles (configurable via `TILE_SIZE` constant). This size is chosen to balance local memory usage with compute efficiency on integrated GPUs. Tiling improves cache locality by loading matrix blocks into shared local memory before computation.

**Memory Hierarchy**
- Global memory: Stores input matrices A and B, and output matrix C
- Local memory: Each work group maintains two tile-sized buffers (As and Bs) in fast local memory
- Work groups: Organized as 16×16 blocks matching the tile size

**Kernel Execution Flow**
1. Each work item identifies its global position (row, col) and local position within the tile
2. The kernel iterates through tiles along the shared dimension
3. For each tile: load data into local memory, synchronize work items, compute partial sums, synchronize again
4. After processing all tiles, write the final result to global memory

**Warm-up Execution**
A single warm-up kernel execution is performed before timing to exclude initial compilation and JIT overhead from measurements. This ensures timing accuracy for actual computational performance.

**Timing Measurements**
Two distinct metrics are collected:
- Kernel-only time: Measures pure GPU computation time
- Total time: Includes host-to-device transfers, kernel execution, and device-to-host transfers

### Data Generation and Metrics

**Matrix Generation**
Random integer matrices are generated using values in the range [1, 10). The same random matrices are used across all methods within each repetition to ensure fair comparison.

**Statistical Metrics**
For each matrix size and method, the implementation collects:
- Sum of all execution times across repetitions
- Average execution time
- Standard deviation to measure consistency

**Benchmark Configuration**
- Repetitions: 10 iterations per matrix size
- CPU matrix sizes: 2×2, 4×4, 8×8, 16×16, 32×32, 64×64, 128×128, 256×256, 512×512, 1024×1024
- XPU matrix sizes: Extends to 2048×2048 and 4096×4096

## Performance Analysis

### Small Matrices (2×2 to 16×16)

For small matrices, sequential execution outperforms all parallel approaches. The overhead of thread creation, task scheduling, or GPU data transfers far exceeds the computational work required. At 2×2, sequential execution takes 0.00001 seconds while Rayon takes 0.000236 seconds and XPU total time is 0.000828 seconds. The parallelization overhead dominates in this regime.

### Medium Matrices (32×32 to 128×128)

Parallel CPU implementations begin showing benefits around 32×32 matrices. At 128×128:
- Sequential: 0.103 seconds average
- Rayon: 0.024 seconds average (4.3× speedup)
- Tokio: 0.023 seconds average (4.5× speedup)
- Std::thread: 0.024 seconds average (4.3× speedup)

XPU kernel-only time (0.0001 seconds) is already faster than CPU implementations, but total time (0.00183 seconds) remains higher due to transfer overhead.

### Large Matrices (256×256 to 1024×1024)

The gap widens significantly as matrix size increases. At 1024×1024:
- Sequential: 56.48 seconds average
- Parallel CPU: ~13.5 seconds average (4.2× speedup)
- XPU kernel-only: 0.039 seconds average (1450× speedup vs sequential)
- XPU total: 0.080 seconds average (706× speedup vs sequential)

At this scale, parallel CPU implementations achieve consistent 4× speedup, limited by the CPU core count and memory bandwidth. GPU acceleration becomes dominant even including transfer costs.

### Extra Large Matrices (2048×2048 to 4096×4096)

These sizes exceed practical CPU computation time, demonstrating GPU necessity for large-scale operations:
- 2048×2048 XPU total: 0.860 seconds average
- 4096×4096 XPU total: 7.646 seconds average
- 4096×4096 XPU kernel-only: 3.785 seconds average

Transfer overhead represents approximately 50% of total XPU time at these scales, indicating that optimizations should focus on reducing data movement or batching operations.

### Key Observations

**Thread Overhead Analysis**
For CPU parallelism, the overhead threshold lies between 16×16 and 32×32 matrices. Below this point, thread creation, context switching, and synchronization costs exceed computational savings.

**GPU Transfer Bottleneck**
The difference between kernel-only and total XPU time reveals that memory transfers constitute 50-60% of total GPU execution time. This suggests strategies like:
- Keeping data on GPU across multiple operations
- Using pinned memory for faster transfers
- Batching multiple matrix operations

**Parallel Efficiency**
CPU parallel implementations show near-linear scaling up to the available core count (approximately 4× on a quad-core system), then plateau due to memory bandwidth and cache contention. GPU implementations continue scaling due to massive parallelism (hundreds of execution units).

**Algorithm Complexity**
All implementations maintain O(n³) complexity for n×n matrices. The performance differences stem from constant factors: memory access patterns, parallelism degree, and overhead costs.

## Technical Implementation Notes

### Type Conversions

The OpenCL kernel operates on `f32` (32-bit floating point) values for GPU efficiency, while the application uses `i32` (32-bit integers). Conversion functions flatten 2D matrices to 1D arrays, convert types, and reverse the process after computation. Results are rounded to nearest integers during conversion.

### Memory Alignment

Global work sizes are rounded up to multiples of `TILE_SIZE` to ensure proper alignment for tiled kernel execution. Padding is added when necessary and trimmed from final results to maintain correct dimensions.

### Kernel Compilation Caching

The `ProQue` (kernel program queue) is cached using `OnceCell` to avoid recompiling the OpenCL kernel on every invocation. The kernel source is generated once with the tile size template parameter substituted at initialization.

### Synchronization

OpenCL kernels use barrier synchronization (`barrier(CLK_LOCAL_MEM_FENCE)`) to ensure all work items in a work group complete memory loads before computation begins, and complete computation before proceeding to the next tile. This prevents race conditions in local memory access.

## Building and Running

### Dependencies

- Rust 1.70 or later
- OpenCL runtime (Intel GPU drivers with OpenCL support)
- Cargo dependencies: `once_cell`, `ocl`, `rand`, `rayon`, `tokio`

### Execution

```bash
cd src/deliverables/deliverable_01/matrix_multiplication
cargo run --release
```

The `--release` flag is critical for accurate performance measurements. Debug builds include extensive runtime checks that skew results.

### Output

Results are printed to terminal during execution and written to `results.txt` in the same directory. The output format includes:
- Per-size CPU implementation timings (sequential, Rayon, Tokio, std::thread)
- Per-size XPU implementation timings (total and kernel-only)
- Statistical metrics (sum, average, standard deviation)

## Conclusions

This implementation demonstrates that parallelization strategy must match problem scale. Sequential code dominates for tiny matrices, CPU parallelism works well for medium matrices (providing 4× speedup), and GPU acceleration becomes essential for large matrices (providing 700-1400× speedup).

The XPU implementation reveals that raw computational power is only part of the equation. Memory transfer overhead remains a critical bottleneck, consuming up to 50% of execution time. Future optimizations should focus on minimizing data movement between host and device.

The consistent low standard deviation across measurements indicates stable performance characteristics, validating the statistical reliability of these benchmarks.
