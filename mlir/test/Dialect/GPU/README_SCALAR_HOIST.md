# MLIR GPU Scalar Hoisting — Build & Run Guide

## 1. Overview

This guide documents how to build a full MLIR pipeline capable of running
`gpu.launch` kernels on Intel GPU via the Level Zero runtime, using the
intel/llvm `sycl` branch.

The pipeline enables:
- Writing GPU kernels in MLIR `gpu.func` / `gpu.launch_func` dialect
- The `scalar_hoist` dialect for explicit host-side precomputation
- Lowering through SPIR-V to Intel GPU ISA
- JIT execution on Intel GPU via `mlir-runner` + Level Zero

## 2. Prerequisites

- Intel GPU with Level Zero driver installed
- `libhwloc-dev` (for hwloc)
- `opencl-headers` (for OpenCL headers)
- `libze-dev` (Level Zero headers, usually from Intel GPU driver package)

```bash
sudo apt-get install -y libhwloc-dev opencl-headers
# Level Zero headers should already be at /usr/include/level_zero/ from GPU driver
```

## 3. Source Code

- **Repository**: `https://github.com/jianyizh/llvm.git`
- **Branch**: `scalar-hoist`

```bash
git clone https://github.com/jianyizh/llvm.git
cd llvm
git checkout scalar-hoist
```

## 4. Build

```bash
cd /home2/jianyizh/llvm
mkdir build_imex

cmake -G Ninja -B build_imex -S llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;SPIRV" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_LEVELZERO_RUNNER=1

# Build the required tools and runtime libraries
ninja -C build_imex mlir-opt mlir-translate mlir-runner \
  mlir_levelzero_runtime mlir_runner_utils
```

Key CMake options:
| Option | Purpose |
|--------|---------|
| `LLVM_ENABLE_PROJECTS=mlir` | Build MLIR (no clang needed) |
| `LLVM_TARGETS_TO_BUILD=X86;SPIRV` | x86 host + SPIR-V GPU target |
| `MLIR_ENABLE_LEVELZERO_RUNNER=1` | Build Level Zero runtime for JIT GPU execution |

## 5. Verify GPU Access

```bash
# Check GPU is visible via Level Zero
source /opt/intel/oneapi/2026.0/oneapi-vars.sh --force
sycl-ls 2>/dev/null | grep level_zero
```

Expected output:
```
[level_zero:gpu][level_zero:0] Intel(R) Data Center GPU Max 1550 ...
```

## 6. BiasAdd Benchmark — 32M Elements (128MB)

### Source File

`mlir/test/Dialect/GPU/bias-add-runtime-shape.mlir`

- Shape: `N=8, C=64, H=256, W=256 => tot=33554432 (32M, 128MB f32)`
- `chw = C*H*W = 4194304, hw = H*W = 65536`
- Grid: `131072 blocks × 256 threads`
- Each work-item: `arith.remui + arith.divui + memref.load + arith.addf + memref.store`

### Run Baseline (no optimization)

```bash
BUILD=/home2/jianyizh/llvm/build_imex

$BUILD/bin/mlir-opt \
  mlir/test/Dialect/GPU/bias-add-runtime-shape.mlir \
  -pass-pipeline='builtin.module(
    spirv-attach-target{ver=v1.0 caps=Addresses,Int64,Kernel},
    convert-gpu-to-spirv{use-64bit-index=true},
    gpu.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce)),
    func.func(llvm-request-c-wrappers),
    convert-scf-to-cf,
    convert-to-llvm,
    gpu-to-llvm{use-bare-pointers-for-kernels=true},
    gpu-module-to-binary{format=isa},
    expand-strided-metadata,
    lower-affine,
    reconcile-unrealized-casts
  )' -o /tmp/ba_base.mlir

LD_LIBRARY_PATH=$BUILD/lib unitrace -d $BUILD/bin/mlir-runner \
  --shared-libs=$BUILD/lib/libmlir_levelzero_runtime.so \
  --shared-libs=$BUILD/lib/libmlir_runner_utils.so \
  --entry-point-result=void /tmp/ba_base.mlir
```

### Run Optimized (with scalar_hoist dialect)

```bash
$BUILD/bin/mlir-opt \
  mlir/test/Dialect/GPU/bias-add-runtime-shape.mlir \
  -pass-pipeline='builtin.module(
    gpu-scalar-hoist,
    lower-scalar-hoist,
    spirv-attach-target{ver=v1.0 caps=Addresses,Int64,Kernel},
    convert-gpu-to-spirv{use-64bit-index=true},
    gpu.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce)),
    func.func(llvm-request-c-wrappers),
    convert-scf-to-cf,
    convert-to-llvm,
    gpu-to-llvm{use-bare-pointers-for-kernels=true},
    gpu-module-to-binary{format=isa},
    expand-strided-metadata,
    lower-affine,
    reconcile-unrealized-casts
  )' -o /tmp/ba_opt.mlir

LD_LIBRARY_PATH=$BUILD/lib unitrace -d $BUILD/bin/mlir-runner \
  --shared-libs=$BUILD/lib/libmlir_levelzero_runtime.so \
  --shared-libs=$BUILD/lib/libmlir_runner_utils.so \
  --entry-point-result=void /tmp/ba_opt.mlir
```

### Inspect Intermediate IR (dialect ops visible)

```bash
$BUILD/bin/mlir-opt \
  mlir/test/Dialect/GPU/bias-add-runtime-shape.mlir \
  -pass-pipeline='builtin.module(gpu-scalar-hoist)'
```

This shows `scalar_hoist.precompute` and `scalar_hoist.yield` ops in the
host function wrapping the magic-number computation.

### Benchmark Results

**Platform:** Intel Data Center GPU Max 1550 (Ponte Vecchio), 128 GB HBM2e
**Profiling:** Intel unitrace (Level Zero kernel timing)

#### BiasAdd (MLIR, 32M elements, 128MB f32)

| Configuration | Elements | Kernel Args | Avg Latency (ns) | Speedup |
|---------------|----------|-------------|-------------------|---------|
| Baseline      | 33,554,432 | 6         | 284,981           | —       |
| **Optimized** | 33,554,432 | **10**    | **239,836**       | **+18.8%** |

#### GroupNorm (SYCL, N=1024 D=192 S=784, Welford + affine norm)

| Configuration | Groups | Kernel Args | Avg Latency (ns) | Speedup |
|---------------|--------|-------------|-------------------|---------|
| Baseline (`c = (j+v) / S`) | 1024 | 9 | 1,093,440 | — |
| **Optimized** (magic mul)   | 1024 | **11** | **766,861** | **+42.6%** |

GroupNorm is compute-bound (Welford reduction + per-element affine norm
with 4 divisions per vec-4 per iteration), so the division hoisting has
a much larger impact than the bandwidth-bound BiasAdd.

## 7. GroupNorm SYCL Benchmark

### Source Files

| File | Description |
|------|-------------|
| `mlir/test/Dialect/GPU/sycl/group-norm-baseline.cpp` | Baseline: `c = (j+v) / S` (runtime division) |
| `mlir/test/Dialect/GPU/sycl/group-norm-optimized.cpp` | Optimized: host-precomputed magic multiply |

### Build & Run

```bash
source /opt/intel/oneapi/2026.0/oneapi-vars.sh --force

# Build
icpx -fsycl -O2 -o gn_baseline  mlir/test/Dialect/GPU/sycl/group-norm-baseline.cpp
icpx -fsycl -O2 -o gn_optimized mlir/test/Dialect/GPU/sycl/group-norm-optimized.cpp

# Run with unitrace profiling
unitrace -d ./gn_baseline  --N 1024 --D 192 --S 784 --G 1 --wg 1024 --iters 30
unitrace -d ./gn_optimized --N 1024 --D 192 --S 784 --G 1 --wg 1024 --iters 30
```

### Key Difference (Pass 2 only)

```cpp
// BASELINE: genuine runtime division per element
int32_t c = (j + v) / S;   // ~30 GPU cycles per division

// OPTIMIZED: host-precomputed magic multiply (zero-cost on host)
uint32_t q = (uint32_t)(((uint64_t)n * s_magic) >> 32);
uint32_t c = s_add ? (((n - q) >> 1) + q) >> s_shift : (q >> s_shift);
// ~6 GPU cycles (mulhi + shift)
```

The magic constants (`s_magic`, `s_shift`, `s_add`) are computed once on
the CPU from the runtime value of S, verified exhaustively over [0, DS),
and passed to the kernel as 3 additional scalar arguments.

## 7. Pass Pipeline (with scalar_hoist dialect)

```
Input MLIR (gpu.func + gpu.launch_func)
  │
  ├─ gpu-scalar-hoist          ← Phase 1: dependency classification
  │                               Phase 2: find divui/remui by SCALAR_ONLY divisors
  │                               Phase 3: wrap magic/shift in scalar_hoist.precompute
  │                               Phase 4: replace kernel div with magic multiply
  │
  ├─ lower-scalar-hoist        ← Inline precompute regions, erase dialect ops
  │
  ├─ spirv-attach-target       ← Add SPIR-V target to gpu.module
  ├─ convert-gpu-to-spirv      ← Lower gpu ops → SPIR-V dialect
  │    └─ gpu.module:
  │         ├─ spirv-lower-abi-attrs
  │         └─ spirv-update-vce
  ├─ func.func:
  │    └─ llvm-request-c-wrappers
  ├─ convert-scf-to-cf
  ├─ convert-to-llvm
  ├─ gpu-to-llvm
  ├─ gpu-module-to-binary{format=isa}
  ├─ expand-strided-metadata
  ├─ lower-affine
  └─ reconcile-unrealized-casts
         │
         ▼
  LLVM dialect (with gpu.binary embedded)
         │
    mlir-runner + Level Zero → GPU execution
```

## 8. scalar_hoist Dialect

The `scalar_hoist` dialect wraps host-side precomputation in explicit
IR ops, making the cross-boundary optimization visible and verifiable.

### Ops

| Op | Description |
|----|-------------|
| `scalar_hoist.precompute` | Region op wrapping host-side computation. Takes scalar inputs, yields precomputed values (magic, shift). |
| `scalar_hoist.yield` | Terminator for precompute region. |

### Intermediate IR Example

After `gpu-scalar-hoist`, before `lower-scalar-hoist`:

```mlir
func.func @main() {
    %hw = arith.constant 262144 : i32
    // ...
    %magic, %shift = "scalar_hoist.precompute"(%hw) ({
    ^bb0(%d: i32):
        %dm1  = arith.subi %d, %c1 : i32
        %clz  = math.ctlz %dm1 : i32
        %s    = arith.subi %c32, %clz : i32
        // ... 64-bit magic number computation ...
        %m    = arith.trunci %m64 : i64 to i32
        "scalar_hoist.yield"(%m, %s) : (i32, i32) -> ()
    }) : (i32) -> (i32, i32)

    gpu.launch_func @kernel::@kernel ...
        args(..., %magic : i32, %shift : i32)
}
```

After `lower-scalar-hoist`, the precompute region is inlined and the
dialect ops are erased, leaving only standard arith/math ops.

## 9. Source Files

| File | Description |
|------|-------------|
| `mlir/lib/Dialect/GPU/Transforms/ScalarHoist.cpp` | Main pass: classify + hoist + replace |
| `mlir/lib/Dialect/GPU/Transforms/LowerScalarHoist.cpp` | Lowering pass: inline precompute regions |
| `mlir/include/mlir/Dialect/GPU/Transforms/ScalarHoistDialect.h` | Dialect definition (header-only) |
| `mlir/include/mlir/Dialect/GPU/Transforms/Passes.td` | Pass registration |
| `mlir/include/mlir/Dialect/GPU/Transforms/Passes.h` | Pass declarations |
| `mlir/test/Dialect/GPU/bias-add-runtime-shape.mlir` | BiasAdd MLIR benchmark (32M elements, 128MB) |
| `mlir/test/Dialect/GPU/PATENT_SCALAR_HOIST.md` | Patent document |
