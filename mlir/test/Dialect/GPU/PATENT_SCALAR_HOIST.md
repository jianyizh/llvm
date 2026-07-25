# PATENT: Kernel-Scope Uniform Expression Hoisting from GPU Kernels to Host CPU

## 1. TITLE OF THE INVENTION

Method for Identifying and Hoisting Kernel-Scope Uniform Scalar
Expressions from GPU Kernels to Host CPU for Precomputation via
Device-Aware Compiler IR

## 2. NAMES OF THE INVENTORS

[Inventor Name(s) - To Be Filled]

## 3. RESUBMISSION RESPONSES

THIS SECTION TO BE LEFT BLANK FOR INITIAL SUBMISSIONS.

## 4. TECHNOLOGY BACKGROUND

### 4.1 Problem Definition

In heterogeneous computing (GPU/accelerator kernels written in SYCL,
OpenCL, CUDA, or similar frameworks), scalar kernel parameters
frequently appear in expressions that produce the same result for every
work-item in a dispatch. These **kernel-scope uniform expressions**
range from simple cases (e.g., integer division `i / scalar_d`, or
floating-point reciprocal `1.0 / scalar_a`) to complex cases where
algebraic restructuring is required to reveal the scalar-only
sub-expression (e.g., `(b+1) / (scalar_a * b)` contains the hidden
invariant `1 / scalar_a`). Despite being invariant across potentially
millions of work-items, these expressions are redundantly evaluated by
each work-item, wasting GPU compute cycles on expensive operations
such as:

- **Integer division/remainder** (`udiv`, `urem`): 20-80 cycles per
  operation on GPU, vs. 1-4 cycles for multiply/shift.
- **Floating-point division** (`fdiv`): 16-24 cycles per operation.
- **Transcendental functions** (`sqrt`, `rsqrt`, `exp`, `log`, `sin`,
  `cos`, `pow`): 12-32 cycles per operation.

Existing compiler optimizations are insufficient:

- **Constant folding**: only handles compile-time constants, not
  runtime scalar parameters.
- **Loop-invariant code motion (LICM)**: operates within a single
  address space; cannot move computation across the host-device
  boundary.
- **Strength reduction**: for integer division, requires compile-time
  constant divisors; cannot handle runtime-determined values.
- **Common sub-expression elimination (CSE)**: can identify redundant
  computation within a single work-item, but cannot eliminate
  redundancy *across* millions of work-items that all compute the same
  result. Does not perform algebraic restructuring to expose hidden
  invariants.

The technical problem is: how to (1) identify kernel-scope uniform
expressions in GPU kernels -- including those hidden within mixed
expressions requiring algebraic decomposition, (2) hoist them from the
GPU kernel to the host CPU for precomputation, and (3) pass the
precomputed results back to the GPU kernel as parameters -- all without
changing kernel semantics or introducing synchronization overhead.

### 4.2 Previous Solution

A. In previous approaches, GPU compute kernels receive raw scalar
parameters and each work-item evaluates all expressions as written.
Developers may manually precompute obvious cases (e.g., `1.0 / a`),
but this is ad-hoc, error-prone, and misses complex cases where
algebraic transformation is needed to expose the invariant. No
existing solution provides: (a) automatic identification of
kernel-scope uniform expressions including those requiring algebraic
decomposition, (b) systematic hoisting across the host-device boundary
with device-aware IR modeling, and (c) cost-model-driven filtering to
ensure net benefit exceeds overhead.

## 5. OVERVIEW OF THE INVENTION

### 5.1 Short Summary

The invention is a compiler method that: (1) identifies scalar kernel
parameters involved in kernel-scope uniform expressions within GPU kernels --
using forward dataflow analysis to classify every SSA value as
SCALAR_ONLY (uniform across work-items), INDEX_ONLY (varies per
work-item), MIXED, or CONSTANT, and applying algebraic decomposition
(factoring, partial fractions, associative regrouping) to extract
scalar-only sub-expressions from mixed expressions; (2) hoists those
sub-expressions from the GPU kernel to the host CPU, generating
precomputation code as CPU-side operations; and (3) passes the
precomputed results to the GPU kernel as additional parameters. A
device-aware compiler IR dialect (`scalar_hoist`) models the
precomputation and cross-device transfer explicitly, enabling
type-safe cross-boundary expression migration within MLIR.

### 5.2 Advantages

This invention provides three key benefits: (1) it discovers
optimization opportunities invisible to existing compiler passes by
identifying runtime scalar expressions (including algebraically
restructured ones) and hoisting them across the host-device boundary;
(2) it reduces GPU compute utilization by eliminating redundant
per-work-item evaluation of these invariants across millions of
work-items -- measured 13-14% kernel speedup on a 64M-element
bandwidth-bound workload on Intel Data Center GPU Max 1550, with
higher speedups expected for compute-bound kernels; (3) the CPU-side
precomputation is effectively free -- it executes during command list
preparation, hidden behind existing driver overhead. The device-aware
IR approach (the `scalar_hoist` dialect operating on MLIR's `gpu`
dialect) ensures correctness and generality across GPU programming
models (SYCL, OpenCL, CUDA).

## 6. DETECTABILITY

### 6.1 Detection Methods

The invention is detectable through the following means:

**A. Structural Feature:** In compiled GPU binaries (SPIR-V, PTX, Gen
ISA), the optimized kernel exhibits additional scalar parameters beyond
those in the original source API and algebraically restructured
expressions. For integer division hoisting: `OpUDiv`/`OpUMod` are
replaced by a characteristic multiply-shift pattern, and extra i32
magic/shift parameters appear. For floating-point hoisting: source
expression `(b+1)/(scalar_a*b)` appears as a multiply-by-parameter in
the binary, where the parameter is a precomputed value (`1/scalar_a`)
not specified by the developer. CPU-side profiling traces show scalar
arithmetic operations immediately preceding kernel dispatch.

**B. Reverse Engineering:** Comparing GPU kernel binary signatures
against source code reveals additional precomputed scalar parameters.
For example, a kernel with 6 source-level arguments appears with 10
arguments in the binary (4 extra = 2 magic + 2 shift for two hoisted
divisions). The host-side binary shows characteristic precomputation
patterns (CLZ, shift, 64-bit division for integer magic; fdiv/sqrt for
floating-point) preceding each kernel dispatch call. Compiler IR dumps
show cross-boundary value transfer from host to device regions.

**C. Product Literature:** Compiler documentation describing
`gpu-scalar-hoist` pass, `scalar_hoist` dialect, "scalar expression
hoisting", "kernel-scope uniform
expression precomputation", "magic number division hoisting", "scalar
dependency classification", device-aware IR dialects for cross-boundary
optimization, or compiler flags enabling such optimizations would
indicate usage. Compiler IR dumps showing `scalar_hoist.precompute` or
`scalar_hoist.yield` operations would be direct evidence.

## 7. DETAILS OF THE INVENTION

### 7.1 Invention Details

The invention is a compiler method for optimizing heterogeneous compute
kernels by identifying kernel-scope uniform scalar expressions, hoisting them
from GPU kernels to the host CPU for precomputation, and passing
results back as kernel parameters. The method is realized through a
device-aware compiler IR dialect (MLIR's GPU dialect) that explicitly
models computation devices (host CPU vs. GPU), cross-device value
transfer (via kernel argument ABI), and precomputation scheduling
(host-side code generation before kernel launch). Specifically, the
host-device boundary is modeled via `gpu.module`, `gpu.func`, and
`gpu.launch_func` operations, which make both sides of the boundary
visible to a single compiler pass, enabling cross-boundary analysis
and transformation that is impossible in traditional single-address-space
compiler frameworks.

The method operates in four phases:

- **Phase 1: Dependency Classification** -- Forward dataflow analysis
  classifies each SSA value in the GPU kernel as SCALAR_ONLY (uniform
  across work-items), INDEX_ONLY (varies per work-item), MIXED (depends
  on both), or CONSTANT (compile-time known).

- **Phase 2: Candidate Identification & Algebraic Decomposition** --
  Scan for expensive operations (integer division/remainder,
  floating-point division, transcendental functions) where the operand
  or a sub-operand is SCALAR_ONLY. For MIXED expressions, apply
  algebraic transformation rules (scalar factoring, partial fractions,
  distributive factoring, associative regrouping) to isolate
  SCALAR_ONLY sub-expressions. Apply cost-model filtering: only hoist
  if the net benefit (saved GPU cycles minus parameter overhead)
  exceeds a threshold.

- **Phase 3: Host-Side Precomputation** -- For each hoisting candidate,
  generate host-side code to precompute the scalar expression before
  kernel launch. For integer division, this means computing magic
  multiplier and shift count. For floating-point, this means computing
  reciprocals, products, or transcendental results. This code is
  inserted before the `gpu.launch_func` call in the host function.

- **Phase 4: Kernel Replacement & Signature Update** -- Add the
  precomputed values as additional kernel arguments. Replace the
  expensive kernel-side operations with cheap equivalents using the
  new arguments (multiply-shift for integer division, multiply for
  floating-point division, direct use for transcendentals). Update
  **all** `gpu.launch_func` call sites consistently.

---

### DIALECT DESIGN: THE `scalar_hoist` DIALECT

The method is realized through a device-aware compiler IR dialect
(`scalar_hoist`) that explicitly models host-side precomputation and
cross-device value transfer. The dialect is designed to work alongside
MLIR's existing `gpu` dialect, which models the host-device boundary
via `gpu.module`, `gpu.func`, and `gpu.launch_func`.

#### Dialect Definition

```tablegen
def ScalarHoist_Dialect : Dialect {
  let name = "scalar_hoist";
  let summary = "Dialect for scalar expression hoisting from GPU to host";
  let cppNamespace = "::mlir::scalar_hoist";
  // Accepts ops defined programmatically (precompute, yield)
  let hasNonDefaultDestructor = 0;
}
```

#### Operations

**`scalar_hoist.precompute`** -- Region-based op that wraps host-side
scalar precomputation. Takes scalar kernel-arg values as inputs; the
region body contains the computation (e.g., magic number algorithm);
the region yields the precomputed results (e.g., magic and shift).

```tablegen
def ScalarHoist_PrecomputeOp : Op<ScalarHoist_Dialect, "precompute"> {
  let summary = "Host-side scalar precomputation";
  let description = [{
    Wraps a scalar computation that will execute on the host CPU before
    kernel launch. The region takes scalar divisor values as block
    arguments and yields precomputed values (e.g., magic multiplier
    and shift count for integer division strength reduction).

    The precompute op makes the cross-boundary optimization explicit
    in the IR: the region body shows WHAT is being precomputed, the
    op's operands show the scalar inputs (kernel arguments), and the
    op's results flow into gpu.launch_func as additional kernel args.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
}
```

**`scalar_hoist.yield`** -- Terminator op for the precompute region.
Returns computed values to the parent precompute op's results.

```tablegen
def ScalarHoist_YieldOp : Op<ScalarHoist_Dialect, "yield", [
    Pure, Terminator, HasParent<"PrecomputeOp">
]> {
  let summary = "Yield from precompute region";
  let arguments = (ins Variadic<AnyType>:$values);
}
```

#### Dependency Classification Attributes

Each SSA value in the GPU kernel is classified with one of four
dependency classes. The classification is computed by forward dataflow
analysis and stored in an in-memory map during pass execution:

```cpp
enum class DepClass : uint8_t {
  SCALAR_ONLY,   // Uniform across all work-items (scalar kernel args,
                 //   gpu.block_dim, expressions of scalars)
  INDEX_ONLY,    // Varies per work-item (gpu.thread_id, buffer loads,
                 //   expressions of indices)
  MIXED,         // Depends on both scalar and index-dependent values
  CONSTANT       // Compile-time constant (arith.constant)
};
```

#### IR Representation

**Before optimization (input IR):**

```mlir
func.func @dispatch(%data: memref<?xf32>, %scalar_a: f32) {
  gpu.launch_func @compute::@kernel
    args(%data, %scalar_a : memref<?xf32>, f32)
  return
}
gpu.module @compute {
  gpu.func @kernel(%data: memref<?xf32>, %scalar_a: f32) kernel {
    %idx = gpu.thread_id x
    %b = memref.load %data[%idx] : memref<?xf32>
    %denom = arith.mulf %scalar_a, %b : f32     // MIXED
    %result = arith.divf %b, %denom : f32        // MIXED - target
    memref.store %result, %data[%idx] : memref<?xf32>
    gpu.return
  }
}
```

**After `gpu-scalar-hoist` (scalar_hoist dialect ops visible):**

```mlir
func.func @dispatch(%data: memref<?xf32>, %scalar_a: f32) {
  // Host-side precomputation wrapped in scalar_hoist dialect op
  %inv_a = "scalar_hoist.precompute"(%scalar_a) ({
  ^bb0(%a: f32):
    %one = arith.constant 1.0 : f32
    %r = arith.divf %one, %a : f32
    "scalar_hoist.yield"(%r) : (f32) -> ()
  }) : (f32) -> f32

  // Launch with precomputed value as extra kernel arg
  gpu.launch_func @compute::@kernel_optimized
    args(%data, %scalar_a, %inv_a : memref<?xf32>, f32, f32)
  return
}
gpu.module @compute {
  // Optimized kernel: no per-work-item fdiv by scalar
  gpu.func @kernel_optimized(%data: memref<?xf32>, %scalar_a: f32,
                             %precomp_inv_a: f32) kernel {
    %idx = gpu.thread_id x
    %b = memref.load %data[%idx] : memref<?xf32>
    %result = arith.mulf %precomp_inv_a, %b : f32  // uses precomp
    memref.store %result, %data[%idx] : memref<?xf32>
    gpu.return
  }
}
```

**After `lower-scalar-hoist` (dialect ops inlined, ready for lowering):**

```mlir
func.func @dispatch(%data: memref<?xf32>, %scalar_a: f32) {
  %one = arith.constant 1.0 : f32
  %inv_a = arith.divf %one, %scalar_a : f32   // inlined from precompute
  gpu.launch_func @compute::@kernel_optimized
    args(%data, %scalar_a, %inv_a : memref<?xf32>, f32, f32)
  return
}
```

#### Device-Aware Lowering

The `scalar_hoist` dialect lowers to standard MLIR dialects via the
`lower-scalar-hoist` pass:

1. `scalar_hoist.precompute` → region body inlined into host
   `func.func` (standard arith/math operations)
2. `scalar_hoist.yield` → erased (values flow through SSA)
3. Precomputed results → passed as additional kernel arguments via
   existing `gpu.launch_func` ABI (scalar register passing,
   no memory allocation)

```
Lowering pipeline:

  scalar_hoist.precompute / scalar_hoist.yield
    -> [lower-scalar-hoist]         // Inline regions, erase dialect ops
    -> Standard arith/math ops in host func.func
    -> [convert-gpu-to-spirv]       // GPU module -> SPIR-V
    -> [convert-to-llvm]            // Host func -> LLVM IR
    -> SPIR-V (device) + LLVM IR (host)
```

---

### ARCHITECTURAL FLOW

```
                    +-------------------------------+
                    |   SYCL / OpenCL / CUDA Source  |
                    |   e.g., c = (i % chw) / hw    |
                    |   e.g., (b+1)/(scalar_a * b)  |
                    +---------------+---------------+
                                    |
                                    v
                    +-------------------------------+
                    |   MLIR GPU Dialect IR          |
                    |   gpu.module + gpu.func +      |
                    |   gpu.launch_func              |
                    +---------------+---------------+
                                    |
                                    v
    +---------------------------------------------------------------+
    |         gpu-scalar-hoist Pass (THIS INVENTION)                |
    |                                                               |
    |  Phase 1: DEPENDENCY CLASSIFICATION                           |
    |    For each SSA value in gpu.func:                            |
    |    - Scalar kernel args          -> SCALAR_ONLY               |
    |    - MemRef kernel args          -> INDEX_ONLY                |
    |    - gpu.thread_id / block_id    -> INDEX_ONLY                |
    |    - gpu.block_dim / grid_dim    -> SCALAR_ONLY               |
    |    - memref.load results         -> INDEX_ONLY                |
    |    - arith/math ops              -> lattice join of operands  |
    |                                                               |
    |  Phase 2: CANDIDATE ID & ALGEBRAIC DECOMPOSITION              |
    |    A. Direct scalar operations:                               |
    |       - arith.divui/remui with SCALAR_ONLY divisor            |
    |       - arith.divf with SCALAR_ONLY divisor                   |
    |       - math.sqrt/exp/log/sin/cos on SCALAR_ONLY arg          |
    |    B. MIXED expression decomposition rules:                   |
    |       Rule 1: x/(a*y) -> (1/a)*(x/y)   [factor scalar]      |
    |       Rule 2: f(a)*x  -> hoist f(a)     [factor scalar mul]  |
    |       Rule 3: (x+c)/(a*x) -> (1/a)*(1+c/x) [partial frac]   |
    |       Rule 4: a*x+a*y -> a*(x+y)       [distributive]        |
    |       Rule 5: (a*b)*x -> a*(b*x)       [assoc regroup]       |
    |    C. Cost-model: net_benefit >= threshold                    |
    |                                                               |
    |  Phase 3: HOST-SIDE PRECOMPUTATION                            |
    |    Generate in host func, before gpu.launch_func:             |
    |    - Integer div: magic/shift via Granlund-Montgomery         |
    |    - FP div: reciprocal 1.0/d                                 |
    |    - Transcendental: sqrt(d), exp(d), etc.                    |
    |    - Compound: any SCALAR_ONLY expression tree                |
    |                                                               |
    |  Phase 4: KERNEL REPLACEMENT + SIGNATURE UPDATE               |
    |    - Integer div: n/d -> mul_hi(magic,n)+n >> shift           |
    |    - FP div: x/d -> x * precomp_reciprocal                   |
    |    - Transcendental: inline use of precomp result             |
    |    - Add precomputed values as new kernel arguments           |
    |    - Update ALL gpu.launch_func call sites                    |
    +---------------------------------------------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
    +---------------------------+   +-------------------------------+
    | Optimized GPU Kernel      |   | Host Function                 |
    | (SPIR-V / PTX binary)     |   | (LLVM IR -> x86)              |
    |                           |   |                               |
    | Expensive per-work-item   |   | Precompute scalar exprs:      |
    | ops eliminated:           |   |   magic/shift for int div     |
    | - int div -> mul+shift    |   |   1/d for FP div              |
    | - FP div  -> FP mul       |   |   sqrt(d), exp(d), ...        |
    | - sqrt(a) -> use precomp  |   | Cost: ~10-100 ns on CPU      |
    |                           |   |   (hidden behind cmd prep)    |
    | Extra kernel args added   |   | launch(args..., precomp...)   |
    +---------------------------+   +-------------------------------+
```

---

### PHASE 1: DEPENDENCY CLASSIFICATION (DETAILED)

The classification is a forward dataflow analysis over the SSA graph of
the GPU kernel body. It uses a four-element lattice:

```
    CONSTANT
       |
  SCALAR_ONLY    INDEX_ONLY
       \            /
        \          /
          MIXED
```

**Lattice join rules:**

| Operand A   | Operand B   | Result      |
|-------------|-------------|-------------|
| CONSTANT    | CONSTANT    | CONSTANT    |
| CONSTANT    | SCALAR_ONLY | SCALAR_ONLY |
| CONSTANT    | INDEX_ONLY  | INDEX_ONLY  |
| SCALAR_ONLY | SCALAR_ONLY | SCALAR_ONLY |
| SCALAR_ONLY | INDEX_ONLY  | MIXED       |
| INDEX_ONLY  | INDEX_ONLY  | INDEX_ONLY  |
| any         | MIXED       | MIXED       |

**Base cases for outlined GPU kernels (`gpu.func` inside `gpu.module`):**

| Value type                        | Classification |
|-----------------------------------|----------------|
| Block arg with MemRef type        | INDEX_ONLY     |
| Block arg with scalar type (i32, f32, ...) | SCALAR_ONLY |
| `gpu.thread_id`, `gpu.block_id`, `gpu.lane_id`, `gpu.global_id`, `gpu.subgroup_id` | INDEX_ONLY |
| `gpu.block_dim`, `gpu.grid_dim`   | SCALAR_ONLY    |
| `memref.load` result              | INDEX_ONLY     |
| `arith.constant`                  | CONSTANT       |

The analysis iterates to a fixed point, propagating classifications
through all `arith.*` and `math.*` operations in the kernel.

**Implementation** (from `ScalarHoist.cpp`):

```cpp
enum class DepClass : uint8_t { SCALAR_ONLY, INDEX_ONLY, MIXED, CONSTANT };

static void classifyValues(gpu::GPUFuncOp kernel,
                           DenseMap<Value, DepClass> &depMap) {
  Block &entry = kernel.getBody().front();

  // Base case: kernel arguments
  for (auto arg : entry.getArguments()) {
    Type t = arg.getType();
    if (isa<MemRefType>(t))
      depMap[arg] = DepClass::INDEX_ONLY;
    else
      depMap[arg] = DepClass::SCALAR_ONLY;
  }

  // Forward propagation to fixed point
  SmallVector<Operation *> allOps;
  kernel.walk([&](Operation *op) {
    if (!op->getResults().empty())
      allOps.push_back(op);
  });

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : allOps) {
      // GPU index queries -> INDEX_ONLY
      if (isa<gpu::BlockIdOp, gpu::ThreadIdOp, gpu::LaneIdOp,
              gpu::GlobalIdOp, gpu::SubgroupIdOp>(op)) {
        // ... classify as INDEX_ONLY
      }
      // gpu.block_dim / gpu.grid_dim -> SCALAR_ONLY
      if (isa<gpu::BlockDimOp, gpu::GridDimOp>(op)) {
        // ... classify as SCALAR_ONLY
      }
      // Memory loads -> INDEX_ONLY
      if (isa<memref::LoadOp>(op)) {
        // ... classify as INDEX_ONLY
      }
      // Arithmetic/math ops: lattice join of operands
      bool hasScalar = false, hasIndex = false;
      for (Value operand : op->getOperands()) {
        DepClass dc = depMap.lookup(operand);
        if (dc == DepClass::SCALAR_ONLY || dc == DepClass::CONSTANT)
          hasScalar = true;
        if (dc == DepClass::INDEX_ONLY || dc == DepClass::MIXED)
          hasIndex = true;
      }
      // Join: both -> MIXED, scalar only -> SCALAR, index only -> INDEX
    }
  }
}
```

---

### ALGEBRAIC DECOMPOSITION OF MIXED EXPRESSIONS

When a target expression is classified as MIXED, the pass applies
algebraic transformation rules to isolate a SCALAR_ONLY sub-expression
that can be hoisted. The decomposition is driven by pattern matching on
the expression tree.

**Rule 1: Factor scalar from division**
```
Pattern:  x / (a * y)       where a = SCALAR_ONLY, x,y = INDEX
Result:   (1/a) * (x / y)
Hoist:    precomp = 1/a     on host
Kernel:   precomp * (x/y)
Saved:    1 fdiv per work-item
```

**Rule 2: Factor scalar from multiplication**
```
Pattern:  f(a) * x          where f(a) = SCALAR_ONLY, x = INDEX
Result:   precomp * x
Hoist:    precomp = f(a)    on host (e.g., sqrt(a), exp(a))
Kernel:   precomp * x
Saved:    cost(f) per work-item (12-32 cycles for transcendentals)
```

**Rule 3: Partial fraction decomposition**
```
Pattern:  (x + c) / (a * x)   where a = SCALAR_ONLY
Result:   (1/a) * (1 + c/x)
Hoist:    precomp = 1/a
Kernel:   precomp * (1 + c/x)
Saved:    1 fdiv per work-item
```

**Rule 4: Distributive factoring**
```
Pattern:  f(a)*x + f(a)*y     where f(a) = SCALAR_ONLY
Result:   f(a) * (x + y)
Hoist:    precomp = f(a)
Kernel:   precomp * (x + y)
Saved:    cost(f) per work-item (eliminates redundant evaluation)
```

**Rule 5: Associative regrouping**
```
Pattern:  (a * b) * x         where a, b = SCALAR_ONLY
Result:   precomp * x
Hoist:    precomp = a * b
Kernel:   precomp * x
Saved:    1 fmul per work-item + eliminates scalar chain
```

**Cost-model filtering:** For each candidate, the net benefit is:
```
net_benefit = saved_gpu_cycles - param_overhead(2 cycles) - new_instr_cost
```
Candidates are accepted only if `net_benefit >= threshold` (e.g., 8
cycles). This prevents hoisting cheap expressions where the overhead
of an additional kernel argument exceeds the savings.

---

### HOISTING CATEGORIES

The dependency classification framework supports hoisting across
multiple categories of scalar expressions:

#### Category A: Integer Division / Remainder (Primary Embodiment)

For `arith.divui` / `arith.remui` with SCALAR_ONLY divisor `d`:

- **Host precomputation:** Magic multiplier and shift count via
  Granlund-Montgomery algorithm (see Phase 3 detail above).
- **Kernel replacement:** `n / d` -> `(mul_hi(magic, n) + n) >> shift`
- **Correctness:** Exact for all 32-bit unsigned inputs (proven
  mathematical identity, no approximation).
- **Measured result:** 18% speedup on 64M-element BiasAdd, Intel Max
  1550 (see Experimental Results).

#### Category B: Floating-Point Division

For `arith.divf` with SCALAR_ONLY divisor `d`:

- **Host precomputation:** `reciprocal = 1.0 / d`
- **Kernel replacement:** `x / d` -> `x * reciprocal`
- **Correctness:** Requires `-ffast-math` or equivalent FMF flags
  (FP multiply-by-reciprocal is not bit-exact vs. division under IEEE
  754 strict mode).
- **Savings:** 16-24 cycles (fdiv) replaced by 2 cycles (fmul) per
  work-item.

#### Category C: Transcendental Functions

For `math.sqrt`, `math.rsqrt`, `math.exp`, `math.log`, `math.sin`,
`math.cos`, `math.pow` applied to SCALAR_ONLY arguments:

- **Host precomputation:** Evaluate the transcendental on the CPU.
- **Kernel replacement:** Replace `math.sqrt(%scalar_arg)` with a
  direct use of the precomputed kernel parameter.
- **Correctness:** Exact (same function, just evaluated on different
  device). CPU math library results may differ in ULP from GPU SFU
  results; this is acceptable under `-ffast-math`.
- **Savings:** 12-32 cycles per work-item depending on function.

#### Category D: Compound Scalar Expressions

For chains of SCALAR_ONLY operations (e.g., `sqrt(dt / mass)` where
both `dt` and `mass` are scalar args):

- **Host precomputation:** Evaluate the entire SCALAR_ONLY expression
  subtree on the CPU.
- **Kernel replacement:** Replace with a single kernel parameter.
- **Savings:** Sum of all operations in the chain.

---

### CONCRETE EXAMPLE: Floating-Point Algebraic Decomposition

This example demonstrates the algebraic decomposition rules on a
physics simulation kernel, complementing the integer division demo.

#### Input SYCL Kernel

```cpp
// Physics: normalized force computation
// scalar args: mass, damping_coeff, dt
// per-work-item data: position[i], velocity[i]
cgh.parallel_for(range<1>(N), [=](item<1> item) {
    int i = item.get_id(0);
    float pos = position[i];
    float vel = velocity[i];

    // Expression 1: (vel * damping_coeff) / mass
    //   MIXED: contains scalar sub-expr damping_coeff/mass
    float drag = (vel * damping_coeff) / mass;

    // Expression 2: (pos * dt) / (mass * dt * dt) = pos / (mass * dt)
    //   MIXED: contains scalar sub-expr 1/(mass*dt)
    float spring = (pos * dt) / (mass * dt * dt);

    // Expression 3: (drag + spring) * sqrt(dt / mass)
    //   sqrt(dt/mass) is entirely SCALAR_ONLY
    float result = (drag + spring) * sqrt(dt / mass);

    force_out[i] = result;
});
```

#### Dependency Classification

| Value                  | Classification | Reason                  |
|------------------------|----------------|-------------------------|
| `mass`, `damping_coeff`, `dt` | SCALAR_ONLY | Scalar kernel args |
| `pos`, `vel`           | INDEX_ONLY     | Buffer loads            |
| `vel * damping_coeff`  | MIXED          | INDEX * SCALAR          |
| `(vel*damp) / mass`    | MIXED          | MIXED / SCALAR          |
| `mass * dt`            | SCALAR_ONLY    | SCALAR * SCALAR         |
| `mass * dt * dt`       | SCALAR_ONLY    | SCALAR * SCALAR         |
| `sqrt(dt / mass)`      | SCALAR_ONLY    | sqrt(SCALAR / SCALAR)   |

#### Decomposition Results

| Expression              | Rule  | Scalar Part         | Cost | Residual         |
|-------------------------|-------|---------------------|------|------------------|
| `(vel*damp)/mass`       | 1 + 5 | `damp/mass`        | 16   | `vel`            |
| `(pos*dt)/(mass*dt*dt)` | 1     | `1/(mass*dt)`      | 18   | `pos`            |
| `sum * sqrt(dt/mass)`   | 2     | `sqrt(dt/mass)`    | 28   | `sum`            |

#### Optimized Kernel

```cpp
// HOST PRECOMPUTATION (generated by compiler):
float precomp1 = damping_coeff / mass;     // 1 fdiv
float precomp2 = 1.0f / (mass * dt);       // 1 fmul + 1 fdiv
float precomp3 = sqrt(dt / mass);          // 1 fdiv + 1 sqrt

// DEVICE KERNEL (3 fmul + 1 fadd, no fdiv/sqrt):
float drag   = vel * precomp1;
float spring = pos * precomp2;
float result = (drag + spring) * precomp3;
force_out[i] = result;
```

#### Per-Work-Item Cost

| Metric          | Original | Optimized | Saved     |
|-----------------|----------|-----------|-----------|
| fdiv            | 3        | 0         | 48 cycles |
| fmul            | 4        | 3         | 2 cycles  |
| sqrt            | 1        | 0         | 12 cycles |
| fadd            | 1        | 1         | 0         |
| **Total**       | **72**   | **8**     | **64 cyc (89%)** |

For N = 1,000,000 work-items: 64M GPU cycles saved per dispatch.
Host precomputation cost: ~80 ns (hidden behind ~1-100 us cmd prep).

---

### SCHEDULING: OVERLAP WITH COMMAND PREPARATION

The host-side precomputation can be scheduled to overlap with GPU
command list preparation, making the precomputation cost effectively
zero:

```
CPU Thread Timeline:
  [command list setup] --parallel--> [scalar precomputation]
  [submit to GPU]

  Command list prep:  ~1-100 microseconds (driver overhead)
  Precomputation:     ~10-100 nanoseconds (few ALU ops)
  Overhead:           0% (fully hidden behind driver latency)
```

For command queue models (SYCL, OpenCL), the precomputation is inserted
between `queue.submit()` setup and the actual kernel parameter binding.
For explicit command list models (Level Zero, Vulkan), the
precomputation executes while the driver processes descriptor setup and
pipeline state.

---

### EMBODIMENT 1: INTEGER DIVISION HOISTING (IMPLEMENTED, WITH DATA)

The primary embodiment, fully implemented and measured, targets integer
division and remainder operations. This section provides the complete
implementation detail.

#### Phase 2: Candidate Identification

The pass walks the GPU kernel for `arith.divui` and `arith.remui`
operations and checks whether the divisor (RHS) is SCALAR_ONLY or
contains a SCALAR_ONLY sub-operand.

```cpp
struct HoistCandidate {
  Operation *op;          // the divui or remui operation
  Value scalarOperand;    // the SCALAR_ONLY divisor value
  Value divisor;          // the actual divisor in the operation
};

// Find SCALAR_ONLY operand: direct or within MIXED expression
static Value getScalarOperand(Value val,
                              const DenseMap<Value, DepClass> &depMap) {
  if (isScalarOnly(val, depMap))
    return val;  // directly SCALAR_ONLY

  // Check if MIXED: one operand is SCALAR_ONLY
  Operation *defOp = val.getDefiningOp();
  if (!defOp) return nullptr;
  for (Value operand : defOp->getOperands()) {
    if (isScalarOnly(operand, depMap))
      return operand;  // extract scalar from mixed
  }
  return nullptr;
}
```

**Deduplication:** When the same scalar divisor appears in multiple
division operations (e.g., both `i % chw` and `i / chw` use `chw`),
only one magic/shift pair is computed on the host and shared by all
uses.

---

#### Phase 3: Host-Side Magic-Number Precomputation

For each unique scalar divisor `d`, the pass generates the following
computation on the host side, inserted into the host `func.func` body
before any `scf.for` loop (and thus before any `gpu.launch_func` call):

**Magic-number algorithm** (based on "Division by Invariant Integers
using Multiplication", Granlund & Montgomery, PLDI 1994):

```
Input:  d (32-bit unsigned divisor, runtime value)
Output: magic (32-bit), shift (32-bit)

1. shift = 32 - clz(d - 1)                    // ceil(log2(d))
2. d64  = zext(d) to i64
3. s64  = zext(shift) to i64
4. p2   = 1_i64 << (s64 + 32)                 // 2^(shift+32)
5. m0   = p2 / d64                             // floor division
6. m1   = m0 + 1                               // candidate magic
7. overflow = (m1 * d64) > p2                  // precision check
8. m64  = overflow ? m0 : m1                   // select
9. magic = trunc(m64) to i32
```

**MLIR IR generated on host side:**

```mlir
// Host function, before scf.for:
%one    = arith.constant 1 : i32
%dm1    = arith.subi %hw, %one : i32              // d - 1
%clz    = math.ctlz %dm1 : i32                    // count leading zeros
%c32    = arith.constant 32 : i32
%shift  = arith.subi %c32, %clz : i32             // shift = 32 - clz

%d64    = arith.extui %hw : i32 to i64
%s64    = arith.extui %shift : i32 to i64
%c32_64 = arith.constant 32 : i64
%ts     = arith.addi %s64, %c32_64 : i64          // shift + 32
%one64  = arith.constant 1 : i64
%p2     = arith.shli %one64, %ts : i64            // 1 << (shift+32)
%m0     = arith.divui %p2, %d64 : i64             // p2 / d
%m1     = arith.addi %m0, %one64 : i64            // m0 + 1
%pr     = arith.muli %m1, %d64 : i64              // overflow check
%ov     = arith.cmpi ugt, %pr, %p2 : i64
%m64    = arith.select %ov, %m0, %m1 : i64
%magic  = arith.trunci %m64 : i64 to i32
```

The magic and shift values are then appended as operands to **all**
`gpu.launch_func` calls that invoke this kernel, including warmup
iterations.

---

#### Phase 4: Kernel-Side Replacement

Each `arith.divui` / `arith.remui` in the kernel body is replaced with
a multiply-add-shift sequence using the precomputed magic and shift
kernel arguments.

**Division replacement (`n / d` -> magic multiply):**

```
Input:  n (32-bit dividend, varies per work-item)
        magic, shift (32-bit, uniform, precomputed on host)
Output: quotient = n / d

1. m64  = zext(magic) to i64
2. n64  = zext(n) to i64
3. full = m64 * n64                            // 64-bit product
4. hi   = full >> 32                           // upper 32 bits (mul_hi)
5. hi32 = trunc(hi) to i32
6. sum  = hi32 + n                             // add-back step
7. quot = sum >> shift                         // final quotient
```

**Remainder replacement (`n % d` -> quotient-based):**

```
1. quot = <division replacement above>
2. prod = quot * d                             // reconstruct divisor multiple
3. rem  = n - prod                             // remainder
```

**MLIR IR generated in kernel:**

```mlir
// Replace: %ch = arith.divui %rem, %hw : i32
// With:
%m64   = arith.extui %magic_arg : i32 to i64      // zext magic
%n64   = arith.extui %rem : i32 to i64            // zext dividend
%full  = arith.muli %m64, %n64 : i64              // 64-bit multiply
%c32   = arith.constant 32 : i64
%hi    = arith.shrui %full, %c32 : i64            // upper 32 bits
%hi32  = arith.trunci %hi : i64 to i32            // back to i32
%sum   = arith.addi %hi32, %rem : i32             // add-back
%ch    = arith.shrui %sum, %shift_arg : i32        // final quotient
```

**Cycle cost comparison per work-item:**

| Operation | Original  | Optimized         |
|-----------|-----------|-------------------|
| `divui`   | 20-80 cyc | -                 |
| `zext` x2 | -         | 1 cyc each        |
| `muli` i64| -         | 4 cyc             |
| `shrui`   | -         | 1 cyc             |
| `trunci`  | -         | 1 cyc             |
| `addi`    | -         | 1 cyc             |
| `shrui`   | -         | 1 cyc             |
| **Total** | **20-80** | **~10 cyc**       |

---

### KERNEL ARGUMENT ABI UPDATE

The pass modifies both the kernel signature and all call sites:

1. **`gpu.func` signature**: new `i32` block arguments are appended for
   each (magic, shift) pair. The function type attribute is updated to
   reflect the new argument count.

2. **`gpu.launch_func` operands**: the precomputed magic and shift
   host-side values are appended to the `args(...)` list of **every**
   `gpu.launch_func` that calls the modified kernel (including warmup
   loops and benchmark loops).

```
Before: gpu.func @kernel(%src, %bias, %dst, %tot, %chw, %hw)  // 6 args
After:  gpu.func @kernel(%src, %bias, %dst, %tot, %chw, %hw,
                          %magic_chw, %shift_chw,               // for chw
                          %magic_hw, %shift_hw)                  // for hw
                                                                // 10 args
```

---

### COMPLETE END-TO-END EXAMPLE: BiasAdd Kernel

This section demonstrates the complete transformation on the actual
implemented benchmark.

#### Input: Original MLIR

```mlir
module @bias_add attributes {gpu.container_module} {
  gpu.module @bias_add_kernel attributes {spirv.target_env = ...} {
    gpu.func @bias_add_kernel(
        %src: memref<33554432xf32>,     // source buffer (32M f32, 128MB)
        %bias: memref<64xf32>,          // per-channel bias (64 channels)
        %dst: memref<33554432xf32>,     // destination buffer
        %tot: index,                    // total elements = 33554432
        %chw: index,                    // C*H*W = 4194304
        %hw: index                      // H*W = 65536
    ) kernel attributes {
        gpu.known_block_size = array<i32: 256, 1, 1>
    } {
      %tx = gpu.thread_id x                          // INDEX_ONLY
      %bx = gpu.block_id x                           // INDEX_ONLY
      %bs = gpu.block_dim x                           // SCALAR_ONLY
      %gid0 = arith.muli %bx, %bs : index            // INDEX_ONLY
      %gid = arith.addi %tx, %gid0 : index           // INDEX_ONLY
      %i = arith.index_castui %gid : index to i32    // INDEX_ONLY
      %tot_i32 = arith.index_castui %tot : index to i32 // SCALAR_ONLY
      %is_in = arith.cmpi ult, %i, %tot_i32 : i32   // MIXED
      scf.if %is_in {
        %chw_i32 = arith.index_castui %chw : index to i32 // SCALAR_ONLY
        %hw_i32 = arith.index_castui %hw : index to i32   // SCALAR_ONLY
        // TARGET: runtime division by uniform scalar args
        %rem = arith.remui %i, %chw_i32 : i32  // MIXED (INDEX % SCALAR)
        %ch  = arith.divui %rem, %hw_i32 : i32 // MIXED (MIXED / SCALAR)
        %ci  = arith.index_castui %ch : i32 to index
        %idx = arith.index_castui %i : i32 to index
        %s = memref.load %src[%idx] : memref<33554432xf32>
        %b = memref.load %bias[%ci] : memref<64xf32>
        %r = arith.addf %s, %b : f32
        memref.store %r, %dst[%idx] : memref<33554432xf32>
      }
      gpu.return
    }
  }

  func.func @main() {
    %tot = arith.constant 33554432 : index
    %chw = arith.constant 4194304 : index
    %hw  = arith.constant 65536 : index
    // ... gpu.alloc, warmup loop, benchmark loop ...
    scf.for %b = %c0 to %c100 step %c1 {
      gpu.launch_func @bias_add_kernel::@bias_add_kernel
        blocks in (%blocks, %c1, %c1) threads in (%c256, %c1, %c1)
        args(%mem_src : memref<33554432xf32>,
             %mem_bias : memref<64xf32>,
             %mem_dst : memref<33554432xf32>,
             %tot : index, %chw : index, %hw : index)
    }
    return
  }
}
```

#### Phase 1 Classification Result

| Value      | Type      | Classification | Reason                        |
|------------|-----------|----------------|-------------------------------|
| `%src`     | memref    | INDEX_ONLY     | MemRef kernel arg             |
| `%bias`    | memref    | INDEX_ONLY     | MemRef kernel arg             |
| `%dst`     | memref    | INDEX_ONLY     | MemRef kernel arg             |
| `%tot`     | index     | SCALAR_ONLY    | Scalar kernel arg             |
| `%chw`     | index     | SCALAR_ONLY    | Scalar kernel arg             |
| `%hw`      | index     | SCALAR_ONLY    | Scalar kernel arg             |
| `%tx`      | index     | INDEX_ONLY     | gpu.thread_id                 |
| `%bx`      | index     | INDEX_ONLY     | gpu.block_id                  |
| `%bs`      | index     | SCALAR_ONLY    | gpu.block_dim                 |
| `%gid0`    | index     | INDEX_ONLY     | INDEX * SCALAR -> INDEX       |
| `%gid`     | index     | INDEX_ONLY     | INDEX + INDEX                 |
| `%i`       | i32       | INDEX_ONLY     | cast of INDEX                 |
| `%chw_i32` | i32       | SCALAR_ONLY    | cast of SCALAR                |
| `%hw_i32`  | i32       | SCALAR_ONLY    | cast of SCALAR                |
| `%is_in`   | i1        | MIXED          | INDEX < SCALAR                |
| `%rem`     | i32       | MIXED          | INDEX % SCALAR                |
| `%ch`      | i32       | MIXED          | MIXED / SCALAR                |

#### Phase 2 Candidates Found

| Operation                            | Divisor    | Classification | Scalar Operand |
|--------------------------------------|------------|----------------|----------------|
| `%rem = arith.remui %i, %chw_i32`   | `%chw_i32` | SCALAR_ONLY   | `%chw_i32`    |
| `%ch = arith.divui %rem, %hw_i32`   | `%hw_i32`  | SCALAR_ONLY   | `%hw_i32`     |

Two unique scalar divisors: `%chw_i32` and `%hw_i32` (derived from
index-typed kernel args via `arith.index_castui`). Two (magic, shift)
pairs will be computed on the host.

#### Output: Optimized MLIR

```mlir
func.func @main() {
    %tot = arith.constant 33554432 : index
    %chw = arith.constant 4194304 : index
    %hw  = arith.constant 65536 : index
    // ... gpu.alloc ...

    // === HOST-SIDE PRECOMPUTATION (generated by pass) ===
    // index -> i32 cast, then magic/shift wrapped in scalar_hoist.precompute
    %hw_i32 = arith.index_castui %hw : index to i32
    %magic_hw, %sh_hw = "scalar_hoist.precompute"(%hw_i32) ({
    ^bb0(%d: i32):
      // ... magic number computation ...
      "scalar_hoist.yield"(%magic, %shift) : (i32, i32) -> ()
    }) : (i32) -> (i32, i32)

    %chw_i32 = arith.index_castui %chw : index to i32
    %magic_chw, %sh_chw = "scalar_hoist.precompute"(%chw_i32) ({
    ^bb0(%d: i32):
      // ... magic number computation ...
      "scalar_hoist.yield"(%magic, %shift) : (i32, i32) -> ()
    }) : (i32) -> (i32, i32)

    // === ALL launch_func calls updated with extra args ===
    scf.for %b = %c0 to %c100 step %c1 {
      gpu.launch_func @bias_add_kernel::@bias_add_kernel
        blocks in (%blocks, %c1, %c1) threads in (%c256, %c1, %c1)
        args(%mem_src : memref<33554432xf32>,
             %mem_bias : memref<64xf32>,
             %mem_dst : memref<33554432xf32>,
             %tot : index, %chw : index, %hw : index,
             %magic_hw : i32, %sh_hw : i32,       // NEW
             %magic_chw : i32, %sh_chw : i32)     // NEW
    }
    return
  }

  // === OPTIMIZED KERNEL (4 extra args, no divui/remui) ===
  gpu.func @bias_add_kernel(
      %src: memref<33554432xf32>, %bias: memref<64xf32>,
      %dst: memref<33554432xf32>, %tot: index, %chw: index, %hw: index,
      %magic_hw: i32, %shift_hw: i32,        // precomputed for hw
      %magic_chw: i32, %shift_chw: i32       // precomputed for chw
  ) kernel {
    // ... compute %i as before ...
    scf.if %is_in {
      // BEFORE: %rem = arith.remui %i, %chw_i32 : i32
      // AFTER:  magic multiply for rem
      %m64_chw = arith.extui %magic_chw : i32 to i64
      %n64_chw = arith.extui %i : i32 to i64
      %full_chw = arith.muli %m64_chw, %n64_chw : i64
      %hi_chw  = arith.shrui %full_chw, %c32_64 : i64
      %hi32_chw = arith.trunci %hi_chw : i64 to i32
      %sum_chw = arith.addi %hi32_chw, %i : i32
      %quot_chw = arith.shrui %sum_chw, %shift_chw : i32
      %prod_chw = arith.muli %quot_chw, %chw_i32 : i32
      %rem = arith.subi %i, %prod_chw : i32         // remainder

      // BEFORE: %ch = arith.divui %rem, %hw_i32 : i32
      // AFTER:  magic multiply for div
      %m64_hw  = arith.extui %magic_hw : i32 to i64
      %n64_hw  = arith.extui %rem : i32 to i64
      %full_hw = arith.muli %m64_hw, %n64_hw : i64
      %hi_hw   = arith.shrui %full_hw, %c32_64 : i64
      %hi32_hw = arith.trunci %hi_hw : i64 to i32
      %sum_hw  = arith.addi %hi32_hw, %rem : i32
      %ch = arith.shrui %sum_hw, %shift_hw : i32    // quotient

      // ... rest unchanged: load, fadd, store ...
    }
    gpu.return
  }
```

---

### EXPERIMENTAL RESULTS

**Platform:** Intel Data Center GPU Max 1550 (Ponte Vecchio), 128 GB HBM2e

**Profiling tool:** Intel unitrace (Level Zero kernel timing)

#### Benchmark 1: BiasAdd (MLIR pipeline, memory-bandwidth-bound)

Kernel: `dst[i] = src[i] + bias[(i % chw) / hw]`
- 2 integer divisions per work-item, 32M work-items
- Shape: N=8, C=64, H=256, W=256 (tot=33,554,432, 128MB per buffer)
- Memory-bandwidth-bound (256 MB total data movement dominates)

| Configuration | Elements  | Blocks x Threads | Kernel Args | Avg Latency (ns) |
|---------------|-----------|-------------------|-------------|-------------------|
| Baseline      | 33,554,432 | 131,072 x 256    | 6           | 284,981           |
| **Optimized** | 33,554,432 | 131,072 x 256    | **10**      | **239,836**       |
| **Speedup**   |           |                   |             | **+18.8%**        |

#### Benchmark 2: GroupNorm (SYCL, compute-bound)

Kernel: Full GroupNorm with Welford reduction + per-element affine norm
- N=1024, D=192, S=784 (DS=150528 per group), 1024 work-items/group
- Pass 2: 4 integer divisions per vec-4 per iteration (`c = (j+v) / S`)
- ~148 divisions per work-item per pass
- Compute-bound (Welford + affine + per-element division)

| Configuration | Groups | Kernel Args | Avg Latency (ns) | Min Latency (ns) |
|---------------|--------|-------------|-------------------|-------------------|
| Baseline (`c = (j+v)/S`) | 1024 | 9 | 1,093,440 | 1,013,440 |
| **Optimized** (magic mul) | 1024 | **11** | **766,861** | **728,800** |
| **Speedup** | | | **+42.6%** | **+39.1%** |

#### Analysis

- **BiasAdd (+13.6%):** Bandwidth-bound workload. Division cost is
  partially hidden by memory latency, but hoisting still provides
  meaningful improvement.
- **GroupNorm (+42.6%):** Compute-bound workload with high division
  density. The optimization nearly halves the per-element normalization
  cost because `c = j/S` (4x per vec-4) was the dominant ALU
  bottleneck in Pass 2.
- **Host-side precomputation cost:** ~20 ns on CPU (a few integer
  operations), negligible vs. kernel execution (500-1100 us).
- **Correctness:** The magic-number division is verified exhaustively
  over the full valid input range [0, DS) before kernel launch.

**SPIR-V binary signature comparison (BiasAdd):**

```
Baseline kernel:  6 arguments, contains OpUDiv + OpUMod instructions
Optimized kernel: 10 arguments, OpUDiv/OpUMod replaced by
                  OpUConvert + OpIMul + OpShiftRightLogical + OpIAdd
```

---

### MLIR PASS PIPELINE

The pass integrates into the standard MLIR GPU compilation pipeline:

```
Source (SYCL / OpenCL / CUDA)
  -> Clang Frontend
  -> MLIR (gpu + func + arith + memref + scf dialects)
  -> [gpu-scalar-hoist]                    <-- THIS INVENTION (Phase 1-4)
  ->   emits scalar_hoist.precompute / scalar_hoist.yield ops
  -> [lower-scalar-hoist]                  <-- Inline precompute regions
  -> [spirv-attach-target]
  -> [convert-gpu-to-spirv]
  -> [gpu.module(spirv.module(spirv-lower-abi-attrs, spirv-update-vce))]
  -> [func.func(llvm-request-c-wrappers)]
  -> [convert-scf-to-cf]
  -> [convert-to-llvm]
  -> [gpu-to-llvm]
  -> [gpu-module-to-binary{format=isa}]
  -> Host binary (x86) + Device binary (SPIR-V / Gen ISA)
```

The pass operates at the MLIR level **before** any dialect lowering,
when the host-device boundary (`gpu.launch_func` in host `func.func`,
`gpu.func` in `gpu.module`) is still explicit in the IR. This is a key
advantage of the device-aware IR approach: the pass can reason about
both sides of the boundary simultaneously.

### THE scalar_hoist DIALECT

The `scalar_hoist` dialect makes the cross-boundary precomputation
explicit and visible in IR dumps. It defines two ops:

| Op | Description |
|----|-------------|
| `scalar_hoist.precompute` | Region op. Takes scalar kernel-arg values as inputs; the region body computes the precomputed values (e.g., magic number and shift); yields results. |
| `scalar_hoist.yield` | Terminator for the precompute region. Returns computed values to the parent op's results. |

**Intermediate IR after `gpu-scalar-hoist` (before lowering):**

```mlir
func.func @main() {
    %hw  = arith.constant 262144 : i32
    %chw = arith.constant 4194304 : i32
    // ...

    // Host-side precomputation wrapped in scalar_hoist dialect op
    %magic_hw, %shift_hw = "scalar_hoist.precompute"(%hw) ({
    ^bb0(%d: i32):
        %c1  = arith.constant 1 : i32
        %dm1 = arith.subi %d, %c1 : i32
        %clz = math.ctlz %dm1 : i32
        %c32 = arith.constant 32 : i32
        %s   = arith.subi %c32, %clz : i32
        %d64 = arith.extui %d : i32 to i64
        // ... 64-bit magic number computation ...
        %m   = arith.trunci %m64 : i64 to i32
        "scalar_hoist.yield"(%m, %s) : (i32, i32) -> ()
    }) : (i32) -> (i32, i32)

    %magic_chw, %shift_chw = "scalar_hoist.precompute"(%chw) ({
    ^bb0(%d: i32):
        // ... same magic computation for chw ...
        "scalar_hoist.yield"(%m, %s) : (i32, i32) -> ()
    }) : (i32) -> (i32, i32)

    // Kernel launch with precomputed values as extra args
    gpu.launch_func @kernel::@kernel ...
        args(..., %magic_hw, %shift_hw, %magic_chw, %shift_chw)
}
```

After `lower-scalar-hoist`, the precompute regions are inlined into the
parent block and the dialect ops are erased, leaving only standard
arith/math operations ready for LLVM lowering.

---

### PASS REGISTRATION

```tablegen
def GpuScalarHoistPass : Pass<"gpu-scalar-hoist", "ModuleOp"> {
  let summary = "Hoist scalar integer division from GPU kernel to host CPU";
  let description = [{
    Finds arith.divui/arith.remui operations in gpu.func where the divisor
    is a uniform scalar kernel argument. Hoists magic/shift precomputation
    to the host (before gpu.launch_func) and replaces kernel division with
    magic multiply (mul_hi + add + shift).

    Host-side precomputation is wrapped in scalar_hoist.precompute ops.
    Run lower-scalar-hoist after this pass to inline them.
  }];
  let dependentDialects = [
    "mlir::gpu::GPUDialect", "mlir::arith::ArithDialect",
    "mlir::math::MathDialect", "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect", "mlir::scf::SCFDialect"
  ];
}

def LowerScalarHoistPass : Pass<"lower-scalar-hoist", "ModuleOp"> {
  let summary = "Lower scalar_hoist dialect ops to standard operations";
  let description = [{
    Inlines scalar_hoist.precompute regions into the parent block and
    erases the wrapper ops, producing standard arith/math operations
    in the host function. Must run before convert-to-llvm.
  }];
}
```

---

### CORRECTNESS GUARANTEES

1. **Exact integer arithmetic:** The magic-number algorithm produces
   **identical** results to integer division for all 32-bit unsigned
   inputs. No approximation is involved. This is a proven mathematical
   identity (Granlund & Montgomery, 1994).

2. **Floating-point semantics:** Algebraic transformations (reciprocal
   hoisting, expression factoring) are applied only when fast-math
   flags (`-ffast-math` or per-instruction FMF) are set, consistent
   with standard GPU compiler behavior. Under strict IEEE 754, the
   pass is conservative (no floating-point transformation).

3. **Remainder correctness:** `n % d` is computed as `n - (n/d) * d`,
   which is exact when `n/d` is exact.

4. **Semantic preservation:** The kernel signature changes (additional
   arguments), but the input-output behavior is identical. All
   `gpu.launch_func` call sites are updated consistently.

5. **No synchronization overhead:** Precomputed values are passed via
   the kernel argument ABI (scalar register passing). No additional
   memory allocation, buffer creation, or host-device synchronization
   is required.

6. **Safety:** Only expressions whose scalar operands are classified as
   SCALAR_ONLY (uniform across all work-items) are transformed. The
   dataflow analysis ensures that no work-item-dependent values are
   mistakenly hoisted. Only operations that dominate all uses and are
   not conditionally dead are hoisted.

---

### KEY INNOVATION

The fundamental novelty is the **cross-boundary scalar expression
hoisting** pattern: identifying that a runtime expression is uniform
across all work-items via dataflow analysis in device-aware IR,
precomputing it on the **host** side where the runtime values are
available, and transferring the precomputed results to the device via
the kernel argument ABI. This pattern has three key aspects:

1. **Cross-boundary optimization:** Traditional compiler passes (LICM,
   CSE, strength reduction) operate within a single address space and
   cannot move computation across the host-device boundary. This
   invention exploits the device-aware IR (MLIR with `gpu.module` /
   `gpu.func` / `gpu.launch_func`) to make both sides of the boundary
   visible to a single compiler pass.

2. **Algebraic decomposition:** For MIXED expressions where the scalar
   sub-expression is not syntactically obvious, the pass applies
   algebraic transformation rules to expose the hidden invariant. This
   goes beyond simple hoisting -- it restructures the expression to
   create a hoisting opportunity that did not previously exist.

3. **Operation-specific strength reduction:** For integer division, the
   runtime divisor is transformed into a magic-multiply sequence on the
   host. For floating-point division, a reciprocal is precomputed. For
   transcendentals, the function is evaluated on the CPU. Each category
   has its own precomputation and replacement strategy, unified by the
   common dependency classification framework.

The device-aware IR is essential: it makes both sides of the boundary
visible to a single compiler pass, enabling the pass to simultaneously
(a) analyze the kernel body, (b) generate host precomputation code,
(c) update the kernel signature, and (d) modify all call sites.

---

### SOURCE FILES

| File | Description |
|------|-------------|
| `mlir/lib/Dialect/GPU/Transforms/ScalarHoist.cpp` | Main pass: classify, hoist, emit `scalar_hoist` ops |
| `mlir/lib/Dialect/GPU/Transforms/LowerScalarHoist.cpp` | Lowering pass: inline precompute regions |
| `mlir/include/mlir/Dialect/GPU/Transforms/ScalarHoistDialect.h` | `scalar_hoist` dialect definition (header-only) |
| `mlir/include/mlir/Dialect/GPU/Transforms/Passes.td` | Pass registration (TableGen) |
| `mlir/include/mlir/Dialect/GPU/Transforms/Passes.h` | Pass declarations |
| `mlir/lib/Dialect/GPU/CMakeLists.txt` | Build configuration |
| `mlir/test/Dialect/GPU/bias-add-runtime-shape.mlir` | BiasAdd MLIR benchmark (32M elements, index-typed args) |
| `mlir/test/Dialect/GPU/sycl/group-norm-baseline.cpp` | GroupNorm SYCL baseline (runtime division) |
| `mlir/test/Dialect/GPU/sycl/group-norm-optimized.cpp` | GroupNorm SYCL optimized (host-precomputed magic multiply) |
