// BiasAdd GPU benchmark (index-typed args): dst[i] = src[i] + bias[c], c = (i % chw) / hw
// Shape: N=8, C=64, H=256, W=256 => tot=33554432 (32M, 128MB f32)
// chw = C*H*W = 4194304, hw = H*W = 65536
// 256 threads x 131072 blocks, each thread: index_castui + remui + divui + load + fadd + store
// Kernel args use index type (runtime shape variant)

module @bias_add attributes {gpu.container_module} {
  gpu.module @bias_add_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @bias_add_kernel(%src: memref<33554432xf32>, %bias: memref<64xf32>, %dst: memref<33554432xf32>,
                              %tot: index, %chw: index, %hw: index) kernel
    attributes {gpu.known_block_size = array<i32: 256, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %tx = gpu.thread_id x
      %bx = gpu.block_id x
      %bs = gpu.block_dim x
      %gid0 = arith.muli %bx, %bs : index
      %gid = arith.addi %tx, %gid0 : index
      %i = arith.index_castui %gid : index to i32
      %tot_i32 = arith.index_castui %tot : index to i32
      %is_in = arith.cmpi ult, %i, %tot_i32 : i32
      scf.if %is_in {
        %chw_i32 = arith.index_castui %chw : index to i32
        %hw_i32 = arith.index_castui %hw : index to i32
        // === TARGET: runtime division by uniform scalar args ===
        %rem = arith.remui %i, %chw_i32 : i32
        %ch = arith.divui %rem, %hw_i32 : i32
        %ci = arith.index_castui %ch : i32 to index
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
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %blocks = arith.constant 131072 : index

    %mem_src = gpu.alloc host_shared () : memref<33554432xf32>
    %mem_dst = gpu.alloc host_shared () : memref<33554432xf32>
    %mem_bias = gpu.alloc host_shared () : memref<64xf32>

    %v1 = arith.constant 1.0 : f32
    memref.store %v1, %mem_bias[%c0] : memref<64xf32>

    %tot = arith.constant 33554432 : index
    %chw = arith.constant 4194304 : index
    %hw = arith.constant 65536 : index

    %c5 = arith.constant 5 : index
    scf.for %w = %c0 to %c5 step %c1 {
      gpu.launch_func @bias_add_kernel::@bias_add_kernel blocks in (%blocks, %c1, %c1) threads in (%c256, %c1, %c1)
        args(%mem_src : memref<33554432xf32>, %mem_bias : memref<64xf32>, %mem_dst : memref<33554432xf32>,
             %tot : index, %chw : index, %hw : index)
    }

    %c100 = arith.constant 100 : index
    scf.for %b = %c0 to %c100 step %c1 {
      gpu.launch_func @bias_add_kernel::@bias_add_kernel blocks in (%blocks, %c1, %c1) threads in (%c256, %c1, %c1)
        args(%mem_src : memref<33554432xf32>, %mem_bias : memref<64xf32>, %mem_dst : memref<33554432xf32>,
             %tot : index, %chw : index, %hw : index)
    }

    return
  }
}
