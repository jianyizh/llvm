// GroupNorm SYCL Benchmark — BASELINE (runtime integer division)
//
// This is the baseline version where Pass 2 uses genuine runtime integer
// division (c = (j+v) / S) for per-element channel indexing.
//
// Build:
//   icpx -fsycl -O2 -o gn_baseline group-norm-baseline.cpp
//
// Run:
//   unitrace -d ./gn_baseline --N 1024 --D 192 --S 784 --G 1 --wg 1024 --iters 30
//
// The key runtime division is:
//   int32_t c = (j + v) / S;     // S is a runtime kernel argument
// This appears 4 times per vec-4 per iteration (VEC_SIZE=4).
// Each work-item processes ~37 vec-4 groups per pass -> ~148 divisions.

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

constexpr int VEC_SIZE = 4;
constexpr int SIMD = 32;

struct WelfordState {
  float mean, m2, nf;
};

inline WelfordState welford_combine(WelfordState a, WelfordState b) {
  float delta = b.mean - a.mean;
  float total = a.nf + b.nf;
  float r = (total > 0.f) ? (b.nf / total) : 0.f;
  return {a.mean + delta * r, a.m2 + b.m2 + delta * delta * a.nf * r, total};
}

inline WelfordState welford_shfl(sycl::sub_group sg, WelfordState s, int offset) {
  return {
      sycl::shift_group_left(sg, s.mean, offset),
      sycl::shift_group_left(sg, s.m2, offset),
      sycl::shift_group_left(sg, s.nf, offset)};
}

void run_kernel(sycl::queue& q, sycl::half* X, sycl::half* Y,
                 const sycl::half* gamma, const sycl::half* beta, float eps,
                 int32_t D, int32_t S, int32_t DS, int64_t n_groups, int wg_size) {
  int n_sg_host = wg_size / SIMD;
  int64_t slm_size = n_sg_host * 3 + 2 + 2 * D;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> slm(slm_size, cgh);
    cgh.parallel_for(
      sycl::nd_range<1>(n_groups * wg_size, wg_size),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        using vec_t = sycl::vec<sycl::half, VEC_SIZE>;
        auto sg = item.get_sub_group();
        int32_t ng = static_cast<int32_t>(item.get_group(0));
        int lid = item.get_local_id(0);
        int wg_sz = item.get_local_range(0);
        int n_sg = wg_sz / SIMD;
        int sg_id = sg.get_group_linear_id();
        int sg_lid = sg.get_local_linear_id();
        int32_t DS_vec = DS / VEC_SIZE;
        sycl::half* xbase = X + static_cast<int64_t>(ng) * DS;
        sycl::half* ybase = Y + static_cast<int64_t>(ng) * DS;
        auto* xvec = reinterpret_cast<const vec_t*>(xbase);

        // === Pass 1: Welford reduction ===
        constexpr float inv_vec = 1.0f / static_cast<float>(VEC_SIZE);
        WelfordState st{0.f, 0.f, 0.f};
        for (int32_t vi = lid; vi < DS_vec; vi += wg_sz) {
          vec_t xv = xvec[vi];
          float x0 = static_cast<float>(xv[0]);
          float x1 = static_cast<float>(xv[1]);
          float x2 = static_cast<float>(xv[2]);
          float x3 = static_cast<float>(xv[3]);
          float batch_sum = (x0 + x1) + (x2 + x3);
          float batch_sum_sq = (x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3);
          float batch_mean = batch_sum * inv_vec;
          float batch_M2 = batch_sum_sq - batch_sum * batch_mean;
          st = welford_combine(st, WelfordState{batch_mean, batch_M2, static_cast<float>(VEC_SIZE)});
        }

        #pragma unroll
        for (int off = SIMD / 2; off > 0; off >>= 1) {
          WelfordState r = welford_shfl(sg, st, off);
          st = welford_combine(st, r);
        }

        if (sg_lid == 0) {
          slm[sg_id * 3 + 0] = st.mean; slm[sg_id * 3 + 1] = st.m2; slm[sg_id * 3 + 2] = st.nf;
        }
        sycl::group_barrier(item.get_group());

        if (sg_id == 0) {
          st.mean = (sg_lid < n_sg) ? slm[sg_lid * 3 + 0] : 0.f;
          st.m2   = (sg_lid < n_sg) ? slm[sg_lid * 3 + 1] : 0.f;
          st.nf   = (sg_lid < n_sg) ? slm[sg_lid * 3 + 2] : 0.f;
          #pragma unroll
          for (int off = SIMD / 2; off > 0; off >>= 1) {
            WelfordState r = welford_shfl(sg, st, off);
            st = welford_combine(st, r);
          }
        }

        float g_mean, g_rstd;
        if (lid == 0) {
          g_mean = st.mean;
          g_rstd = sycl::rsqrt(st.m2 / st.nf + eps);
          slm[n_sg * 3] = g_mean;
          slm[n_sg * 3 + 1] = g_rstd;
        }
        sycl::group_barrier(item.get_group());

        g_mean = slm[n_sg * 3];
        g_rstd = slm[n_sg * 3 + 1];

        // Precompute a[c], b[c] into SLM
        float* a_slm = &slm[n_sg * 3 + 2];
        float* b_slm = &slm[n_sg * 3 + 2 + D];
        for (int32_t c = lid; c < D; c += wg_sz) {
          float gv = static_cast<float>(gamma[c]);
          float bv = static_cast<float>(beta[c]);
          float a_c = g_rstd * gv;
          a_slm[c] = a_c;
          b_slm[c] = bv - g_mean * a_c;
        }
        sycl::group_barrier(item.get_group());

        // === Pass 2: BASELINE — genuine runtime division c = (j+v) / S ===
        auto* yvec = reinterpret_cast<vec_t*>(ybase);
        for (int32_t vi = lid; vi < DS_vec; vi += wg_sz) {
          int32_t j = vi * VEC_SIZE;
          vec_t xv = xvec[vi];
          vec_t yv;
          #pragma unroll
          for (int v = 0; v < VEC_SIZE; v++) {
            int32_t c = (j + v) / S;  // <-- RUNTIME INTEGER DIVISION
            yv[v] = static_cast<sycl::half>(a_slm[c] * static_cast<float>(xv[v]) + b_slm[c]);
          }
          yvec[vi] = yv;
        }
      });
  });
}

int main(int argc, char** argv) {
  bool inplace = false;
  int iters = 30;
  int64_t N = 1024, D = 192, S = 784, G = 1;
  int wg_size = 1024;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--inplace") inplace = true;
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--N" && i + 1 < argc) N = std::atoll(argv[++i]);
    else if (a == "--D" && i + 1 < argc) D = std::atoll(argv[++i]);
    else if (a == "--S" && i + 1 < argc) S = std::atoll(argv[++i]);
    else if (a == "--G" && i + 1 < argc) G = std::atoll(argv[++i]);
    else if (a == "--wg" && i + 1 < argc) wg_size = std::atoi(argv[++i]);
  }
  int64_t DS = D * S;
  int64_t n_groups = N * G;

  sycl::queue q(sycl::gpu_selector_v);
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  printf("shape: N=%lld D=%lld S=%lld G=%lld (DS=%lld) wg_size=%d | inplace=%d\n",
         (long long)N, (long long)D, (long long)S, (long long)G, (long long)DS, wg_size, (int)inplace);

  float eps = 1e-5f;
  int64_t total_elems = n_groups * DS;
  auto* X = sycl::malloc_device<sycl::half>(total_elems, q);
  auto* Y = inplace ? X : sycl::malloc_device<sycl::half>(total_elems, q);
  auto* gamma = sycl::malloc_device<sycl::half>(D, q);
  auto* beta = sycl::malloc_device<sycl::half>(D, q);

  q.parallel_for(sycl::range<1>(total_elems), [=](sycl::id<1> i) {
    uint32_t x = static_cast<uint32_t>(i[0]) * 2654435761u + 1;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    X[i[0]] = sycl::half(static_cast<float>(x) * 1e-9f);
  }).wait();
  q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> i) {
    uint32_t x = static_cast<uint32_t>(i[0]) * 2654435761u + 7;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    gamma[i[0]] = sycl::half(0.5f + static_cast<float>(x % 1000) * 1e-3f);
    beta[i[0]] = sycl::half(static_cast<float>((x >> 8) % 1000) * 1e-3f - 0.5f);
  }).wait();

  auto dispatch = [&]() {
    run_kernel(q, X, Y, gamma, beta, eps,
               static_cast<int32_t>(D), static_cast<int32_t>(S), static_cast<int32_t>(DS),
               n_groups, wg_size);
    q.wait();
  };

  // Warmup
  for (int i = 0; i < 5; i++) dispatch();

  // Benchmark
  for (int i = 0; i < iters; i++) dispatch();

  std::vector<sycl::half> y_h(8);
  q.memcpy(y_h.data(), Y, 8 * sizeof(sycl::half)).wait();
  printf("sample Y[0..3]=%f %f %f %f\n", (float)y_h[0], (float)y_h[1], (float)y_h[2], (float)y_h[3]);
  printf("done\n");
  return 0;
}
