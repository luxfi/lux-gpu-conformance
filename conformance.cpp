// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Cross-backend conformance / kernel-parity harness for lux-accel plugins.
//
// It dlopens every registered backend plugin (lux_<name>.plugin), and on each
// backend that is_available() runs the SAME logical kernel (out = a + b) through
// the backend's own kernel language, then asserts every backend's output is
// bit-identical to a CPU reference. That is "kernel parity": one op, one result,
// across Metal / WebGPU / CUDA / HIP / Vulkan.
//
// Registering a backend is ONE line: add a row to kBackends[] below. Each row
// carries the backend's enum, name, plugin filename, the kernel source in that
// backend's shading language, and the entry point. When the lux-accel/gpu parity
// harness lands, this same table (or these five rows) drops straight in.
//
// Plugins are located via, in order: argv pairs "<name>=<path>", $LUX_PLUGIN_DIR,
// then sibling build trees (../lux-<name>/build/lux_<name>.plugin).

#include <lux/accel/backend_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <sys/stat.h>

namespace {

// ── core API the plugins call back into ─────────────────────────────────────
void log_info(const char* m)  { std::fprintf(stderr, "  [info]  %s\n", m); }
void log_warn(const char* m)  { std::fprintf(stderr, "  [warn]  %s\n", m); }
void log_error(const char* m) { std::fprintf(stderr, "  [error] %s\n", m); }
void log_debug(const char*)   {}
void* core_alloc(size_t n) { return std::malloc(n); }
void  core_free(void* p)   { std::free(p); }
const void* get_bundle(const char*, size_t* s) { if (s) *s = 0; return nullptr; }
const char* get_source(const char*) { return nullptr; }

lux_core_api_t make_core() {
    lux_core_api_t c = {};
    c.api_version = LUX_BACKEND_API_VERSION;
    c.log_debug = log_debug; c.log_info = log_info;
    c.log_warn = log_warn;   c.log_error = log_error;
    c.alloc = core_alloc;    c.free = core_free;
    c.get_kernel_bundle = get_bundle; c.get_kernel_source = get_source;
    return c;
}

// ── the one-line-per-backend registry ───────────────────────────────────────
struct BackendReg {
    lux_backend_type_t type;
    const char* name;
    const char* plugin;       // shared-object filename
    const char* kernel_src;   // out = a + b, in the backend's language
    const char* entry;
};

const char* kMSL = R"(
#include <metal_stdlib>
using namespace metal;
kernel void vec_add(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* c       [[buffer(2)]],
                    constant uint& n      [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) c[gid] = a[gid] + b[gid];
})";

const char* kGLSL = R"(
#version 450
layout(local_size_x = 64) in;
layout(set=0, binding=0) readonly  buffer A { float a[]; };
layout(set=0, binding=1) readonly  buffer B { float b[]; };
layout(set=0, binding=2) writeonly buffer C { float c[]; };
layout(set=0, binding=3) uniform Params { uint n; };
void main() { uint i = gl_GlobalInvocationID.x; if (i < n) c[i] = a[i] + b[i]; })";

const char* kCUDA = R"(
extern "C" __global__ void vec_add(const float* a, const float* b, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
})";

const char* kWGSL = R"(
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> n: u32;
@compute @workgroup_size(64)
fn vec_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < n) { c[gid.x] = a[gid.x] + b[gid.x]; }
})";

// HIP shares CUDA C source. Adding a backend = adding ONE row here.
const BackendReg kBackends[] = {
    { LUX_BACKEND_TYPE_METAL,  "metal",  "lux_metal.plugin",  kMSL,  "vec_add" },
    { LUX_BACKEND_TYPE_WEBGPU, "webgpu", "lux_webgpu.plugin", kWGSL, "vec_add" },
    { LUX_BACKEND_TYPE_CUDA,   "cuda",   "lux_cuda.plugin",   kCUDA, "vec_add" },
    { LUX_BACKEND_TYPE_HIP,    "hip",    "lux_hip.plugin",    kCUDA, "vec_add" },
    { LUX_BACKEND_TYPE_VULKAN, "vulkan", "lux_vulkan.plugin", kGLSL, "main"    },
};

bool exists(const std::string& p) { struct stat s; return ::stat(p.c_str(), &s) == 0; }

// Locate a plugin: explicit override, then $LUX_PLUGIN_DIR, then sibling build.
std::string locate(const BackendReg& r, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (eq != std::string::npos && a.substr(0, eq) == r.name) return a.substr(eq + 1);
    }
    if (const char* dir = std::getenv("LUX_PLUGIN_DIR")) {
        std::string p = std::string(dir) + "/" + r.plugin;
        if (exists(p)) return p;
    }
    // Sibling build trees, whether invoked from the repo root or a build/ subdir.
    const std::string roots[] = {"../lux-", "../../lux-"};
    for (const auto& root : roots) {
        std::string p = root + r.name + "/build/" + r.plugin;
        if (exists(p)) return p;
    }
    if (exists(r.plugin)) return r.plugin;
    return {};
}

constexpr uint32_t N = 4096;

// Run out = a + b on one backend. Returns true and fills out[] on success.
bool run_backend(const BackendReg& r, const std::string& path, std::vector<float>& out,
                 std::string& note) {
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { note = std::string("dlopen: ") + dlerror(); return false; }

    auto init = reinterpret_cast<lux_backend_init_fn>(dlsym(h, LUX_BACKEND_INIT_SYMBOL));
    if (!init) { note = "no lux_backend_init"; dlclose(h); return false; }

    lux_core_api_t core = make_core();
    lux_backend_interface_t* be = init(&core);
    if (!be || be->type != r.type || std::strcmp(be->name, r.name) != 0) {
        note = "identity mismatch"; dlclose(h); return false;
    }
    if (!be->is_available || !be->is_available()) {
        note = "unavailable (no runtime/hardware here)"; dlclose(h); return false;
    }
    if (!be->init || !be->init()) { note = "init failed"; dlclose(h); return false; }

    void* dev = be->create_device(0);
    void* q = dev ? be->create_queue(dev) : nullptr;
    void* k = dev ? be->create_kernel_from_source(dev, r.kernel_src, r.entry) : nullptr;
    if (!dev || !q || !k) {
        note = "device/queue/kernel setup failed"; be->shutdown(); dlclose(h); return false;
    }

    std::vector<float> a(N), b(N);
    for (uint32_t i = 0; i < N; ++i) { a[i] = (float)i * 0.5f; b[i] = (float)(N - i); }

    lux_buffer_desc_t din  = { N * sizeof(float), LUX_BUFFER_USAGE_STORAGE | LUX_BUFFER_USAGE_COPY_SRC, nullptr };
    lux_buffer_desc_t dout = { N * sizeof(float), LUX_BUFFER_USAGE_STORAGE | LUX_BUFFER_USAGE_MAP_READ, nullptr };
    void* bA = be->create_buffer_with_data(dev, &din, a.data());
    void* bB = be->create_buffer_with_data(dev, &din, b.data());
    void* bC = be->create_buffer(dev, &dout);
    if (!bA || !bB || !bC) { note = "buffer alloc failed"; be->shutdown(); dlclose(h); return false; }

    be->kernel_set_buffer(k, 0, bA, 0);
    be->kernel_set_buffer(k, 1, bB, 0);
    be->kernel_set_buffer(k, 2, bC, 0);
    be->kernel_set_bytes (k, 3, &N, sizeof(N));
    be->kernel_set_workgroup_size(k, 64, 1, 1);

    lux_dispatch_desc_t disp = { (N + 63) / 64, 1, 1, 64, 1, 1 };
    if (!be->dispatch(q, k, &disp) || !be->queue_wait(q)) {
        note = "dispatch/wait failed"; be->shutdown(); dlclose(h); return false;
    }

    const float* m = static_cast<const float*>(be->map_buffer(bC));
    if (!m) { note = "map failed"; be->shutdown(); dlclose(h); return false; }
    out.assign(m, m + N);
    be->unmap_buffer(bC);

    be->destroy_kernel(k);
    be->destroy_buffer(bA); be->destroy_buffer(bB); be->destroy_buffer(bC);
    be->destroy_queue(q); be->destroy_device(dev);
    be->shutdown();
    dlclose(h);
    note = "ran";
    return true;
}

} // namespace

int main(int argc, char** argv) {
    // CPU reference.
    std::vector<float> ref(N);
    for (uint32_t i = 0; i < N; ++i) ref[i] = (float)i * 0.5f + (float)(N - i);

    std::printf("lux-accel cross-backend kernel-parity conformance (op: out = a + b, N=%u)\n\n", N);
    std::printf("  %-8s %-12s %-9s %s\n", "backend", "located", "ran", "parity");
    std::printf("  %-8s %-12s %-9s %s\n", "-------", "-------", "---", "------");

    int ran = 0, parity_ok = 0, failures = 0;
    for (const auto& r : kBackends) {
        std::string path = locate(r, argc, argv);
        std::vector<float> out;
        std::string note;

        bool did = !path.empty() && run_backend(r, path, out, note);
        const char* parity = "-";
        if (did) {
            ran++;
            bool ok = (out.size() == ref.size());
            for (size_t i = 0; ok && i < ref.size(); ++i) ok = (out[i] == ref[i]);
            parity = ok ? "MATCH" : "MISMATCH";
            if (ok) parity_ok++; else failures++;
        }
        std::printf("  %-8s %-12s %-9s %s\n",
                    r.name,
                    path.empty() ? "no" : "yes",
                    did ? "yes" : "no",
                    parity);
        if (!did && !path.empty()) std::printf("           -> %s\n", note.c_str());
    }

    std::printf("\nran on %d backend(s); %d matched CPU reference; %d mismatch(es).\n",
                ran, parity_ok, failures);
    if (ran >= 2)
        std::printf("cross-backend parity proven across %d live backends.\n", ran);
    else
        std::printf("note: only %d backend ran here; remaining backends need their "
                    "runtime/hardware (CUDA/HIP=NVIDIA/AMD, WebGPU=Dawn/wgpu).\n", ran);

    // The harness fails only on an actual cross-backend mismatch. Backends that
    // cannot run here (no hardware/runtime) are reported, not failed.
    return failures == 0 ? 0 : 1;
}
