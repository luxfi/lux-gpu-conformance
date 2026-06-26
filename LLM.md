# lux-gpu-conformance — cross-backend kernel-parity harness

A single self-contained runner (`conformance.cpp`) that proves **kernel parity**
across every lux-accel backend plugin: it dlopens each `lux_<name>.plugin`, runs
the SAME logical op (`out = a + b`, N=4096) through each backend's own shading
language, and asserts every backend's output is bit-identical to a CPU reference.

## One-line registration

The registry is the `kBackends[]` array in `conformance.cpp`. Each backend is
ONE row: `{ type, name, plugin-filename, kernel-source, entry }`. Today it
carries metal (MSL), webgpu (WGSL), cuda (CUDA C), hip (CUDA C), vulkan (GLSL).
This is the structure meant to drop straight into the lux-accel/gpu parity
harness when it exists — adding a backend stays one line.

## Plugin discovery

In order: `"<name>=<path>"` argv overrides → `$LUX_PLUGIN_DIR` → sibling build
trees (`../lux-<name>/build/lux_<name>.plugin` or `../../...` from a `build/`
subdir). Backends that can't be located or aren't available are reported, not
failed; the harness exits non-zero only on an actual cross-backend mismatch.

## Build + run (macOS, Metal + Vulkan live)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLUXACCEL_INCLUDE_DIR=/path/to/lux-accel/include
cmake --build build -j
export VK_ICD_FILENAMES=/opt/homebrew/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/molten-vk/lib:/opt/homebrew/opt/vulkan-loader/lib
./build/lux_gpu_conformance
```

Verified on `Apple M1 Max`: metal=MATCH, vulkan=MATCH (2 live backends,
bit-identical). cuda/hip report "unavailable (no runtime/hardware here)";
webgpu needs Dawn/wgpu-native. Run cuda/hip on NVIDIA/AMD hosts (evo/spark) to
extend the proven set to all five.
