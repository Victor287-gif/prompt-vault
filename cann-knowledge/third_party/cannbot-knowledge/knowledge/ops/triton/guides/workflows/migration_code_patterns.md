---
type: Guide
title: GPU→NPU 代码迁移模式（before/after）
description: GPU Triton kernel 迁移到 Ascend NPU 的 9 类代码改写模式：设备替换、Grid 对齐核数、移除 num_warps/num_stages、数据类型替换、care_padding、核内
  Tiling、Vector CMP 转 fp32、离散访存连续化。
tags: [task.migration, topic.quality.compatibility.version_migration, topic.compilation, topic.implementation.masking, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# GPU→NPU 代码迁移模式（before/after）

迁移项速查：`device='cuda'`→`'npu'`；补 `import torch_npu`；删 `get_active_torch_device()`；Grid 对齐物理核数；删 `num_warps`/`num_stages`；uint 系列/fp64 换受支持类型；Block Pointer 转置只能改 `order`；tl.load 需对齐（Vector 32B / CV 512B）。

## 模式 1：设备标识与导入替换

```diff
 import torch
+ import torch_npu
 import triton
 import triton.language as tl

- DEVICE = triton.runtime.driver.active.get_active_torch_device()
- x = torch.rand(size, device='cuda')
+ x = torch.rand(size, device='npu')
```

要点：必须 `import torch_npu`；全局替换 device；移除 `get_active_torch_device()` 与 `assert x.device == DEVICE` 一类校验。

## 模式 2：Grid 配置对齐物理核数

```diff
-    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
-    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
+    from triton.runtime import driver
+    props = driver.active.utils.get_device_properties(0)
+    num_cores = props["num_aicore"]
+    grid = lambda meta: (num_cores,)
+    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=triton.cdiv(n_elements, num_cores))
```

要点：Grid 优先 1D（2D 会被合并）；纯 Vector 算子 grid = Vector Core 数，含 `tl.dot` 算子 grid = AI Core 数；也可 `kernel[n, 1, 1](...)` 直接指定核数。

## 模式 3：移除 num_warps / num_stages

```diff
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 128}),
```

要点：NPU 无 Warp / 软件流水线概念，两参数无效；NPU 特有 autotune 参数 `multibuffer`，如 `triton.Config({'XS': 128, 'multibuffer': True})`。

## 模式 4：数据类型替换

```diff
-    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float64)
+    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
- x = torch.randn(size, dtype=torch.float64, device='cuda')
+ x = torch.randn(size, dtype=torch.float32, device='npu')
```

替换表：fp64→fp32（精度降低需评估）；uint8→int8、uint16→int16、uint32→int32、uint64→int64（注意符号位与范围）。性能相关：Vector ADD 用 int32 不用 int64（scalar 退化）；Vector CMP 用 fp32 不用 int64/int32。

## 模式 5：tl.load/tl.store 参数调整

```diff
-    data = tl.load(x_ptr + idx, mask=mask)
+    data = tl.load(x_ptr + idx, mask=mask, care_padding=False)
```

| 参数 | GPU | NPU | 说明 |
|------|-----|-----|------|
| `cache_modifier` / `eviction_policy` / `volatile` | 支持 | 不支持 | NPU 无 PTX cache 语义，移除 |
| `care_padding` | 不存在 | 新增 | 设 False 消除 Vector 置零与 MTE2 的依赖，提升并行度 |
| `other` | 默认填 0 | 默认填 0 | NPU 先 Vector 置零再 MTE2 搬运 |

## 模式 6：核内 Tiling 分块（避免 UB 溢出）

```diff
 @triton.jit
-def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
+def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
     pid = tl.program_id(axis=0)
-    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
-    mask = offsets < N
-    x = tl.load(x_ptr + offsets, mask=mask)
-    out = x * 2.0
-    tl.store(out_ptr + offsets, out, mask=mask)
+    base_offset = pid * BLOCK_SIZE
+    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
+    for sub_idx in range(num_sub_blocks):
+        sub_offset = base_offset + sub_idx * BLOCK_SIZE_SUB
+        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
+        mask = offsets < N
+        x = tl.load(x_ptr + offsets, mask=mask)
+        out = x * 2.0
+        tl.store(out_ptr + offsets, out, mask=mask)
```

要点：`BLOCK_SIZE` 控 coreDim（核间），`BLOCK_SIZE_SUB` 控 UB 用量（核内）；子块循环让搬入/计算/搬出流水并行；UB 容量 910B 192KB、910_95 256KB。

## 模式 7：Vector CMP 类型转换（性能优化）

```diff
     cols = tl.arange(0, BLOCK_N)
-    xbar = tl.where(cols < N, x - mean, 0.0)
+    cols_cmp = cols.to(tl.float32)
+    xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

要点：`tl.where` 条件中 int64/int32 比较退化为 scalar；转 fp32 后走 Vector CMP。`tl.load`/`tl.store` 的 mask 比较编译器通常能自动优化，`tl.where` 需手动转。

## 模式 8：离散访存改为连续访存

```diff
- block_ptr = tl.make_block_ptr(base=input_ptr, shape=(1024,), strides=(32,),
-     offsets=(i_t * 16,), block_shape=(BT,), order=(0,))
+ block_ptr = tl.make_block_ptr(base=input_ptr, shape=(1024, 32), strides=(32, 1),
+     offsets=(i_t * BT, 0), block_shape=(BT, 32), order=(1, 0))
```

要点：把一维展平数据视为二维，最低维 stride=1 保证连续访存；GPU 线程绑定最低维的写法在 NPU 上是离散访存，性能极差。

## 模式 9：Block Pointer 转置写法

```diff
 block_ptr = tl.make_block_ptr(base=x_ptr, shape=(M, N),
-    strides=(1, M),       # stride 交换实现转置（NPU 不支持）
-    offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))
+    strides=(N, 1),       # stride 反映真实内存布局
+    offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))   # order 表达转置
```

## 适配要点小结

1. 第一步永远是设备替换 + `import torch_npu`。
2. 删除全部 GPU 专属参数（`num_warps`/`num_stages`/`cache_modifier`/`eviction_policy`）。
3. Grid 对齐物理核数，避免分批调度。
4. 注意 UB 容量，必要时核内 Tiling。
5. 数据类型优先 fp32/int32，避免 scalar 退化。
6. 访存优先连续，避免离散访存。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [migrate_from_gpu.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/migrate_from_gpu.md)
- [performance_guidelines.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/performance_guidelines.md)
