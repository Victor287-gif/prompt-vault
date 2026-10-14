---
type: Guide
title: Triton-Ascend 环境搭建与第一个 Kernel
description: Triton-Ascend 环境搭建命令序列（pip/源码/Docker）、版本兼容矩阵、环境验证步骤，以及第一个向量加法 kernel
  的完整最小示例与 NPU 上手要点。
tags: [task.implementation, topic.compilation, topic.implementation.masking, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# Triton-Ascend 环境搭建与第一个 Kernel

## 版本与硬件要求

- Python 3.9 ~ 3.11；CANN 建议 8.5.0（配套 Triton-Ascend 3.2.0）；torch_npu 2.7.1。
- `TRITON_ASCEND_ARCH` 合法值：Ascend910B1-B4 / Ascend910_93xx 系列 / Ascend910_95xx 系列 / Ascend310B1-B4。
- UB 容量：910_95/950 系列 256KB（独立编译路径 `linalg_to_bin_enable_npu_compile_910_95`，不支持 FFTS）；910B 等其他系列 192KB。
- 操作系统 Linux (aarch64/x86_64)，推荐单卡 32GB 显存。

## 安装命令序列

### pip 安装（推荐快速上手）

```bash
# 若报 "No matching distribution found for torch==2.7.1+cpu"，先手动装 CPU 版 torch
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.7.1
pip install triton-ascend
```

社区 Triton 与 Triton-Ascend **互斥**（其他包可能自动装社区版 triton 将其覆盖），被覆盖时：

```bash
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

Nightly（未稳定测试，注意匹配 Python 版本与架构）：

```bash
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir \
  --trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org
```

### 源码安装（关键步骤）

```bash
sudo apt install zlib1g-dev clang-15 lld-15        # CentOS: yum install -y zlib-devel；clang/lld >= 15
pip install ninja cmake wheel pybind11
git clone https://gitcode.com/Ascend/triton-ascend/tree/main
cd triton-ascend
LLVM_SYSPATH=/path/to/LLVM pip install -e python   # 可选：指定本地 LLVM（Triton 依赖 LLVM 20）
```

GCC < 9.4.0 报 `unable to find library -lstdc++fs` 时，取消 `CMakeLists.txt` 中 `link_libraries(stdc++fs)` 的注释后重建。

### Docker 安装

构建参数：`CHIP_TYPE=A3|910b`（A3 = Atlas A3 训练/910C；910b = Atlas A2），`CANN_VERSION=8.5.0|8.3.RC1|8.3.RC2|8.2.RC1|8.2.RC2`。

```bash
docker build --build-arg CHIP_TYPE=A3 --build-arg CANN_VERSION=8.5.0 \
  -t triton-ascend-image:latest -f ./docker/Dockerfile .
```

启动容器需 `--privileged` 并挂载 `/dev/davinci0-7`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`、`/usr/local/Ascend` 等设备与目录。

## 环境验证步骤

```bash
npu-smi info                                        # 1. 检测 NPU 设备与 AICore 数
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 2. CANN 环境变量（非 root: ${HOME}/Ascend/...）
```

```python
import torch, torch_npu                              # 3. 验证 torch_npu
print(torch.npu.is_available(), torch.npu.device_count())
x = torch.randn(3, 3, device='npu')
import triton, triton.backends.ascend                # 4. 验证 Triton-Ascend 后端（确认非社区版）
```

```bash
# 5. 跑通官方示例，预期 "The maximum difference between torch and triton is 0.0"
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

查询核数（可选）：`driver.active.utils.get_device_properties(torch.npu.current_device())` 返回 `num_aicore` / `num_vectorcore`。

## 第一个 Kernel：向量加法（完整最小示例）

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
x = torch.rand(98432, device='npu')
y = torch.rand(98432, device='npu')
print(f'max diff: {torch.max(torch.abs((x + y) - add(x, y)))}')
```

核心 API 语义：`tl.program_id(axis)` 对应当前 AI Core 任务编号；`tl.arange(0, BLOCK_SIZE)` 生成块内元素索引；`tl.load`/`tl.store` 对应 GM↔UB 数据搬运，`mask` 防越界；`tl.constexpr` 参数编译期确定，可用作 shape。`grid` 方括号传并行配置、圆括号传 kernel 参数；用 lambda 形式可随 autotune 的 BLOCK_SIZE 自动调整。

## NPU 与 GPU 的上手差异

1. 设备标识：`device='cuda'` → `device='npu'`，且必须 `import torch_npu`。
2. 分核策略：NPU 推荐**固定核数 + 核内循环**（`for block_idx in range(pid, NUM_BLOCKS, NUM_CORE)`），避免按数据量动态分核带来的核启动开销。
3. 对齐要求：VV 类（纯 Vector）算子尾轴 32B 对齐；CV 类（Cube+Vector 融合）尾轴 512B 对齐。
4. UB 限制：BLOCK_SIZE 过大报 `ub overflow`（910B 192KB / 910_95 256KB）。
5. 分核数规则：纯 Vector 算子 grid = Vector Core 数（约为 AI Core 数 2 倍）；含 `tl.dot` 算子 grid = AI Core 数；coreDim ≤ 65535。
6. NPU 默认开启 multibuffer（双缓冲，可用 UB 减半），实现搬运与计算重叠。
7. `num_warps` 在默认 SIMD 编译模式下影响很小（NPUOptions 默认 num_warps=4, warp_size=32）。

## 常见问题速查

| 问题 | 解决 |
|---|---|
| `No matching distribution found for torch==2.7.1+cpu` | 先装 CPU 版 torch 2.7.1，再装 torch_npu |
| `ASCEND_HOME_PATH is not set` | `source <ascend-toolkit>/set_env.sh` |
| 安装其他包后 Triton-Ascend 失效 | 社区 triton 覆盖了 triton-ascend，卸载两者后重装 triton-ascend |
| `ub overflow, requires ... bits` | 减小 BLOCK_SIZE，或核内 Tiling 子块循环（BLOCK_SIZE_SUB） |
| `coreDim=xxxx can't be greater than UINT16_MAX` | 固定核数 + 核内循环，勿按数据量动态分核 |
| nightly 安装 SSL 报错 | 追加 `--trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org` |

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [main

#](https://gitcode.com/Ascend/triton-ascend/tree/main

#)
- [installation_guide.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/installation_guide.md)
- [quick_start.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/quick_start.md)
- [get_ascend_devices.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/tools/get_ascend_devices.py)
- [utils.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/utils.py)
- [01-vector-add.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/01-vector-add.py)
- [programming_guide.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/programming_guide.md)
- [compiler.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py)
