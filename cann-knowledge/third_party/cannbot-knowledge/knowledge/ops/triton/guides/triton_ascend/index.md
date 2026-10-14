# triton-ascend

上游 [https://gitcode.com/Ascend/triton-ascend](https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/) 锁定 commit `3c83caf1`。

本 bundle 记录 Triton Ascend 的**平台能力与 API 契约**——随上游版本变化的部分。
经验性的「怎么写快 / 怎么写对」不在此处，见
设计期约束（`../constraints/index.md`）、优化期改写（`../../optimizations/operator_optimization/index.md`）
与单算子实现手册（`../operator_playbooks/index.md`）。

> **溯源说明**：本 bundle 的卡片由实践提炼而成，非从上游文档文件逐节抽取，
> 故 `resource` 指向锁定 commit 的仓库树根而非具体文档路径。它声明的是
> 「本卡描述的是该 commit 的 API 行为」，可据此在上游升级时定位需复核的卡片。

## 本 bundle 卡片

- [Triton Ascend API 参考](api_reference.md)：def kernel_function(...):。
- [Triton Ascend 完整代码示例](code_examples.md)：def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):。
- [Grid 启动限制与核心数查询](grid_launch_limits.md)：import triton.runtime.driver as driver。
- [编程基础：核心概念与标准 Kernel 结构](programming_basics.md)：Triton Ascend 采用与 Triton GPU 类似的执行模型：。
