---
type: Guide
title: sparse_unfold 类算子设计约束（邻域展开 Sparse-展开型 attention）
description: F.unfold/im2col 邻域展开（Sparse-展开型）attention 的判别特征与设计硬约束：aclnn 行布局舍入契约、softmax
  分 dtype 逐位配方、运行时除数解码 miscompile、dot 操作数 2 的幂、合并投影 BN 档。
tags: [task.design, topic.attention, topic.implementation.masking, topic.compute.neighborhood.convolution, topic.precision, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-26T08:23:10Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# sparse_unfold 类算子设计约束（邻域展开 Sparse-展开型 attention）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）；
> attention 主链约束（Layer 1、F1~F6、精度闸门、测量口径）继承 [flash_attention_constraints.md](flash_attention_constraints.md)，
> 家族内写法分类见 [attention_writing_classification.md](attention_writing_classification.md)（本类对应 **三·3 邻域展开 / Sparse-展开型**）。

适用范围：attention 主链的 KV 由 `F.unfold` / im2col **邻域展开**构造的算子
（`su-halo` 含 nn.Linear 投影段 / `su-plain` 无投影段 / `su-win` 固定几何窗口手写 gather）。
核心优化哲学：unfold 物化 `[b, c·S_kv, L]` 巨量拷贝是**显性浪费**——halo 重叠使同一像素被读进多个窗口，
读放大 `(1+2·halo/blk)²` 倍（block=8/halo=4 时 **4 倍**），外加 NCHW→块主序 permute 链。
**第一优化动作永远是：把展开从"物化"改成"每像素单次 dense 投影 + 按块窗口 gather"**；
同时认清投影 GEMM 占总算力 **>90%**（c=2048 case 估算：投影 ~866 GFLOP vs attention ~68 GFLOP），
**attention 侧天花板有限，先攻投影侧**。

## 本类别的算子级实现手册

- [`operators/sparse_unfold__halo_attention.md`](../operator_playbooks/sparse_unfold__halo_attention.md) — HaloAttention（邻域展开）实现手册：五阶段骨架、投影侧 vs attention 侧收益判据、舍入契约与性能基准（50/50，2.0973/2.1243）。

## §0 适用范围与算子分类

形态由**入参与写法**识别，不按算子名匹配。子类：`su-halo`（unfold 构造 KV + nn.Linear 投影段 + 块局部 softmax，**本类头号失败源在舍入契约**）、`su-plain`（同上但无投影段）、`su-win`（非 unfold 但同为"固定几何窗口选 KV"）。

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. 参考实现出现 `F.unfold(...)` / `tensor.unfold(...)` 且其输出喂进 attention 的 K/V；
2. 输入是 NCHW 特征图，attention 的 token 来自 `x.reshape(b,c,h//blk,blk,w//blk,blk).permute(0,2,4,3,5,1)` 分块展开；
3. 存在 `kernel_size=block+2*halo` 这种"块 + halo 邻域"参数对；
4. 输出需要从块主序 token scatter 回 NCHW（`.permute(0,5,1,3,2,4).reshape(b,c,h,w)` 形态）。

### §0.2 ★ 形态识别五问（Phase 2 第一步必须回答）

| # | 问题 | 影响 |
|---|------|------|
| **Q1** | `S_kv = (blk+2·halo)²` 是否 ≤ 256？ | 本类普遍成立 ⇒ **恒单块、单趟精确 softmax**，online rescale 分支零执行（FA 卡 L1.3 保留但退化）；>256 回落 FA 卡分块流式 |
| **Q2** | halo padding 位置参考是否掩码？ | 展开型参考**普遍不掩码**——padding 列 k=v=0 → score=0 **真实参与 softmax 分母**，与 FA 卡 F1 语义**相反**（勿当 padding 排除） |
| **Q3** | 投影段第三个 `nn.Linear` 是否 `bias=True`（默认）？ | 逐字核对参考源码；**漏 bias 是本类隐蔽失败源**（op50 首版实测踩坑） |
| **Q4** | 权重是否 forward 内 seed 化创建（`hash(key)`）？ | 复刻配方见 §1 S1 |
| **Q5** | 评测 dtype 是否三档混合（fp16/bf16/fp32）？ | 决定 L1.2 的 softmax 域配方是否分档——fp16 专属一步 cast，bf16/fp32 **不能**做 |

## §1 通用经验（跨算子，首次生成必须遵守）

### S1 ★ 权重复刻：`hash(key)` seed 配方（FA 卡 L1.11 在本类的落点）

forward 内 `torch.manual_seed(42)` **无条件调用**（逐字复刻，勿省）；cache-miss 分支
rng save → `manual_seed(hash(key) & 0xFFFFFFFF)` → **CPU fp32** 逐字复刻 `nn.Linear` 初始化 →
rng restore → `.to(dev,dt)` → 预转置 `[in,out]`（FA 卡 L1.12）。

```python
# ❌ NPU 上低精度采样 / 公式手推 bound —— 构造 device、采样 dtype、顺序次数、rng 四处缺一即全错
wq = torch.empty(INNER, c, device='npu', dtype=torch.float16).uniform_(-b, b)
# ✅ CPU fp32 字面量常数（host 侧禁 math.sqrt，repr 级字面量 round-trip 逐位一致）
_GAIN  = 0.5773502691896257      # sqrt(2/(1+sqrt(5)**2)) 的双精度 repr
_SQRT3 = 1.7320508075688772
bound_in  = _SQRT3 * (_GAIN / (c ** 0.5))      # kaiming_uniform_(a=√5) ≡ uniform(±1/√fan_in)
bound_out = _SQRT3 * (_GAIN / (INNER ** 0.5))
wq  = torch.empty(INNER, c).uniform_(-bound_in, bound_in)     # CPU fp32，顺序/次数与参考一致
wkv = torch.empty(2*INNER, c).uniform_(-bound_in, bound_in)
wo, bo = ..., ...               # ⚠️ to_out 默认 bias=True：weight 与 bias 都要建（Q3）
```

同进程内 `hash(key)` 稳定（tuple 含 int/dtype/device，无 str）；**权重单验先行**——
先逐个 `torch.equal` 再验投影最后验 attention（FA 卡 L1.11 诊断顺序；
权重错与布局错**互相掩盖**，只修一处 passed_cases 仍为 0）。

### S2 空输入分支逐字复刻

参考含 `if b == 0 or c == 0 or h == 0 or w == 0: return x` 时，实现同样直接返回 x，
**不进任何 kernel launch**（b=0 case 在 benchmark 里 `PROFILER_COLLECT_FAIL` 属预期，
勿当成失败去修）。

### S3 常量表 host 预计算是被允许的模式，但有边界

窗口行表 / tile 表 / 像素表等**纯索引常量**在 host Python 预计算（shape 级 cache）
不违反 FA 卡 L1.6（那禁的是**数据布局搬运**）。但注意两点：
① 表的生成不能依赖"无 dot kernel 自己算行号"——大规模下同样踩 L1.3 的解码
miscompile（op50 实测 rowmap kernel 自身产出越界行号 qrow=4108 > 4095）；
② 终态更优解是 **constexpr 除数直接在 kernel 内解码**（零额外表与 launch，L1.3），
host 表只作兜底。

## Layer 1 设计约束（首次生成就要全部满足）

> 本节共 5 条（L1.1 ~ L1.5），全部在本类算子上实测。违反前两条的表现是
> **"结构正确但 ~96-99% 元素错 1-2 ulp"**，违反后三条是 miscompile/编译崩溃——
> 都极难从最终输出反推，precheck 必须整节摘录。

- **L1.1 ★★★ 必须 与参考共用 aclnn 的边界保持同布局；换成自研 Triton GEMM 后 M 行布局才不敏感**
  - **Why:** 实测（op50，case `[6,192,38,10]` fp16）：同一份输入数据，aclnn `F.linear`
    像素主序 vs 块主序（参考 `q_inp/kv_inp/out` 的 `(b, nblk, s)` 序）——块主序与参考输出
    **maxdiff = 0.0（逐位一致）**；像素主序 **98.7% 元素不等、maxd 0.0449、mismatch(2⁻¹⁰) 96.3%**。
    aclnn 分块/累加序随 M 布局变化，1 ulp 舍入差经 softmax→PV→out 投影逐层放大后**整盘越阈**。
  - **How to apply:** 中间量若还回 torch/aclnn 计算（或对齐 aclnn golden），行序必须逐字复刻参考；
    **自研 Triton GEMM（K 循环顺序累加）对 M 布局不敏感**——终态 q/kv 投影即按像素主序每像素投影一次
    （实测 50/50 通过）。attention 输出 → out 投影边界按参考**块主序**落盘是实测通过的稳妥路径：
    ```python
    # ❌ attention 输出按像素主序落盘，再喂 out 投影 —— 即使数值路径全对也全错
    tl.store(o_ptr + pixel_row[:, None] * INNER + ...)          # 像素主序
    y = aclnn_linear_or_next_stage(o)                            # 下游还是 aclnn 语义
    # ✅ attention 输出按参考块主序落盘（bh*SQ + s），下游边界与参考同布局
    tl.store(o_ptr + (bh * SQ + ms)[:, None] * INNER + ...)     # 块主序 = 参考 to_out 输入
    ```

- **L1.2 ★ 必须 按 L1.2 分 dtype 配方复刻 aclnn softmax（sim 物化在输入 dtype；仅 fp16 把 `(sim−m)` 舍回 fp16 再 exp）**
  - **Why:** fp16 case 穷举 8 种实现变体（内部全 fp32 / 全 fp16 / sum fp16 / exp 移位 / …），
    **只有下述配方逐位一致**，其余全部差 4.88e-4（fp16 ulp 级）；`(x−m)` 舍回输入 dtype 这步
    **bf16/fp32 做了反而引入新偏差**（终态 15 个 bf16 case 实测通过为证）。
  - **How to apply:**
    ```python
    sim = tl.dot(qs, kt, out_dtype=tl.float32)
    sim = sim.to(tl.float16).to(tl.float32)          # 舍入点③：bmm 输出物化在输入 dtype
    row_max = tl.max(sim, axis=1)[:, None]
    d = sim - row_max
    if IN_DT == 1:                                    # ⚠️ 仅 fp16 需要，bf16/fp32 禁做
        d = d.to(tl.float16).to(tl.float32)           # (x−max) 减法结果落 fp16 再升回
    e = tl.exp(d)                                      # exp/sum/div 恒 fp32
    p = (e / row_sum).to(elem_ty)                      # 舍入点④：softmax 输出落输入 dtype
    ```
    `q * scale` 同理先升 fp32 乘再落回输入 dtype（scale=2⁻³ 为 2 的幂，精确可折进投影 epilogue）。

- **L1.3 ★ 禁止 运行时（非 constexpr）除数的标量 div/mod 多层解码（大规模 miscompile）；constexpr 除数的解码实测可用**
  - **Why:** 除数是运行时标量（host 传参）时，Nblk=512 规模产生越界行号（qrow=4108 > 4095，非确定），
    Nblk=16 规模却正确；触发特征（op50 mini 系列实测）：**纯 store kernel 中同一解码正确，dot kernel 中才
    miscompile**，错误率 ~75% 均匀分布、跨运行非确定。全 constexpr 化除数以外的部分不解决。
  - **How to apply:** 优先序 **constexpr 除数解码（终态验证）> host 常量表（S3）> 运行时除数（禁用）**；
    行号能用纯乘法 `nbk * SQ + offs_s` 最稳。
    ```python
    # ❌ 除数是运行时标量（host 传参）：Nblk=512 规模产生越界行号（非确定）
    def _attn_kernel(..., nblk_per_b, wb, ...):        # 运行时参数
        hb = nblk // wb                                # miscompile；Nblk=16 规模却正确
    # ✅ 除数 constexpr（编译期强度削减成乘法），终态代码 50/50 验证
    def _attn_kernel(..., NH: tl.constexpr, nhbnwb: tl.constexpr, nwb: tl.constexpr, ...):
        bh = work // NH; head = work - bh * NH
        b = bh // nhbnwb; brem = bh - b * nhbnwb
        hb = brem // nwb;  wb_ = brem - hb * nwb
    ```

- **L1.4 ★ 必须 dot 操作数 2 的幂——ceil16 仅适用向量 tile**
  - **Why:** `ceil16(81)=96` 做 BSKV 触发 bishengir 编译崩溃（`ConvertLinalgRToBinary encounters error:
    LLVM ERROR: Failed to obtain op buffer shape size which should be static`），fp32 ieee dot 同样崩。
    FA 卡 F3 的 ceil16 红利只能在**向量 tile**（load/store/where/arange）上拿。
  - **How to apply:** dot 操作数形状 next_pow2 封顶 + mask；同一 kernel 内"向量 ceil16 + dot 2 幂"两套规则并存：
    ```python
    # ❌ ceil16(81) = 96 做 BSKV：编译崩溃
    BSKV = (SKV + 15) // 16 * 16
    # ✅ dot 操作数形状 next_pow2 封顶 + mask；向量 tile 仍可 ceil16
    BSKV = _pow2(SKV, lo=16, hi=128)                   # 16/32/64/128/256
    ```

- **L1.5 ★ 禁止 合并投影 kernel 的 weight-tile 转置路径在 BN>64 时静默错值**
  - **Why:** 本后端 merged-projection（q|kv 拼单 GEMM，`[BN, BK] weight → tl.trans → tl.dot`）
    在 **BN>64 时静默产出错值**（不报错）；out-projection BN=128 不受影响。BN=256 / BK=128 的
    masked-B-tile padding 缺陷同样要排除。**扫参扫出"更快"的配置必须先过精度再采信**——
    静默错值只会在 verify 里表现为 matched_ratio 断崖。
  - **How to apply:** 16 位 dense 取 `(BM, BN=64, BK=64)` 最快且可靠；任何 tile 档位改动后先跑 verify 再采信。

## §2 证伪方向 / 常见陷阱

### §2.1 ⛔ 证伪方向要点（不要重跑）

| 方向 | 结果 |
|---|---|
| **attention 输出像素主序落盘 + 下游对齐 aclnn golden** | ⛔ **98.7% 元素 1 ulp 不等、mismatch 96.3%**（L1.1）。自研 Triton GEMM 内部换布局不在此列 |
| **softmax 内部全 fp32 或全 fp16** | ⛔ 8 变体穷举全差 4.88e-4，仅 L1.2 配方逐位；`(x−m)` cast 到 bf16/fp32 同样引入新偏差 |
| **运行时（非 constexpr）除数的 kernel 内 div/mod 解码链** | ⛔ Nblk=512 越界行号、~75% 均匀错、非确定；Nblk=16 却正确（规模相关，最险） |
| **用"无 dot kernel"生成行号表再查表** | ⛔ 生成 kernel 自身在大规模踩解码 miscompile，表即污染；且多一张表一次 launch |
| **非 2 幂 dot tile（ceil16 进 dot 操作数）** | ⛔ `should be static` 编译崩溃，fp32 ieee dot 同样（L1.4） |
| **merged 投影 weight-tile 转置 BN>64 / BN=256 / BK=128** | ⛔ 静默错值或 masked-B-tile padding 缺陷（L1.5） |
| **按窗口重复投影（窗口直接进 GEMM A 操作数）** | ⛔ halo 重叠区读放大 R=(1+2h/blk)² 倍投影量；先每像素投影再 gather 严格更省 |

### §2.2 陷阱表

| 现象 | 根因 | 处理 |
|---|---|---|
| NaN ~10% + 后续 case 级联 `aicore exception` | 运行时除数解码 miscompile 越界行号；级联是设备污染 | 先单跑第一个失败 case 定位 → L1.3 |
| ~96-99% 元素差 1-2 ulp、maxd ~0.045 | 舍入契约错位三连：边界布局 / softmax 域 / 漏 bias | §6.1 判定顺序 → L1.1/L1.2/Q3 |
| fp32 case `should be static` 编译崩溃 | 非 2 幂 dot tile | L1.4 |
| 单 case 输出恒 0 或半张图未写 | 空输入分支未复刻 / K6 scatter 漏 tile | S2 + 覆盖率检查 |
| 扫参后更快但 matched_ratio 断崖 | BN>64 weight-tile 转置静默错值 | L1.5 |
| verify 全过但 speedup 分子异常好 | 编译失败 case 被排除在 geomean 外 / host 侧搬运 | 先看 perf 的 `passed_cases == total_cases` |
| host 重算链与参考输出不等 | debug 脚本自身 reshape 序错 | 先修 debug 脚本再怀疑 kernel |
| `q*scale` 折进投影 epilogue 后 fp16 边界失败 | scale 非 2 的幂（仅当 dim_head≠64²ⁿ） | 非 2 幂 scale 改回独立乘法并落输入 dtype |

### §2.3 证伪定位要点（"结构对但 1-2 ulp 全错"时）

1. **先看覆盖率**：`torch.empty` + 部分覆盖 = 未初始化内存（NaN 与精度判定同时触发，像"精度不够"实为覆盖不全）；
2. **再看解码 miscompile**：~75% 均匀错 + 非确定 + 纯 store kernel 正常 ⇒ L1.3，不要去调精度；
3. **最后才是舍入契约**：按六舍入点逐处核对——①q 投影（含 ×scale=2⁻³）②kv 投影 ③QK bmm
   （fp32 acc → cast 输入 dtype → 升回）④softmax 输出 ⑤PV bmm ⑥out 投影（**含 bias**，Q3）。
   六处一个不能少；fp32 case 全程 ieee dot，任何中间量不落低精度。
   **dot vs aclnn 固有累加差可容忍**（sim 端仅 0.04% 元素不等、传播到输出 maxd 0.0039）；**真杀手是舍入点错位**。
