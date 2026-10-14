---
type: Guide
title: MSE 的 Lane 累加与跨核归约
description: 保存两种归约树，补齐短尾、标量写回、工作区和全核到达条件。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss/mse_loss_tiling_example_for_lowering.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss/mse_loss_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss/mse_loss_process_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss_one_core/mse_loss_one_core_tiling_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss_one_core/mse_loss_one_core_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss_one_core/mse_loss_one_core_process_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/inter_core_sync/sync_all.md
  role: constraint
created_at: '2026-09-11T05:05:12Z'
updated_at: '2026-09-11T05:05:12Z'
platforms: []
---

# MSE 的 Lane 累加与跨核归约

mean MSE为sum((pred-target)^2)/N，前提是输入配对/广播与N符合接口。来源一核方案每个tile按lane累计acc[j]+=sq_tile[j]，最后对tileLength个lane归约；多核方案每tile先归约为scalar，核内再累加partial，最后核0汇总全部核。两种求和树在实数等价，浮点顺序与误差不同，不能用一方结果强制宣称另一方错误。

一核方案acc先初始化0，对最后短tile只更新有效cnt个lane，其余lane保留此前贡献；若补齐sq为0可保持整长Add，但不能把acc尾lane清0而丢掉之前满tile贡献。源ceil(nElements/1024)配整块DataCopy会过读；多核pivot+ceil(2048)也同样需要每核myElems和有效cnt。老init floor(elemsPerCore/tileSize)同时可能丢核间余数和核内尾部。

输出仅一个数学结果，来源InferShape复制pred形状是接口错误，需与实际标量/单元素合同一致；outputGm长度1不能靠更大分配假装形状正确。普通DataCopy(...,1)的4B写回及每核workspace[pid]写回不普遍合法，使用满足短尾与地址要求的路径并确认相邻核写入互不破坏。只有已写的nUsed个partial可被最终读取，pad区不能参与sum。

多核阶段需要每个实际参与核写出partial完成后，全核同步，再由核0读。SyncAll不是任意环境中无条件合法：核型、参与者、所有核到达、事件资源和实际重载支持须匹配。硬同步与软件同步不同，软件接口还需GM/UB标记区及初始化，不能把nUsed个partial空间兼作未分隔标记。不能在闲核提前return后让其他核继续等待其到达。

来源CopyOut1从Scalar SetValue直接发DMA且立即Free，未通过出队交接；需明确Scalar→MTE3可见性、DMA完成与buffer复用的合同。ReduceSum后GetValue需要V结果已就绪。核内queue交接与核间发布是不同依赖，SyncAll不能修复发布之前的错误数值或越界。

Host申请userWorkspace=nUsed*sizeof(float)+sysWorkspace并不自动决定Kernel的user pointer，按框架取实际用户区基址。workspaceInQueue需覆盖nUsed有效值及搬运物理补齐，归约scratch按nUsed和重载要求，不因tileSize2048就必定足够所有核数。源一核6个tileLength FP32区加输出至少需24*tileLength有效数据字节和输出对齐，多核7个tileSize区加workspace输入为28*tileSize+4*nUsed，均需额外资源与对齐检查。

N0时均值按接口定义可为NaN或其他行为，不能除以0后默认正常；sum/none reduction也不是这两篇mean算法。减法/平方可能溢出，FP32求和并非所有N和数值范围都足够，验证时保留算法精度合同和独立reference。
