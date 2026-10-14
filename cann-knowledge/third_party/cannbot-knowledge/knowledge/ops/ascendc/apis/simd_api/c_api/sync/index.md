# 同步控制

- [AscLock](asc_lock.md)：用于AI Core内部异步流水线之间的同步，根据mutex_id获取Mutex，可按需阻塞指定流水线的执行。
- [asc_sync（同步）](asc_sync.md)：等待所有流水线操作完成。
- [AscSyncBlockArrive](asc_sync_block_arrive.md)：与asc_sync_block_wait配合使用，通过flag_id关联，用于组内block和subblock间的信号同步。
- [AscSyncBlockWait](asc_sync_block_wait.md)：与asc_sync_block_arrive配合使用，等待所有同步对象到达flag_id对应的同步点后解除阻塞。
- [AscSyncDataBarrier](asc_sync_data_barrier.md)：阻塞后续指令执行，直到所有之前的内存访问指令执行结束，用于解决Scalar的内存依赖。
- [AscSyncInterArrive](asc_sync_inter_arrive.md)：与asc_sync_inter_wait配合使用，通过flag_id关联，用于不同group之间block的信号同步。
- [AscSyncInterWait](asc_sync_inter_wait.md)：与asc_sync_inter_arrive配合使用，等待组间block到达flag_id对应的同步点后解除阻塞。
- [AscSyncIntraArrive](asc_sync_intra_arrive.md)：与asc_sync_intra_wait配合使用，用于block内不同Core的流水线同步，向同步寄存器发送同步信号。
- [AscSyncIntraWait](asc_sync_intra_wait.md)：与asc_sync_intra_arrive配合使用，等待block内其他Core发送同步信号后解除阻塞。
- [AscSyncMte2](asc_sync_mte2.md)：针对PIPE_MTE2流水线执行同步操作，保证其前序指令全部完成后，PIPE_MTE3与PIPE_V流水线的后续指令才能执行。
- [AscSyncMte3](asc_sync_mte3.md)：针对PIPE_MTE3流水线执行同步操作，保证其前序指令全部完成后，PIPE_MTE2与PIPE_V流水线的后续指令才能执行。
- [AscSyncNotify](asc_sync_notify.md)：与asc_sync_wait配对使用，实现AI Core内部不同流水之间的同步控制，当源流水前序指令完成后设置标志位。
- [AscSyncPipe](asc_sync_pipe.md)：完成同一流水线内的同步控制，保证前序指令的所有数据读写完成后，后序指令才能开始执行。
- [AscSyncSubblockArrive](asc_sync_subblock_arrive.md)：与asc_sync_subblock_wait配合使用，通过flag_id关联，用于组内不同subblock间的信号同步。
- [AscSyncSubblockWait](asc_sync_subblock_wait.md)：与asc_sync_subblock_arrive配合使用，等待组内subblock到达flag_id对应的同步点后解除阻塞。
- [AscSyncVec](asc_sync_vec.md)：针对PIPE_V流水线执行同步操作。无入参函数原型已废弃，请使用asc_sync执行所有流水线的同步。
- [AscSyncWait](asc_sync_wait.md)：等待特定的同步标志，阻塞当前流水线直到条件满足，与asc_sync_notify配对使用。
- [AscUnlock](asc_unlock.md)：用于同步AI Core内部异步流水的指令，在当前流水的前序指令执行完成后根据mutex_id释放对应Mutex。须与asc_lock成对使用。
- [核间同步能力概述](inter_core_sync_overview.md)：核间同步的使用场景通常是：一个核计算所依赖的数据，需要等待其他相关核的计算全部完成后才能获取。
- [核内同步能力概述](intra_core_sync_overview.md)：AI Core内部的执行单元（如MTE2搬运单元、Vector计算单元等）以异步并行的方式运行，在读写同一存储资源时可能存在数据依赖关系。为确保数据一致性及计算正确性，需通过同步控制协调操作时序。
- [系统同步能力概述](system_sync_overview.md)：在编程中，同步是指协调多个执行单元（如线程、进程等）对共享资源的访问顺序和时机来确保程序的正确性。如果没有同步来严格控制执行顺序，则会出现结果不一致、行为不可预测等多种问题。

- [关键特性说明](key_features.md)：如图1所示，本章节将配合时序图介绍四对核间同步接口支持的四种同步控制模式各自实现的原理。
