将下列python版microgpt，进化到：

单文件、零外部依赖（仅用标准库：math/random/os/array/multiprocessing; 默认 PyBackend），但架构上支持一键切 torch（TorchBackend 可选）。
严格模块边界：用 # ===================== SECTION: ... ===================== 分隔；将来每个 SECTION 可直接拆成独立文件
纯 Python 张量：array('f') 连续存储；grad buffer 同形预分配；支持 view/slice/stack（一般 strides，非 contiguous 会保守触发 contiguous() 物化）
Tape 非递归 backward：显式 op 列表，反向遍历
KV cache 预分配：推理 step 模式每层预分配 [Tmax, n_head, head_dim] 的 K/V
Embedding backward：scatter-add 到 W.grad（按行累加）
注意力：
训练：full-seq attention（scheme=2），稳定 softmax，workspace 复用（probs/dlogits 复用，不重算），weighted-sum 融合
推理：step attention 内核（融合 softmax+weighted sum，workspace 复用，KV cache 预分配）
Adam：纯 Python buffers 预分配
PyKernels 子模块区：所有 fused kernels 集中放这里；PyBackend 调用它们
“更像 X @ W^T”的批量内核：提供 linear_fwd/bwd（B,T,C × O,C）批量路径（虽是 Python 循环，但接口与未来 Torch/CPP 对齐）
QKV 一次 GEMM：用 wqkv 合并权重，single matmul 产出 QKV
FFN 融合：fc1 + relu + fc2 组合接口（仍是 Python 内核，接口对齐）
multiprocessing 数据预取：提供可选的预取器（默认关闭，Windows 需要 if __name__ == "__main__":）

把 view/slice/stack 做成通用 Tensor API
把训练 attention 的 workspace 结构化
把每个 SECTION 的对外 API（函数签名/数据结构）收紧成标准的模块接口、未来可直接拆文件并单元测试
Tensor 的通用多维 indexer、完整的 slice（步长/多维切片）、stack(dim!=0)
训练端 residual/add、split/merge 等也全部做成标准 op + tape 记录
FFN 真正做成一个 fused kernel 接口（前向一次、反向一次）
attention 训练端 workspace 进一步分块（按层复用/按 batch 复用），避免每步重复分配
把 TorchBackend 的接口桩补上（不导入 torch，但给出“需要实现的函数签名/张量布局约定”）

Tensor.grad 对 view 的策略：
A. 只允许 leaf 参数有 grad （默认）
B. view 也有 grad，并正确映射回 base（需要实现通用 strided grad 累加/或者强制 view 在 tape 中物化为 contiguous）

cache 布局：
(Tmax,n_head,B,head_dim)　（但不能强依赖CUDA kernel)

把 CE + RMSNorm backward + activation grads（策略A下的非叶子 grad buffer 管理）一次性补齐
要有下面两个 SECTION（保持接口收紧）： 
1) PyKernels.logsoftmax_nll_fwd/bwd（或 fused cross_entropy_fwd/bwd），并在 Backend 里提供 cross_entropy(...) 标准 op；
2) 补齐 rmsnorm_bwd 的正确实现

实现真正端到端正确反传
把下面“缺口”补上：
标准 op + tape 记录：residual/add、split/merge、stack、slice/view 的 grad 规则
attention workspace 更细分复用（按层/按 batch 复用、避免每步重复清零/分配）
策略 B（view 也有 grad） 的实现方式（strided scatter-add 或强制物化）
TorchBackend 需要实现的函数签名更完整（含 KV cache 布局）
CE + RMSNorm + activation grads 的非叶子 grad buffer 管理更统一

把 merge/split（例如把 Q,K,V merge 回 QKV）也做成标准 op + tape（目前 split 用 slice 记录了，merge 还没单独封装）
进一步把 workspace 复用做成“按层对象 + 按 batch 对象”，并加“dirty flag”避免每步清零整个大 buffer
策略A/B切换：A 模式下强制 view 在 tape 中物化 contiguous，并给出明确的性能/正确性说明

然后继续把它“收紧成可单元测试的模块接口”
