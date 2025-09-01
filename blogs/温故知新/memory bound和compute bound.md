# 理解 Memory-bound、Compute-bound，以及大模型推理中各阶段的 Bound

## 1) 什么是 Memory-bound？

- **Memory-bound（内存受限）**：性能主要受限于数据搬运带宽/延迟（HBM/GDDR/缓存/总线），而不是算力本身。
- 特征：
    - 算术强度（Arithmetic Intensity，单位“每字节多少次浮点运算”）较低
    - GPU/CPU 计算单元空闲较多，等数据
    - 利用率瓶颈出现在“读写参数、KV缓存、激活”上，而不是矩阵乘法峰值 FLOPs

与之相对的是：

## 2) 与之相对的是哪种 Bound？

- **Compute-bound（计算受限）**：性能主要受限于可用算力（FLOPs），而不是带宽。
- 特征：
    - 算术强度较高
    - 大型 GEMM/卷积等能把 Tensor Cores/ALUs“喂满”
    - 带宽足够，主要瓶颈在算子执行时间

两者的分界通常用“屋顶线模型（Roofline Model）”来判断：给定硬件的“峰值带宽”和“峰值 FLOPs”，当某负载的算术强度低于“拐点”时是 memory-bound，高于拐点时是 compute-bound。

---

## 3) 大模型推理的 prefill 阶段是什么 bound？

- 一般来说，**prefill 通常是 Compute-bound 为主**（在足够大的 batch、长 prompt、较大隐藏维度、合理并行度的前提下）。
- 原因：
    - Prefill 阶段会把整段 prompt（长度 (L_{\text{prompt}})）一次性送入模型，形成“密集的批量 GEMM”。
    - 每一层的注意力中 QK^T 的计算规模大、GEMM 密集，同时 KV 缓存还未太大，读 KV 的带宽压力不算最重。
    - 算术强度较高，能更好地填满算力，贴近 Tensor Core 峰值。

但要注意两种会“偏离”的情况：

- 如果 batch 很小、隐藏维很小，或者层间内核融合差、调度不佳，prefill 可能部分受内存或调度开销影响。
- 在极长 prompt 且实现没有做块化注意力（paged/block-sparse）优化时，注意力的临时内存和访存也会增多，略增加 memory 压力。不过总体仍多见 compute-bound 特性。

---

## 4) 后续生成（decode）阶段是什么 bound？为什么？

- **decode 阶段通常是 Memory-bound 为主**。
- 关键原因：
    1. 单步生成时，每次只新增一个 token，批内有效并行度低，GEMM 规模小，不易“喂满”算力。
    2. 注意力需要读取“历史所有 token”的 KV 缓存（长度 (t) 逐步增大），这会带来线性增长的读取量。
        - 每层对当前 token 的注意力要扫描全部历史 key/value，KV 读放大远大于算术量增长，导致算术强度偏低。
    3. 为了降低延迟，生成阶段通常采用较小 batch、步进式调度，进一步降低了算术密度。
    4. 常见优化（如 paged attention、量化 KV、连续批处理/批合并、speculative decoding）本质上都在减少或隐藏带宽压力、提高算术强度或并行度，以缓解 memory-bound。

在实践中，decode 的热点往往是：

- KV 缓存的读带宽（跨层 × 序列长 × 头数 × 精度）
- 小矩阵乘法的启动/调度开销
- Softmax 与归一化的访存

---

## 一图心智模型（文字版）

- Prefill：大批量、大矩阵、KV 尚小 → 算术强度高 → 多为 Compute-bound
- Decode：小批量、步进生成、反复读历史 KV → 算术强度低 → 多为 Memory-bound

---

## 补充：为什么“算术强度”决定 Bound？

- 算术强度 (I = \frac{\text{FLOPs}}{\text{Bytes Moved}})。
- 设备的“带宽屋顶”是 ( \text{Peak Bandwidth} \times I )，而“算力屋顶”是 ( \text{Peak FLOPs} )。
- 实际可达性能 = min(带宽屋顶, 算力屋顶)。
- Prefill 的 (I) 大（大 GEMM），decode 的 (I) 小（KV 反复读取），所以两者分别靠近不同的屋顶。

---

## 实践中的优化方向

- Prefill（偏 Compute-bound）：
    
    - 提升并行度、核融合、张量并行/流水并行合理切分
    - 保持较大的 batch 与序列块，充分利用 Tensor Core
    - 减少核启动/调度开销
- Decode（偏 Memory-bound）：
    
    - 压缩/量化 KV（FP8/INT8/混合精度），减少字节移动
    - 使用 paged/block-sparse attention、分块缓存、连续批处理（continuous batching）
    - 提高步内并行：多 token 解码（EAGLE、Medusa）、speculative decoding、并行解码树
    - 合并小 GEMM、内核融合，减少访存往返
    - 合理增大 batch 或并发请求以提高算术密度（与延迟权衡）
