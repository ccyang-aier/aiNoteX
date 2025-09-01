超长上下文带来了与KVCache相关的挑战，KVCache不断增长的显存占用显著影响了推理服务的各项性能，如吞吐、首Token延迟等。 当前针对超长上下文的一系列有关KVCache的探索方向包括：

**KVCache驱逐(KVCache Eviction)**

1. 思想：既然不能存储所有历史信息，那就选择性地"遗忘"一些内容；
2. 典型：
    - StreamingLLM：保留最开始的几个token（attention sinks）+ 最近的N个token
    - Less：通过恒定大小的低秩缓存累计被逐出的token所包含的信息，并允许对这部分信息进行访问
    - H2O：通过累积Attention Score去决定哪些token重要，并淘汰"不重要"的
    - SnapKV：利用局部窗口选择重要token
3. 局限性：
    - 信息永久丢失：被淘汰的token无法恢复
    - 精度下降：token选择不当导致关键信息丢失，精度下降
    - 长上下文：早期轮次的对话中被认为不重要的信息，可能在后续变得关键

**KVCache卸载(KVCache Offload)**

1. 思想：既然KVCache的存储开销越来越大，而GPU显存又不能一味提升，那就把KVCache卸载到空间更大、但带宽更低的CPU内存甚至SSD等存储介质中；
2. 典型：
    - LMCache
    - Mooncake Store
3. 局限性：
    - PCIe带宽限制：CPU内存或更下级存储的带宽远不如GPU显存，需要高度优化的系统设计去尽力隐藏这部分开销，如异步、并行、预取等

**稀疏注意力(Sparse Attention)**

1. 思想：不是删除信息，依然保留完整KVCache，但选择性地计算注意力，以减少计算开销；
2. 典型：
    - Quest：将token分组为页面，选择注意力分数最高的页面
    - Loki：用PCA降维，在低维空间中选择重要token
    - InfiniGen：将完整的KVCache卸载到CPU，并通过SVD预定义的投影来预取关键Token；
3. 局限性：
    - 虽然计算加速了，但内存占用没有解决
    - 容易选不好

**量化**

1. 略

### ShadowKV的实现

**ShadowKV的设计在本质上也是一种稀疏注意力方向上的探索。**

不过，它与传统稀疏注意力的实现不同，传统稀疏注意力需要在GPU显存驻留完整KVCache，导致长上下文时的显存高占用问题得不到解决，ShadowKV通过将K(Pre-RoPE)矩阵低秩压缩，将V矩阵卸载到CPU，从而实现更少的GPU显存空间占用。

同时，ShadowKV选择将低秩K(Pre-RoPE)矩阵压缩，而不是近似，从而实现更高的准确度，将压缩后的低秩K(Pre-RoPE)矩阵重建与V矩阵的获取通过CUDA多流并行实现，从而隐藏了K的重建过程，以最大限度地减少解码延迟。

总结来说，ShadowKV在：

1. 减少GPU显存占用
2. 在有限的稀疏KVCache预算内保持高准确度
3. 减少推理延迟 通过完整良好的系统设计取得了一种均衡。

#### K(Pre-RoPE)的低秩性

![](chrome-extension://difoiogjjojoaoomphldepapgpbgkhkb/imgs/Pasted%20image%2020250620101454.png) **图一**

1. 随着奇异值索引逐渐增大，K(Pre-RoPE)的相对奇异值衰减最快，具有更强的低秩特性，这也就意味着K(Pre-RoPE)可以用更小的结构去近似，从而实现降维或压缩；

**图二**

1. 基础上下文(16K)或在基础上下文之上继续拓展2K上下文，K(Pre-RoPE)的低秩子空间具有高度相似性，即共享低秩子空间，但不同序列的上下文相似度更低且波动较大，表明不同序列的K(Pre-RoPE)的低秩子空间差异较大，无法共享，模型的K(Pre-RoPE)低秩结构是针对单个序列内部的连续上下文构建的；
2. 随模型层数变化，Context与Extended Context的相似度变化不大，说明该低秩性质贯穿大多数中间层；

**图三**

1. 随着序列增长，64K到384K，Attention计算耗时以及FFN计算耗时均增长显著，SVD计算耗时上升缓慢，说明SVD的计算开销可控，尤其在使用低秩截断时，开销会进一步降低；
2. 随着上下文长度增加，低秩比例也会进一步降低，说明随着序列更长，K(Pre-RoPE)矩阵的有效秩越来越低，低秩性质更明显；

总结：

1. K(Pre-RoPE)的奇异值衰减最快，表明K(Pre-RoPE)具有更强的低秩性，利好压缩；
2. K(Pre-RoPE)在单个序列内部保持低秩特性稳定，但不同序列之间显现差异，无法共享低秩子空间，同时，模型层数对低秩性影响不显著；
3. K(Pre-RoPE)的低秩性结合SVD低秩截断等技术可以有效压缩，同时，SVD的开销在长上下文时远低于FFN和Attention，可以忽略不记；

#### KVCache选择策略

ShadowKV提出了一种精确的KV选择方法，能够在保持准确度的同时，尽可能减少选定token的数量，即减少了Top-K中的K值，ShadowKV称其为稀疏预算(sparse budgets)，这个值一般在1.56%；

大致来说，ShadowKV发现，K(Post-RoPE)与相邻token有更高的相似度，从而能够通过块级近似来选择重要的token，而极少量的异常块（0.3%）更难近似，它们被持久化在GPU显存上以保持准确度。

更深入的，这种选择策略基于两个核心洞察：

1. **空间局部性**：大多数经过RoPE处理后的key向量在空间上具有局部性特征，即：
    - 相邻token的key向量具有很高的余弦相似度
    - 可以将连续的8个token作为一个chunk（块）
    - 每个chunk内的key向量与该chunk均值的相似度很高（除了少数异常值）
    - ShadowKV在128K长度的上下文中进行实验，将post-RoPE key按8个token分组，计算每个chunk内key向量与chunk均值的最小余弦相似度，结果显示大部分chunk内相似度很高，印证了该结论
2. **时间局部性**：相邻解码步骤选择的KV对有很高的重复率，也就是说，当前step选择的重要KV对，在下一个step中可能仍然重要；

基于此，ShadowKV通过chunk均值作为"地标"（landmark）来近似整个chunk的注意力计算，而对于可能包含关键信息而导致难以近似的异常chunk，则直接保留完整KV Cache。

当计算注意力时，Q只需要与选择后的Landmarks进行注意力计算即可，而不需要再与完整的K进行计算。

#### 执行流程

![](chrome-extension://difoiogjjojoaoomphldepapgpbgkhkb/imgs/Pasted%20image%2020250620123123.png)

如图：

1. Prefill阶段：
    - Value矩阵卸载到CPU
    - K(Pre-RoPE)矩阵由于具有更好的低秩性，进行SVD压缩后驻留在GPU中；
    - K(Post-RoPE)会以8个token为一组chunk，计算key向量均值，得到Landmarks地标；
    - 难以近似的异常值(Outliers)完整驻留在GPU上；
2. Decode阶段：
    - 对于Q，先基于Landmarks计算近似的注意力得分，通过识别得分最高的前K个chunk，从CPU中检索对应的Value，并根据低秩投影重建K，从而隐藏K的重建过程；
    - 得到这部分KV后，再结合Outliers进行稀疏注意力计算；
    - 又因为，在前面的洞察中，K(Pre-RoPE)在单个序列内部保持低秩特性稳定，因此Decode阶段生成的K可以与Prefill阶段的K共享低秩子空间，即共享同一个低秩基矩阵，从而进一步提高性能；

#### 实验评估

1. ShadowKV可以将KVCache的GPU显存占用减少60%以上，而不会显著降低准确性；
2. ShadowKV可以支撑更大的推理批次，提升6倍以上，并将吞吐提升3.04倍；

#### 实现

ShadowKV提供了论文的代码实现，代码仓：[https://github.com/ByteDance-Seed/ShadowKV](https://github.com/ByteDance-Seed/ShadowKV)

#### 技术项目复现可行性分析

ShadowKV无法利用现有推理引擎如vLLM或SGLang直接运行，仅提供了代码Demo，模型支持有限，当前仅支持：

1. Llama-3-8B-1M
2. GLM-4-9B-1M
3. Llama-3.1-8B
4. Yi-9B-200K
5. Phi-3-Mini-128K（仅支持NIAH）
6. Qwen2-7B-128K（仅支持NIAH）

若要在vLLM中支持集成ShadowKV，可能需要：

1. 增强ShadowKV的能力，以支持更多模型，这需要深入分析ShadowKV的Demo代码，并进行更多的拓展；
2. 在vLLM的注意力实现框架中新增一种注意力后端，即ShadowKVBackend，该Backend需要实现AttentionBackend的所有方法，提供配置参数以支持覆盖vLLM的默认Attention选择行为；
3. 最关键的难题：**需要解决ShadowKV与PagedAttention和连续批处理的设计理念冲突问题以及支持ShadowKV的混合KVCacheManager**；

**冲突一：** vLLM基于PagedAttention，实现了统一的KVCache Block管理机制，KV Cache具有统一的block布局，而ShadowKV需要的是KV Cache的异构存储结构。 比如：

```python
# vLLM源码: 所有KV数据都按固定大小的block组织
self.block_table = torch.zeros(
    (max_num_reqs, max_num_blocks_per_req),  # 每个请求都有统一的block布局
    device=self.device, dtype=torch.int32,
) 

# ShadowKV需要的是异构存储结构，每个序列都不同
seq1_structure = {
    'compressed_k_blocks': [b1, b3, b7],      # 压缩K，不规则分布，K在GPU
    'landmark_blocks': [b1, b5, b9],          # landmarks，稀疏分布  
    'outlier_blocks': [b2, b4],               # outliers，异常分布
    'cpu_v_blocks': [all_blocks_on_cpu],      # V全部在CPU
}

# 每个序列可能是完全不同的分布模式
seq2_structure = {
    'compressed_k_blocks': [b2, b4, b6, b8], 'landmark_blocks': [b1, b3, b7, b11],
    'outlier_blocks': [b5], 'cpu_v_blocks': [all_blocks_on_cpu],
} 
```

**冲突二：** vLLM基于slot_mapping统一寻址，将所有活跃请求当前要处理的token拼成一个长序列（one big tensor），再把它们各自的KV cache（past_key_value）也按请求顺序拼成一个大的KV cache张量K_all、V_all，然后一次性送进模型，借助slot_mapping在前向结束后把结果再分发回各自的请求； 然而，ShadowKV的异构存储形式无法完全复用原有的vLLM slot_mapping机制，需要在此之上拓展甚至一套新的逻辑去支持连续批处理；

**冲突三：** vLLM最近一个月刚刚支持了HybridKVCacheManager，仅支持滑窗注意力，想要在vLLM中使用ShadowKV，还需要对HybridKVCacheManager进行增强；

**总结（快速落地产品可能性较低）：**

1. **ShadowKV的Demo实现在模型支持度上有限，几乎不支持任何主流模型，需要一定的拓展工作开发量；**
2. **ShadowKV集成到vLLM等主流推理引擎难度较大，需要开发新的Backend接入vLLM，同时，需要重构或者拓展vLLM的block_table、hybrid_kvcache_manager、slot_mapping等原有实现以兼容PagedAttention和连续批处理等核心特性, 需要工作量较大；**