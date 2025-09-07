# 基于“H₂O: Heavy-Hitter Oracle”的文字矩阵可视化演示：一次从输入到生成的完整推理流程（修正版：LaTeX/Typora 风格）

下面基于论文“H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models”的核心思想，构造一个可运行于“纸上”的文字驱动可视化示例。我们用小尺寸的矩阵与步进式的 Softmax 注意力计算，逐步展示在 KV cache 限制下，H₂O 如何用“重击手”Heavy-Hitter（简称 HH）与“最近”Recent 两类 token 的组合，完成低成本、低丢失率的逐步生成，同时解释其原理。

参考来源摘自你提供的论文内容（如观察与算法定义、动态子模形式、算法 1 的本地统计打分流程、保持 HH 与最近 token 的组合策略等）。

---

## 场景设定

- 模型：单头自注意力（为便于演示），隐藏维度 $d=3$。
- 序列步数：将演示从已有 4 个上下文 token 开始，逐步生成第 5 个 token。
- KV cache 预算：$k=3$，即最多只保留 3 个历史 token 的 KV（远小于全部长度）。
- H₂O 策略：每步只保留
  - 一部分为 Heavy-Hitters（累计注意力较高的历史 token），
  - 一部分为最近的 token（时序上越靠近当前步，语义相关性通常更强）。
- 矩阵说明：
  - Query 行向量 $Q_{i,*}\in\mathbb{R}^{1\times d}$
  - Key 矩阵 $K_{\le i,*}\in\mathbb{R}^{i\times d}$
  - 用简小整数构造，方便手算 Softmax，可读可查。

为强调流程，本示例只展示“注意力得分与归一化”的关键环节，不展开 Value 聚合与输出层（其对理解 H₂O 的缓存策略非必须）。

---

## 步 0：已有上下文与累计注意力初始化

- 已有上下文 token 索引：$[1,2,3,4]$
- KV cache 预算：$k=3$，意味着任一时刻最多存 3 行 K/V。
- 定义“累计注意力分数”向量 $\mathrm{acc}\in\mathbb{R}^{4}$，初始为全 0：
$$
\mathrm{acc}=\begin{bmatrix}0 & 0 & 0 & 0\end{bmatrix}
$$

- 初始（示意）各 token 的 Key：
$$
K_{\le 4,*}=
\begin{bmatrix}
\text{tok 1}\\
\text{tok 2}\\
\text{tok 3}\\
\text{tok 4}
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 1\\
0 & 1 & 1\\
1 & 1 & 0\\
1 & -1 & 0
\end{bmatrix}
$$

- 将在接下来的第 5 步生成中使用查询向量：
$$
Q_{5,*}=\begin{bmatrix}1 & 1 & 0\end{bmatrix}
$$

---

## 步 1：第 5 个 token 的注意力打分（全量视图，未裁剪）

- 原始打分（未限缓存）：先计算 $Q_{5,*}K_{\le 4,*}^{\top}\in\mathbb{R}^{1\times 4}$
$$
Q_{5,*}K_{\le 4,*}^{\top}
=
\begin{bmatrix}
1 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 1\\
0 & 1 & 1\\
1 & 1 & 0\\
1 & -1 & 0
\end{bmatrix}^{\top}
=
\begin{bmatrix}
1+0+1 & 0+1+1 & 1+1+0 & 1-1+0
\end{bmatrix}
=
\begin{bmatrix}
2 & 2 & 2 & 0
\end{bmatrix}
$$

- Softmax 权重（演示中略去缩放和 mask）：
$$
\mathrm{softmax}\!\left(\begin{bmatrix}2 & 2 & 2 & 0\end{bmatrix}\right)
=
\frac{1}{e^2+e^2+e^2+e^0}
\begin{bmatrix}
e^2 & e^2 & e^2 & 1
\end{bmatrix}
\approx
\frac{1}{3e^2+1}
\begin{bmatrix}
e^2 & e^2 & e^2 & 1
\end{bmatrix}
$$

- 用“文字版矩阵”表达（行是“被注意的历史 token”，列是“权重”）：
$$
\left[
\begin{array}{c|c}
\text{tok 1} & \omega_1\\
\text{tok 2} & \omega_2\\
\text{tok 3} & \omega_3\\
\text{tok 4} & \omega_4
\end{array}
\right]
=
\left[
\begin{array}{c|c}
\text{tok 1} & \frac{e^2}{3e^2+1}\\
\text{tok 2} & \frac{e^2}{3e^2+1}\\
\text{tok 3} & \frac{e^2}{3e^2+1}\\
\text{tok 4} & \frac{1}{3e^2+1}
\end{array}
\right]
$$

可见 tok1/2/3 权重相同且占绝大头，tok4 权重较小。

- 更新累计注意力 $\mathrm{acc}$（本地统计，论文建议每步累加先前 token 的注意力分数）：
$$
\mathrm{acc}\leftarrow \mathrm{acc} + \begin{bmatrix}\omega_1 & \omega_2 & \omega_3 & \omega_4\end{bmatrix}
\approx
\begin{bmatrix}
\frac{e^2}{3e^2+1} &
\frac{e^2}{3e^2+1} &
\frac{e^2}{3e^2+1} &
\frac{1}{3e^2+1}
\end{bmatrix}
$$

---

## 步 2：H₂O 的“重击手 + 最近”组合选择（KV 缓存限制）

- 预算 $k=3$，需要从 $\{1,2,3,4\}$ 中保留 3 个。
- 重击手（HH）的判定：选累计注意力最高的若干个。此时
$$
\mathrm{acc}(1)=\mathrm{acc}(2)=\mathrm{acc}(3)\gg \mathrm{acc}(4)
\Rightarrow \text{HH 候选优先：}\{1,2,3\}
$$

- 最近（Recent）：通常保留最近的一个或多个（如最近 1 个）。最近者为 tok4。

- H₂O 的关键点：在固定预算内，动态平衡“全局重要性（HH）”与“局部时序相关性（Recent）”。一种简单配比（论文与实现建议）是“一半 HH + 一半 Recent”，这里 $k=3$ 无法刚好对半，我们采用：
  - 2 个 HH：从 $\{1,2,3\}$ 里任选两个（因三者相等，我们就取 $\{1,2\}$）
  - 1 个 Recent：$\{4\}$

- 因此本步缓存集合 $S_4$ 选为
$$
S_4 = \{1,2,4\}
$$

被淘汰：tok3（注意：因 tok1/2/3 得分相同，谁被淘汰主要看实现的并列打破规则；本文为示例）

- 可视化“选择矩阵”（选中记 1，未选记 0）：
$$
\left[
\begin{array}{c|c}
\text{tok 1} & 1\\
\text{tok 2} & 1\\
\text{tok 3} & 0\\
\text{tok 4} & 1
\end{array}
\right],\quad |S_4|=3
$$

---

## 步 3：用裁剪后的缓存再计算第 5 步注意力（与论文公式一致）

按照论文定义（有裁剪的生成过程），只对 $S_4$ 中的 key 参与注意力，其他位置当作 0 并在归一化中扣除：

- 子矩阵 $K_{S_4,*}$：
$$
K_{S_4,*}=
\begin{bmatrix}
\text{tok 1}\\
\text{tok 2}\\
\text{tok 4}
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 1\\
0 & 1 & 1\\
1 & -1 & 0
\end{bmatrix}
$$

- 打分：
$$
Q_{5,*}K_{S_4,*}^{\top}
=
\begin{bmatrix}1 & 1 & 0\end{bmatrix}
\begin{bmatrix}
1 & 0 & 1\\
0 & 1 & 1\\
1 & -1 & 0
\end{bmatrix}^{\top}
=
\begin{bmatrix}
2 & 2 & 0
\end{bmatrix}
$$

- Softmax（仅在被保留的 3 个位置归一化；被淘汰的 tok3 视为被 mask）：
$$
\mathrm{softmax}\!\left(\begin{bmatrix}2 & 2 & 0\end{bmatrix}\right)
=
\frac{1}{e^2+e^2+e^0}
\begin{bmatrix}
e^2 & e^2 & 1
\end{bmatrix}
=
\frac{1}{2e^2+1}
\begin{bmatrix}
e^2 & e^2 & 1
\end{bmatrix}
$$

- 文字矩阵（只有被保留的 3 项有权重；被淘汰的 tok3 视作 0 且不参与分母）：
$$
\left[
\begin{array}{c|c}
\text{tok 1} & \frac{e^2}{2e^2+1}\\
\text{tok 2} & \frac{e^2}{2e^2+1}\\
\text{tok 3} & 0\quad(\text{evicted})\\
\text{tok 4} & \frac{1}{2e^2+1}
\end{array}
\right]
$$

可见，相比“全量注意力”，权重分配几乎不变（tok1/2 仍占大头、tok4 次之），因此这一步的输出近似不受影响——这正是 H₂O 要达到的“低 miss 率”目标。

- 同时，更新累计注意力（供后续步继续决策）：
$$
\mathrm{acc}\leftarrow \mathrm{acc} + 
\begin{bmatrix}
\frac{e^2}{2e^2+1} &
\frac{e^2}{2e^2+1} &
0 &
\frac{1}{2e^2+1}
\end{bmatrix}
$$

---

## 步 4：生成第 5 个 token 后，更新缓存集合到 $S_5$

- 新生成 token 的索引为 5，它的 KV 必须加入缓存。
- H₂O 的执行步骤（呼应论文算法 1 的第 11-13 行）：
  1) 暂时把新索引合并到候选：$G_5=S_4\cup\{5\}=\{1,2,4,5\}$
  2) 用打分函数 $F_{\mathrm{score}}$（本地累计注意力）在 4 个里删除 1 个，使大小回到 $k=3$，即
     $$
     u=\arg\max_{v\in G_5} F_{\mathrm{score}}\big((G_5\setminus\{v\})\big)
     $$
     直观等价于：删掉“被删后剩余集合的累计注意力总和最小化”的那个 $v$，也就是删掉累计注意力最低、或对“最近性+重要性”贡献最弱的那个。
  3) 形成新的 $S_5=(G_5\setminus\{u\})$。

- 在我们的示意中，tok4 既非 HH（累计注意力最低），但它是“最近”之一；而 tok1/2 是显著的 HH。此时常见的策略是优先保 HH 与最近“最新的 5”，因此倾向于淘汰 tok4：
$$
S_5=\{1,2,5\}
$$

- 文字矩阵（选择）：
$$
\left[
\begin{array}{c|c}
\text{tok 1} & 1\\
\text{tok 2} & 1\\
\text{tok 3} & 0\\
\text{tok 4} & 0\\
\text{tok 5} & 1
\end{array}
\right],\quad |S_5|=3
$$

---

## 小结：为何这能“省内存、低丢失、低成本”

- 省内存：预算 $k\ll n$。如上例 $k=3$ 对 $n=5$；论文实验表明仅用 20% 的 KV 预算即可在广泛任务上接近满缓存准确率（见图 2、图 4 与表格）。
- 低丢失率（高质量）：注意力矩阵在推理时高度稀疏（论文观测 $>95\%$ 的稀疏性），而累计注意力展现“长尾+重头”的幂律——少数 token 贡献主要注意力（重击手 HH）。H₂O 用贪心地保留 HH + 最近，从经验与理论上都维持了输出质量。
  - 理论上：论文将其表述为动态子模最大化问题，在温和假设下，贪心具有近似最优保证（如定理 4.4 的 $(1-\alpha)(1-1/e)$ 型下界）。
- 低成本：算法每步只用“本地”统计（到当前步为止的注意力累积），无需看未来 token，即可近似全局最优；工程上还能以固定内存块和循环队列实现 $O(1)$ 级数据搬移。

---

## 对比“仅保留最近”与“仅保留 HH”

- 仅“最近”：对有长程依赖的任务易退化（论文在多任务上显示显著掉分）。
- 仅“HH”：忽略刚刚生成或即将相关的局部线索，也会掉分。
- 二者结合（H₂O）：在极低缓存预算下仍接近满缓存性能，且可大幅提升吞吐与降低延迟（论文表 3、4、5）。

---

## 额外一轮（可选）演示：进入第 6 步再裁剪一次

若继续生成第 6 个 token（索引 6），重复上述流程：

1) 用 $S_5=\{1,2,5\}$ 参与注意力：
$$
Q_{6,*}K_{S_5,*}^{\top}\ \to\ \mathrm{softmax}(\cdot)\ \to\ \omega^{(6)}_{\{1,2,5\}}
$$

2) 累计 $\mathrm{acc}\leftarrow \mathrm{acc} + \omega^{(6)}$

3) 合并新 token：$G_6=S_5\cup\{6\}=\{1,2,5,6\}$

4) 依据 $F_{\mathrm{score}}$ 删除 1 个，使 $S_6$ 回到 3 个。典型地会保留“最强的 HH + 最近的 6”。

只要每步这样“先合并新、再贪心删 1 个”，就与论文算法 1 的第 11-13 行吻合，且与定义的“单步最多驱逐 1 个，缓存大小恒等于 $k$”的约束一致。

---

## 与论文公式对齐的关键细节提示

- 有裁剪的 Softmax 归一化需要扣除被 evict 的位置（论文中记作对对角项的修正），上面的“只在被保留的集合内归一化”就是这个意思。
- 打分函数 $F_{\mathrm{score}}$ 的具体实例：论文中使用“集合内元素的注意力得分之和”（见算法 1），本示例使用累计注意力近似它的动态版本。
- 动态子模假设与贪心保证：在“注意力打分关于集合是次模”的温和假设下，贪心选择接近最优（论文相应引理与定理）。

---

## 一页速记：H₂O 的核心要点

- 发现：推理时注意力天然稀疏，累计注意力呈幂律，存在少数“重击手”。
- 目标：在固定小缓存预算 $k$ 下，最大化后续注意力可用的信息量。
- 策略：每步用本地统计累加注意力，贪心保留“HH + 最近”，并只驱逐 1 个。
- 效果：在 20% KV 预算下可匹配满缓存性能；加速明显，延迟显著下降；能与量化、稀疏注意等方法正交组合。

---

## 你可以如何用这个演示

- 将上述“文字矩阵”换成你自己的 Q/K 向量（从真实模型采样来的小片段），即可按步复刻 H₂O 的决策过程。
- 若你希望我把这个演示扩展为“可复制运行的最小 Python/Numpy 版脚本”，我可以把全部计算和可视化（ASCII 表格）打包给你，并支持你自定义 $k$、配比（HH:Recent）、并列打破规则等参数。