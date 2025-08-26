## 1.相关技术背景以及最接近的现有技术
### 1.1　背景技术
随着大语言模型（Large Language Model，LLM）作为推理服务在云计算、公有云AI平台、私有云与超融合基础设施上的广泛部署，多租户已成为核心能力之一，多租户不仅要求在计算资源上的隔离与公平性，还要在数据层面保证不同租户的数据隐私隔离，避免越权访问。

在LLM推理过程中，KVCache扮演着决定性角色：自回归生成每个token需要重复访问历史序列的键值向量，缓存可以避免重复计算，从而显著降低时延、提升吞吐，因此，如何以高效且安全的方式管理KVCache，成为具有工程与研究双重价值的问题。

当前业界高吞吐推理引擎（如vLLM）采用的PagedAttention，以物理Block为基本单位，将所有请求的KVCache统一管理，配合连续批处理与内存复用实现显存与算力的高利用。但是，在多租户场景下，所有租户共享一块物理显存池，虽然逻辑上以Block Table隔离请求，但物理层缺乏明确的租户边界，带来如下风险与挑战：

- 潜在的数据越权访问：在共享策略与前缀复用下，可能出现跨租户的前缀命中或数据复用，从而导致信息泄露隐患；
- 侧信道攻击：攻击者可通过构造探测型prompt，统计命中率、延迟差异，推断另一租户是否近期使用过某类prompt，这可能导致业务机密/提示工程策略泄露等风险；
- 资源争抢与抖动：不同租户的负载波动导致显存压力变化，缺少租户级的资源控制策略，可能造成重要租户的服务质量下降；
- 审计与合规困难：在统一池中定位、追踪某租户占用与缓存来源困难，影响审计与合规要求（特别在金融、医疗、政务领域）；

为解决上述矛盾，本发明提出一种“Zone + PagedAttention”的KVCache动态隔离方法：在不牺牲vLLM高性能特性的前提下，引入逻辑隔离的租户级动态区域（Zone），实现资源的弹性共享与强安全边界控制。

### 1.2　与本发明相关的现有技术
#### 1.2.1　现有技术的技术方案

基于PagedAttention的KVCache管理

PagedAttention是目前最先进的KVCache管理技术之一，被vLLM等主流推理框架广泛采用，其核心思想是将连续的KV缓存按固定大小分割成多个Block，通过Block Table维护逻辑地址到物理地址的映射关系。

具体实现机制：

1. 系统启动时预先分配大块连续的GPU显存作为KVCache池；
2. 将显存按固定大小(如16个token)划分为多个Block；
3. 为每个推理请求维护独立的Block Table；
4. 通过间接寻址实现KV数据的非连续存储；
5. 支持动态分配和释放Block，提升内存利用率，显著减少内存碎片；

如图1.2.1：TODO

推理服务的多租户方案

传统的多租户隔离方案只是在上层限制每个租户的资源配额，但在底层，所有租户共享一个完整的推理服务资源池；

还有一种方式，基于实例级别进行隔离，不同的租户拉起不同的推理服务，从而达到强隔离效果.


#### 1.2.2　现有技术的缺点

基于PagedAttention管理KVCache的局限性

1. **安全隐患**：所有租户共享统一的Block池，虽然通过逻辑Block Table实现了请求级隔离，但在物理层面缺乏租户边界，恶意租户可能通过内存越界访问、缓冲区溢出、测信道攻击等方式获取其他租户的敏感数据；
2. **资源竞争**：资源争抢不可控，无法执行租户粒度的公平性保障或优先级策略；
3. **缓存污染**：不同租户的数据混合存储在同一Block池中，可能导致缓存局部性降低，影响整体性能；
4. **审计困难**：无法准确追踪特定租户的资源使用情况，难以进行精确的成本核算和性能分析；

基于上层限制/实例级隔离的多租户方案：实例级隔离策略会导致资源利用率降低，而通过上层限制租户配额的方式又会存在安全隐患、资源竞争等等诸多问题，本质上还是在一个共享的推理资源池中进行调度。

## 2.本发明技术方案的详细阐述（发明内容）
### 2.1　本发明所要解决的技术问题

大模型在执行推理时，用户输入的相关注意力信息会保存为KVCache缓存在GPU显存中，而类似vLLM等推理引擎在工作时是不会区分上层的请求是来自与哪一个租户的，这就导致不同租户的KVCache资源被相互使用，从而产生一系列的安全隐患、资源竞争、缓存污染等等问题，因此，在推理引擎层级实现多租户，最关键的是要对KVCache的管理进行租户级别的隔离。

基于此，本发明要解决的核心技术问题如下：
- 在不破坏PagedAttention高性能特性的前提下，实现租户级的KVCache强隔离；
- 支持租户级的动态资源伸缩：按需分配Block并支持回收与再利用，避免静态分区浪费；
- 避免跨租户的前缀复用命中，确保前缀缓存仅在租户内生效；
- 提供统一的控制面：可编排每个租户KVCache资源的生命周期、请求调度、资源驱逐与审计统计；
- 在资源紧张时具备可解释、可配置的驱逐策略，兼顾性能与公平性；

### 2.2  本发明的主要发明点概述

本专利的主要发明点如下：

- 引入“Zone（动态区域）”：为每个租户分配逻辑隔离的KVCache域，每个Zone维护独立的KVCacheManager及其BlockTable；
- Zone Controller（区域控制器）：统一维护租户-Zone映射、管理Zone状态机（Active/Idle/Evicted）、跨Zone资源协调与驱逐策略；
- Zone内KVCacheManager：每个Zone拥有独立的KVCache管理器，负责本区域KVCache分配、释放、引用计数、局部驱逐及前缀缓存等实现；
- Free Block Queue（全局空闲池）：以共享空闲池支持各Zone按需动态扩缩，提升资源利用率与弹性；
- 基于Zone的前缀感知：将zone_id纳入根前缀哈希的输入，确保前缀缓存只在同租户内命中，避免越权复用；
- KVCache驱逐策略：优先空闲Zone中“引用计数为0”的Block，再退到本地Zone最低引用的Block，实现资源的高效回收和重用；

### 2.3  本发明技术的具体实现方案
#### 2.3.1 本发明应用的系统架构或场景

本发明适用于多租户LLM推理系统，以及对合规与隐私要求高的行业（金融、医疗、政务）的大模型在线推理场景。

系统架构如下：TODO。

如图，本发明提出一种基于动态区域（Zone）的KVCache多租户安全隔离方法，其核心思想是在全局统一的物理显存之上，为每个租户创建逻辑上完全隔离的、可动态伸缩的租户级Block区域：Zone，每一个Zone自身依然构建于PagedAttention之上，以实现推理引擎的最小化侵入并能完全复用PagedAttention的特性。

核心组件如下：

Zone控制器 (Zone Controller)：负责整个系统的核心调度逻辑；
- 维护租户-Zone映射表（Tenant-Zone Table）；
- 管理每个Zone的生命周期（创建、销毁、状态变更）；
- 将请求下发到指定Zone的KVCacheManager；

Zone：一个与单一租户绑定的逻辑资源单元，每个Zone内部维护自己独立的请求级Block Table和KVCacheManager，记录其逻辑Token到物理Block的映射，从属于一个Zone的请求，其KVCache分配、查找和释放操作，都严格限制在该Zone的管辖范围内；

KVCacheManager：在原生vLLM实现中，KVCacheManager对全局Block进行统一管理、分配和释放，一个vLLM实例对应一个KVCacheManager，而在本专利方案中，一个租户对应一个KVCacheManager，Zone Controller将租户请求下发到对应的KVCacheManager，然后再由该KVCacheManager执行资源的分配和删除等操作；

Free Block Queue：维护空闲Block的链表，作为共享资源，对所有KVCacheManager可见，当某个租户的资源告急时，KVCacheManager将从该Queue中申请一部分Block并添加到Zone中；

#### 2.3.2 本发明的核心装置/网元/组件/软件逻辑单元
Zone Controller

Zone控制器是本发明的核心组件，承担整个系统的调度和管理职责：
- **租户-Zone映射表管理**：维护Tenant-Zone Table，记录租户ID与Zone ID的对应关系，支持动态更新和快速查询；
- **Zone生命周期管理**：负责Zone的创建、初始化、状态转换、资源分配和销毁等全生命周期操作；
- **请求调度分发**：根据请求携带的租户信息，将请求路由到对应的Zone进行处理；
- **资源协调仲裁**：在资源紧张时协调不同Zone之间的资源分配，执行驱逐策略；

状态机控制：
Zone控制器为每个Zone维护一个状态机，如图，它包括三种关键状态：
- Active (活跃): Zone内有正在处理的推理请求；
- Idle (空闲): Zone内没有活跃请求，但KVCache被保留以备后续使用；
- Evicted (已驱逐): Zone因资源回收或长时间未使用，其所有物理Block已被释放，但保留ZoneID与租户的绑定关系；

驱逐策略：
推理请求是一个KVCache逐步增长的过程，当某个Zone的Blocks不够新的请求占用时，KVCacheManager会尝试从Free Block Queue中申请一部分新的Block并追加到Zone中，若Free Block Queue也无法支撑这一次分配，KVCacheManager将触发驱逐，基于Zone的驱逐策略如下：
1. Zone Controller会遍历所有Zone，并尝试找到一个非活跃的Zone；
2. 在非活跃Zone中，优先尝试查找引用计数为0的KVCache Block，并把这些Block释放；
3. 以上均无法满足，尝试在本地Zone下查找引用计数为0的KVCache Block，并把这些Block释放；
同时，还可以基于Zone实现租户的优先级策略，比如优先释放低优先级租户的Block等；



Zone (逻辑资源单元)

Zone是与单一租户绑定的逻辑资源管理单元：

独立Block Table：每个Zone维护独立的Block Table，记录其管理的逻辑Token到物理Block的映射关系，如图：

状态机管理：持有Active、Idle、Evicted三种状态，根据请求活跃度和资源使用情况进行状态转换，比如：当某个租户的请求长时间不活跃时，基于LRU等策略释放其一部分Block资源，从而供其它租户使用，当该租户的非活跃期更久，甚至其所有Block都已经被驱逐时，则将其从租户映射表中移除，此时，该租户的下一次请求将重新在Zone-Controller中注册；

资源边界控制：严格限制Zone内请求的KVCache操作范围，防止跨Zone的数据访问，该策略基于Zone内部的KVCacheManager管理和控制，每个Zone在初始化时，只会分配固定大小的空闲Block资源，而如果需要申请新的资源，必须由Zone内部的KVCacheManager向Zone-Controller进行申请，再由Zone-Controller调度FreBlockQueue获取空闲Block并分配给该Zone。即：每个Zone的Block资源管理的申请以及释放等操作，都需要向Zone-Controller进行报备，从而进行严格的资源边界控制，如图：

![](imgs/Pasted%20image%2020250827002749.png)

请求流程

如图，基于Zone PageAttention，一个租户的完整推理请求的流程大致如下：

1. vLLM引擎调用CUDA显存分配API，统一申请所有剩余显存用于KVCache管理，并基于block槽位数和每Token占用显存大小进行Block划分，这个阶段与原生PagedAttention策略保持一致；
2. 当推理服务开始运行，请求A到达，并携带租户信息，请求注册到Request-Tenant Table用于追踪和管理；
3. 请求发送到Zone Controller，并基于Tenant信息在Tenant-Zone Table中查找Zone，这里分为两种场景：
   命中Zone：直接在该Zone下继续调度、执行请求；
   未命中Zone：说明当前租户已长期未使用推理服务导致其所有历史数据被驱逐，因此触发了Zone Controller的清理机制，或者当前租户第一次使用推理服务，此时，Zone Controller注册一个新的Zone并为其预分配Blocks；
4. 当请求下发到某个Zone后，该请求的所有有关KVCache Block的操作，如分配、释放等，均被该Zone的KVCacheManager接管；
5. 之后，基于该Zone内部执行完整推理请求，该逻辑与原生vLLM保持近似一致；

Zone感知的前缀缓存（Prefix Caching）

为了使Prefix Caching在Zone隔离的场景下继续生效，并且不会造成数据访问越界，本算法还提出了一种侵入性极低并且易于实现的巧妙设计，即在计算Root Block的Hash时，只需要追加Zone Id作为头部节点的区分标识，可以确保只有来自同一个Zone（即同一个租户）的请求，才可能命中前缀缓存，从而在不牺牲性能的前提下保证了缓存的安全性；

Block哈希计算方式如下：
原生：
根Block：hash_value = hash(token_ids)；
后续Block：hash_value = hash(parent_hash, token_ids)

Zone感知：
根Block：hash_value = hash(zone_id + token_ids)；
后续Block：hash_value = hash(parent_hash, token_ids)

#### 2.3.3  发明方案实施例（本发明最优选的实施例）
##### 2.3.3.1  发明方案实施例的技术方案

以租户A为例，完整的推理请求调度流程实施例如下：

统一显存预分配与Block化
- 在vLLM引擎初始化时，通过CUDA接口申请除去模型权重/激活开销外的全部可用显存作为KVCache池；
- 按固定大小划分为Block（例如：每Block容纳NN个token的KV向量，N依赖模型维度与精度）；
- 生成全局Free Block Queue，将所有Block入队；

Zone的创建与绑定
- 当推理服务收到请求A，带有tenant_id，先在Request-Tenant Table登记；
- 请求进入Zone Controller，查询Tenant-Zone Table；
    - 若命中zone_id且Zone非Evicted，则直接路由到该Zone；
    - 若未命中或Zone处于Evicted，创建/重建Zone：
        - 为Zone分配zone_id（可复用旧ID以保留绑定关系）；
        - 进行预热：从Free Block Queue预分配少量Block作为保留水位（可按租户等级配置）；
        - 初始化该Zone的KVCacheManager和Block Table；
        - 更新Zone状态为Active或Idle；

Zone内KVCache管理
- 当请求A在Zone内进行推理：
    - 前缀缓存查询：对请求前缀计算Zone感知的root_hash = hash(zone_id + token_ids)，在Zone的前缀索引中查找命中；
        - 命中：递增相关Block链的引用计数，挂接到本请求；
        - 未命中：按token扩展分配新Block或从Zone-Controller申请Block，构建新的Block链条，并为根建立索引；
    - 若生成过程中因为资源紧张导致Block无可用，则由Zone-Controller触发一次驱逐，并分配驱逐后的Block；

驱逐策略
触发条件：当前Zone申请Block未果（全局空闲不足），或全局监控检测系统水位低于低水位阈值；
策略顺序：
S1：Zone Controller遍历所有Zone，寻找Idle状态且近端时间窗口无活跃请求的Zone，把该Zone中引用计数为0的Block回收（释放到Free Block Queue或保留最小水位后释放其余）；
S2：若S1仍不足，尝试在本地Zone中回收引用计数为0的Block（优先淘汰最早使用的块，可采用LRU/Clock策略）；
S3（可选）：若仍不足，可对本Zone进行“软压缩”策略，例如合并碎片、强制降级不活跃会话等；

Zone状态机与生命周期
- Active：Zone内存在活跃请求；允许扩容（从Free Block Queue取块）；
- Idle：短期无活跃请求，保留一定水位的Block以待复用；超时后可降级为Evicted；
- Evicted：释放全部物理块，保留Zone元数据与tenant绑定；新请求到来时快速重建；


##### 2.3.3.2 发明方案实施例一区别于现有技术的改进之处以及对应的有益效果
基于PagedAttention的租户级逻辑隔离策略（Zone + Zone-Controller）：
改进点：引入Zone作为租户边界，所有KVCache逻辑操作均在Zone内执行；
效果：避免跨租户的KVCache复用，降低数据越权风险或侧信道攻击风险；便于审计和合规；

Zone感知的前缀缓存：
改进点：在根哈希中引入zone_id，保证仅同租户请求可能命中；
效果：保留前缀缓存性能优势同时消除跨租户泄露隐患；

动态伸缩与共享空闲池：
改进点：全局Free Block Queue支持各Zone按需取/还，避免静态切分的浪费；
效果：提升整体资源利用率，适应负载波动，强化弹性能力；

两级驱逐策略（跨Zone优先 + 本Zone兜底）：
改进点：优先回收Idle Zone中未被引用的Block，避免打扰活跃工作集；若不足再本地回收；
效果：减小对热点请求的干扰，提升稳定性与时延可预测性；

### 2.4  关键技术点概括及关键点对应的有益效果
本专利提出一种Zone + PagedAttention的多租户场景下的动态KVCache隔离方法，其特征在于，在统一管理的物理缓存块池之上，为每个租户创建并维护一个逻辑隔离的、包含独立块映射表的动态区域(Zone)，每个Zone依然运行于PagedAttention机制之上，通过Zone的逻辑隔离，极大程度缓解了不同租户间KVCache数据的越权访问和信息泄露等风险，同时更有利于指定租户级的服务质量差异化策略；

基于Zone控制器动态管理所有Zone的生命周期、并将请求下放到Zone；

基于Zone的KVCache Block驱逐方法，其特征在于，当缓存资源紧张时，按序（Idle Zone引用计数为0的Block -> local Zone引用计数最少的Block）释放物理块以供给当前请求；

基于Zone的前缀缓存方法，其特征在于，将与租户绑定的Zone标识符作为输入参数之一，参与到缓存块根内容的前缀哈希计算中；

以上设计既保留了PagedAttention的优势，并且对vLLM等推理框架的侵入性极低，可以无缝在PagedAttention机制之上构建应用；
## 3.发散思维
弹性借贷：当高优先级租户短时高峰，允许从低优先级租户的Idle Zone临时借用部分Block，借贷Block过期自动归还，记录账单用于计量计费；

跨设备Zone迁移：当GPU设备发生故障或需要维护时，可以实现Zone的跨设备迁移，通过序列化Zone的状态信息(包括Block Table、缓存数据等)，并在目标设备上重建Zone，可以实现租户服务的无缝切换；