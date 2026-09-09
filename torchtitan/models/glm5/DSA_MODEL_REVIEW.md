# GLM DSA 模型审阅

未迁入旧分支 ops/ops_candidate，未实现 absorbed MLA。已接入现有分布式边界，设备验证待完成。

## 入口与边界

- `glm5_configs["debugmodel"]()`：默认使用 Glm5FlexAttention；旧 dense attention 实现已删除。
- `glm5_configs["dsa_debugmodel"]()`：直接复用 debugmodel 工厂，不重复替换 inner attention。
- `glm5_configs["shared_dsa_debugmodel"]()`：相同稀疏计算，index_sources=(0,0,2,2,4,4,6,6)。
- Trainer 通过 `--config glm5_shared_dsa_debugmodel` 选择共享索引模型。该入口用于正式 smoke 验证，不代表尚未执行的多卡拓扑已经通过。

## 数学与代码阅读顺序

1. dsa.py 的 Glm5DsaIndexer / DSAIndexerTopK：保持原有冻结 indexer、FP32 排名公式。
2. Glm5Attention：保持展开式 MLA；没有传入 indices 时计算本层 index，否则复用。
3. dsa.py 的 selected_mask：indices 与文档/因果约束相交，负数无效，重复项按集合处理。
4. build_dsa_block_mask：按块汇总合法位置，块内 mask_mod 精确限制 token。尾部补齐不改变有效位置。
5. Glm5FlexAttention：调用公共 FlexAttention；不手写 kernel。当前支持零 attention dropout。
6. Glm5Model：每次 forward 新建局部索引字典，显式传入各层，不跨 batch 缓存。

这仍保留 O(Q*K) mask 内存和全局 indexer 打分；不宣称全链路线性复杂度或必然加速。
`create_dsa_causal_mask` 是仍在使用的因果/文档边界约束，不是旧 dense attention：
它先限制 indexer 排名，再与选中位置相交形成 FlexAttention 的 BlockMask。
矩阵乘法是否省下足够工作取决于 block 覆盖率。

## 跨层索引与权重

只共享所选位置，各层 Q/K/V 都重新计算。来源必须是自身或更早的 producer。
为保留现有 checkpoint 参数结构，本轮保留共享层的冻结 indexer 参数，但不执行它。
这不是最终 PR 的共享层权重裁剪方案；adapter 和 FLOPs 统计需要在后续配套阶段处理。
跨 PP stage 的索引来源缺失在 parallelize 入口显式拒绝；不得用消费层重新计算代替来源层索引。
未实现 indexer 辅助训练目标，不可宣称从随机初始化完整训练 indexer。

## 验证

`python -m pytest tests/unit_tests/test_glm5_sparse_model.py -q`

包括 CPU mask 集合测试和 CUDA FlexAttention 前向/梯度对照。当前本地默认 Python
没有 torch，未执行运行测试；静态语法检查不能证明设备正确性。

官方参考：
[DeepSeek V4 attention](https://github.com/pytorch/torchtitan/blob/d263ca0a1b569ed198b9943b6e8c2117a61d8843/torchtitan/models/deepseek_v4/attention.py)。
只参考 block-list + token mask 的结构，不引入 V4 压缩序列、sink 等数学定义。

## 文件职责与待验项目

dsa.py 包含 indexer、top-k、文档因果 mask、BlockMask 与 FlexAttention。
model.py 保留展开式 MLA、block、跨层传递，以及旧名字的兼容导出。
官方 V4 将 Indexer 放在 compressor.py；GLM 不使用该序列压缩器，因此集中在 dsa.py。
[官方 Indexer](https://github.com/pytorch/torchtitan/blob/d263ca0a1b569ed198b9943b6e8c2117a61d8843/torchtitan/models/deepseek_v4/compressor.py)。

仍待处理：共享层 indexer 权重裁剪、共享配置 FLOPs 估算、旧 dense 测试适配、
分布式设备验证与 Turbo 行为适配。兼容导出不代表这些路径已通过运行验证。

## 分布式接入

- TP：MLA 上投影按 head 分片，输出投影做归约；indexer 与共享索引在 TP 上复制。BlockMask 在 inner attention 的 local_map 内由普通本地 Tensor 构造。
- SP：attention 输入恢复 TP 复制，输出投影恢复 token 分片；共享索引不跟随 hidden states 在 TP 上切分。
- CP：连续 token 分片；indexer gather 全局 key，attention 使用带反向传播的 K/V gather。mask 是本地 query × 全局 key，top-k 值保持全局 key 编号。不是 ring attention，也未减少 K/V gather 通信量。
- EP：沿用公共 MoE dispatcher、专家布局与 FSDP 包装；DSA 索引不是 MoE 专家路由，不加入 EP token dispatch。
- PP：独立 indexer 的层沿用原阶段接口；共享组必须完整保留在同一 stage。跨 stage 共享尚不支持。
- DP/FSDP：沿用公共参数分片；索引只在本次 forward 中存活，不注册为 buffer 或 checkpoint 状态。

local_map 输入 placement 按位置对应，因此内部调用保留五个位置张量参数（Q/K/V/mask/indices），scale 使用关键字。公共参数名仍是 attention_masks。

新增测试覆盖 CP 本地 query/全局 key 的 mask 语义和 PP 共享组边界；本地无 torch，未执行这些测试。仍需 GPU 单卡、TP、CP、TP+CP、EP、TP+EP、PP 和 FSDP 组合的前后向验证，不能据此宣称所有拓扑已跑通。
