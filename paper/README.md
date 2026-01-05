# AI 经典论文学习

本目录收集了 AI 领域从 1950 年至 2023 年的 77 篇经典论文，用于系统性学习人工智能的发展历程与核心技术。

## 📂 每个论文文件夹包含

- **论文原文** - PDF 格式的原始论文
- **学习资料** - 相关笔记、解读、参考链接等
- **极简代码实现** - 论文核心思想的最小化代码复现，便于动手理解

## 📚 论文目录

|  # | 年份 | 标题                                                                                                                          | 核心贡献                                              | 领域   |    优先级    |
| -: | ---: | :---------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------- | :----- | :----------: |
|  1 | 1950 | [Computing Machinery and Intelligence](https://courses.cs.umbc.edu/471/papers/turing.pdf)                                        | [起源] 图灵测试，提出"机器能思考吗？"                 | Theory |      P2      |
|  2 | 1955 | [Dartmouth Summer Research Project on AI](https://parhamdata.com/Dartmouth_1955_Proposal_for_AI_Research.pdf)                    | [诞生] "人工智能"概念的正式诞生                       | Theory |      P2      |
|  3 | 1969 | [Perceptrons](https://rodsmith.nz/wp-content/uploads/Minsky-and-Papert-Perceptrons.pdf)                                          | [历史] 证明感知机局限性，引发第一次 AI 寒冬           | Theory |      P2      |
|  4 | 1982 | [Hopfield Network](https://www.dna.caltech.edu/courses/cs191/paperscs191/Hopfield82.pdf)                                         | [理论] 引入能量函数证明网络稳定性，物理学拯救 AI      | Theory |      P2      |
|  5 | 1986 | [Learning representations by back-propagating errors](https://gwern.net/doc/ai/nn/1986-rumelhart-2.pdf)                          | [鼻祖] 反向传播算法，神经网络训练的原点               | Theory | **P0** |
|  6 | 1995 | [Support-Vector Networks (SVM)](https://www.marenglenbiba.net/dm/cortes_vapnik95.pdf)                                            | [传统ML] 神经网络的"宿敌"，曾长期统治学术界           | Theory |      P2      |
|  7 | 1997 | [Long Short-Term Memory (LSTM)](https://www.bioinf.jku.at/publications/pdf/2071.pdf)                                             | [RNN] 统治 NLP 20 年的循环神经网络架构                | NLP    |      P2      |
|  8 | 1998 | [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)                                                                   | [CV] CNN 鼻祖，定义了卷积-池化-全连接结构             | CV     | **P0** |
|  9 | 2001 | [Greedy Function Approximation (GBM)](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)                                          | [传统ML] XGBoost 前身，表格数据的统治者               | Theory |      P2      |
| 10 | 2006 | [Deep Belief Nets (DBN)](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)                                                    | [里程碑] "深度学习"诞生日，逐层预训练解决梯度消失     | Theory |      P2      |
| 11 | 2009 | [ImageNet](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)                                                    | [数据] 大规模图像数据库，深度学习爆发的燃料           | CV     | **P0** |
| 12 | 2012 | [AlexNet](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)         | [里程碑] ImageNet 夺冠，现代 AI 时代开端              | CV     | **P0** |
| 13 | 2013 | [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf)                                                                                  | [NLP] 词向量基石，让计算机理解词义关系                | NLP    | **P0** |
| 14 | 2013 | [DQN (Deep Q-Network)](https://arxiv.org/pdf/1312.5602.pdf)                                                                      | [RL] 深度强化学习，只看像素就能玩 Atari 游戏          | Theory |      P2      |
| 15 | 2013 | [VAE (Variational Autoencoder)](https://arxiv.org/pdf/1312.6114.pdf)                                                             | [生成] 变分自编码器，理解"潜空间"的鼻祖               | AIGC   |      P2      |
| 16 | 2014 | [GAN](https://arxiv.org/pdf/1406.2661.pdf)                                                                                       | [生成] 生成对抗网络，开启 AI 图像生成早期时代         | AIGC   | **P0** |
| 17 | 2014 | [Adam](https://arxiv.org/pdf/1412.6980.pdf)                                                                                      | [基建] 深度学习最常用的优化器                         | Theory | **P0** |
| 18 | 2014 | [Seq2Seq](https://arxiv.org/pdf/1409.3215.pdf)                                                                                   | [NLP] 机器翻译突破，Transformer 的雏形                | NLP    | **P0** |
| 19 | 2015 | [ResNet](https://arxiv.org/pdf/1512.03385.pdf)                                                                                   | [CV] 残差连接，解决深层网络梯度消失问题               | CV     | **P0** |
| 20 | 2015 | [U-Net](https://arxiv.org/pdf/1505.04597.pdf)                                                                                    | [实战] 图像分割与医疗影像神作                         | CV     |      P2      |
| 21 | 2015 | [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)                                                                      | [优化] 加速训练、防止过拟合的关键技术                 | Theory |      P2      |
| 22 | 2015 | [Bahdanau Attention](https://arxiv.org/pdf/1409.0473.pdf)                                                                        | [NLP] 注意力机制 (Attention) 的首次提出               | NLP    |      P2      |
| 23 | 2015 | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf)                                            | [安全] 对抗攻击，揭示神经网络的脆弱性                 | Theory |      P2      |
| 24 | 2016 | [YOLO](https://arxiv.org/pdf/1506.02640.pdf)                                                                                     | [CV] 实时目标检测基石                                 | CV     | **P0** |
| 25 | 2016 | [AlphaGo](https://www.nature.com/articles/nature16961.pdf)                                                                       | [RL] AI 击败围棋冠军，MCTS + 深度学习                 | Theory |      P2      |
| 26 | 2016 | [WaveNet](https://arxiv.org/pdf/1609.03499.pdf)                                                                                  | [音频] 现代语音合成 (TTS) 的鼻祖                      | AIGC   |      P2      |
| 27 | 2016 | [HNSW](https://arxiv.org/pdf/1603.09320.pdf)                                                                                     | [工程] 向量数据库核心索引，RAG 必读                   | Infra  |      P1      |
| 28 | 2016 | [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)                                                                      | [基建] Transformer 成功训练的幕后英雄                 | Theory | **P0** |
| 29 | 2016 | [NMT with Subword Units (BPE)](https://arxiv.org/pdf/1508.07909.pdf)                                                             | [NLP] Tokenization 标准，理解 Token 本质              | NLP    |      P2      |
| 30 | 2017 | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)                                                                | [圣经] 提出 Transformer，大模型原点                   | NLP    | **P0** |
| 31 | 2017 | [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)                                                                               | [CV] 实例分割，能描绘物体轮廓                         | CV     |      P2      |
| 32 | 2017 | [GCN](https://arxiv.org/pdf/1609.02907.pdf)                                                                                      | [图网络] 处理社交网络/知识图谱的开山作                | Theory |      P2      |
| 33 | 2017 | [Understanding deep learning requires rethinking generalization](https://arxiv.org/pdf/1611.03530.pdf)                           | [理论] 证明神经网络能记住噪声，颠覆泛化认知           | Theory |      P2      |
| 34 | 2017 | [Network Dissection](https://arxiv.org/pdf/1704.05796.pdf)                                                                       | [可解释性] 网络解剖，可视化神经元含义                 | Theory |      P2      |
| 35 | 2018 | [BERT](https://arxiv.org/pdf/1810.04805.pdf)                                                                                     | [NLP] Encoder 巅峰，确立预训练范式                    | NLP    | **P0** |
| 36 | 2018 | [ELMo](https://arxiv.org/pdf/1802.05365.pdf)                                                                                     | [NLP] 动态上下文词向量，承上启下之作                  | NLP    |      P2      |
| 37 | 2018 | [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf)                                                                              | [移动端] 轻量级网络基石，手机端 AI 必读               | CV     | **P0** |
| 38 | 2018 | [GPT-1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) | [LLM] 确立大规模生成式预训练+下游微调路径             | AIGC   |      P2      |
| 39 | 2019 | [GPT-2](https://d4mucfpotywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)      | [LLM] 验证 Zero-shot 能力，坚持 Decoder 路线          | Infra  |      P2      |
| 40 | 2020 | [GPT-3](https://arxiv.org/pdf/2005.14165.pdf)                                                                                    | [LLM] 验证 Scaling Laws，展示"涌现"能力               | Theory | **P0** |
| 41 | 2020 | [Scaling Laws](https://arxiv.org/pdf/2001.08361.pdf)                                                                             | [理论] AI 领域的摩尔定律                              | Theory |      P2      |
| 42 | 2020 | [ViT (Vision Transformer)](https://arxiv.org/pdf/2010.11929.pdf)                                                                 | [CV] Transformer 跨界进入视觉领域                     | CV     | **P0** |
| 43 | 2020 | [DDPM (Diffusion Models)](https://arxiv.org/pdf/2006.11239.pdf)                                                                  | [AIGC] 扩散模型爆发，Stable Diffusion 数学基础        | AIGC   |      P1      |
| 44 | 2020 | [RAG](https://arxiv.org/pdf/2005.11401.pdf)                                                                                      | [工程] 检索增强生成，解决幻觉问题                     | Theory |      P1      |
| 45 | 2020 | [ZeRO](https://arxiv.org/pdf/1910.02054.pdf)                                                                                     | [基建] 分布式训练基建，DeepSpeed 核心技术             | Infra  |      P2      |
| 46 | 2021 | [CLIP](https://arxiv.org/pdf/2103.00020.pdf)                                                                                     | [多模态] 打通图文理解，AI 绘画的基础                  | Theory |      P2      |
| 47 | 2021 | [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2.pdf)                                                              | [Science] 预测蛋白质结构，解决科学难题                | Theory |      P2      |
| 48 | 2021 | [RoFormer (RoPE)](https://arxiv.org/pdf/2104.09864.pdf)                                                                          | [基建] 旋转位置编码，现代大模型标配                   | Theory |      P2      |
| 49 | 2021 | [LoRA](https://arxiv.org/pdf/2106.09685.pdf)                                                                                     | [微调] 低秩适应，大幅降低微调成本                     | Infra  | **P0** |
| 50 | 2021 | [Switch Transformers](https://arxiv.org/pdf/2101.03961.pdf)                                                                      | [架构] MoE 混合专家模型，稀疏激活平衡性能             | NLP    | **P0** |
| 51 | 2022 | [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf)                                                                              | [对齐] 引入 RLHF，让 GPT 进化为 ChatGPT               | Theory | **P0** |
| 52 | 2022 | [Stable Diffusion](https://arxiv.org/pdf/2112.10752.pdf)                                                                         | [AIGC] 高性能 AI 绘画平民化                           | AIGC   |      P1      |
| 53 | 2022 | [CoT (Chain-of-Thought)](https://arxiv.org/pdf/2201.11903.pdf)                                                                   | [提示词] 思维链，激发逻辑推理能力                     | Theory |      P2      |
| 54 | 2022 | [Chinchilla](https://arxiv.org/pdf/2203.15556.pdf)                                                                               | [训练] 修正 Scaling Laws，强调数据质量                | Theory |      P2      |
| 55 | 2022 | [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf)                                                                           | [工程] IO 感知优化，大模型加速核心                    | Infra  |      P1      |
| 56 | 2022 | [Whisper](https://arxiv.org/pdf/2212.04356.pdf)                                                                                  | [音频] 鲁棒性极强的语音识别                           | Audio  | **P0** |
| 57 | 2022 | [ReAct](https://arxiv.org/pdf/2210.03629.pdf)                                                                                    | [Agent] 智能体鼻祖，让模型学会思考+行动               | Agent  | **P0** |
| 58 | 2022 | [LLM.int8()](https://arxiv.org/pdf/2208.07339.pdf)                                                                               | [量化] 8-bit 量化技术，降低显存门槛                   | Theory |      P2      |
| 59 | 2023 | [LLaMA / Llama 2](https://arxiv.org/pdf/2302.13971.pdf)                                                                          | [开源] 引爆开源大模型生态                             | Theory | **P0** |
| 60 | 2023 | [SAM](https://arxiv.org/pdf/2304.02643.pdf)                                                                                      | [CV] 视觉分割基础模型，分割万物                       | CV     |      P1      |
| 61 | 2023 | [DPO](https://arxiv.org/pdf/2305.18290.pdf)                                                                                      | [对齐] 直接偏好优化，RLHF 的高效替代                  | Theory | **P0** |
| 62 | 2023 | [PagedAttention (vLLM)](https://arxiv.org/pdf/2309.06180.pdf)                                                                    | [工程] 显存管理优化，极大提升推理吞吐                 | Infra  |      P2      |
| 63 | 2023 | [QLoRA](https://arxiv.org/pdf/2305.14314.pdf)                                                                                    | [微调] 量化+LoRA，消费级显卡微调大模型                | Infra  | **P0** |
| 64 | 2023 | [Mistral 7B](https://arxiv.org/pdf/2310.06825.pdf)                                                                               | [开源] 混合专家模型 (MoE) 代表                        | Theory |      P2      |
| 65 | 2023 | [Toolformer](https://arxiv.org/pdf/2302.04761.pdf)                                                                               | [Agent] 让模型自学调用 API 工具                       | Agent  | **P0** |
| 66 | 2023 | [Textbooks Are All You Need](https://arxiv.org/pdf/2306.11644.pdf)                                                               | [数据] 数据质量>数量，合成数据的重要性                | NLP    |      P2      |
| 67 | 2023 | [Tree of Thoughts (ToT)](https://arxiv.org/pdf/2305.10601.pdf)                                                                   | [推理] 思维树，用图搜索算法指挥推理                   | Theory |      P2      |
| 68 | 2023 | [Let&#39;s Verify Step by Step](https://arxiv.org/pdf/2305.20050.pdf)                                                            | [对齐] 过程监督 (PRM)，OpenAI o1 慢思考的核心         | Theory |      P2      |
| 69 | 2023 | [Generative Agents](https://arxiv.org/pdf/2304.03442.pdf)                                                                        | [Agent] 斯坦福"虚拟小镇"，25 个 AI 智能体涌现社交行为 | Agent  | **P0** |
| 70 | 2023 | [Voyager](https://arxiv.org/pdf/2305.16291.pdf)                                                                                  | [Agent] Minecraft 中通过写代码自我进化的智能体        | Agent  | **P0** |
| 71 | 2023 | [DSPy](https://arxiv.org/pdf/2310.03714.pdf)                                                                                     | [框架] "Prompting is Programming"，自动优化提示词     | NLP    |      P2      |
| 72 | 2015 | [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)                                             | [蒸馏] 把大模型知识压缩进小模型，端侧 AI 核心技术     | Infra  | **P0** |
| 73 | 2019 | [The Lottery Ticket Hypothesis](https://arxiv.org/pdf/1803.03635.pdf)                                                            | [剪枝] 彩票假设，揭示 90% 参数可能无效                | Theory |      P2      |
| 74 | 2022 | [Grokking](https://arxiv.org/pdf/2201.02177.pdf)                                                                                 | [动力学] "顿悟"现象，颠覆早停的传统认知               | Theory |      P2      |
| 75 | 2022 | [Constitutional AI](https://arxiv.org/pdf/2212.08073.pdf)                                                                        | [安全] Claude 核心技术，AI 根据"宪法"自我监督         | Theory | **P0** |
| 76 | 2022 | [STaR](https://arxiv.org/pdf/2203.14465.pdf)                                                                                     | [推理] OpenAI o1 的思想前身，自我博弈式推理提升       | Theory |      P2      |
| 77 | 2023 | [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/pdf/2212.09748.pdf)                                        | [新视觉] Sora 核心架构，Transformer 替换 U-Net        | Theory | **P0** |

### 优先级说明

- **P0**: 初学者必读，奠基性论文
- **P1**: 重要论文，建议阅读
- **P2**: 进阶论文，按需阅读

### 领域分布

| 领域   | 说明                   |
| :----- | :--------------------- |
| Theory | 基础理论、算法原理     |
| NLP    | 自然语言处理           |
| CV     | 计算机视觉             |
| AIGC   | 生成式 AI              |
| Infra  | 工程基础设施、训练优化 |
| Agent  | 智能体                 |
| Audio  | 语音处理               |
