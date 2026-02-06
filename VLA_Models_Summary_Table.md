# VLA（Vision-Language-Action）模型发展全景汇总表

> 本表信息来源于各模型在 arXiv 上的论文、官方项目主页及公开资料。无法确认的信息以 "—" 标注，**绝不编造**。
>
> 整理时间：2026年2月

---

## 一、萌芽阶段（2022–2023）

| 序号 | 工作名称 | 论文全称 | 提出时间 | 研究机构 | 模型大小 | 输入 | 输出 | 核心贡献 | 官方链接 | 是否开源 |
|:---:|:---:|:---|:---:|:---|:---:|:---|:---|:---|:---|:---:|
| 1 | CLIPort | CLIPort: What and Where Pathways for Robotic Manipulation | 2021.09 (CoRL 2021) | University of Washington, NVIDIA | — | RGB图像, 自然语言指令 | 机器人操作动作（pick-and-place） | 结合CLIP语义理解与Transporter Networks的空间精度，实现语言条件下的操作 | [arXiv:2109.12098](https://arxiv.org/abs/2109.12098) | 是 |
| 2 | BC-Z | BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning | 2022.02 (CoRL 2021) | Google Research | — | RGB图像, 自然语言指令/视频演示 | 机器人操作动作 | 通过大规模数据收集(25,000+机器人demo)实现零样本任务泛化 | [arXiv:2202.02005](https://arxiv.org/abs/2202.02005) | 是 |
| 3 | GATO | A Generalist Agent | 2022.05 (TMLR 2022) | DeepMind | 1.2B | 多模态（图像, 文本, 本体感知, 关节力矩等） | 多模态（文本, 动作, 关节力矩等） | 单一Transformer网络完成604种不同任务（Atari、图像描述、对话、机器人控制等） | [arXiv:2205.06175](https://arxiv.org/abs/2205.06175) | 否 |
| 4 | VIMA | VIMA: General Robot Manipulation with Multimodal Prompts | 2022.10 (ICML 2023) | Stanford, NVIDIA, Caltech, 清华, UT Austin | — | 多模态提示（交错的文本和视觉token） | 机器人动作（自回归） | 通过多模态提示统一多种机器人操作任务，零样本泛化提升2.9倍 | [arXiv:2210.03094](https://arxiv.org/abs/2210.03094) | 是 |
| 5 | RT-1 | RT-1: Robotics Transformer for Real-World Control at Scale | 2022.12 | Google Research, Everyday Robots | — | 相机图像, 自然语言任务指令 | 电机控制命令 | 基于Transformer的大规模机器人控制，130,000个episode、13个机器人、17个月数据训练 | [arXiv:2212.06817](https://arxiv.org/abs/2212.06817) | 是 |
| 6 | UniPi | Learning Universal Policies via Text-Guided Video Generation | 2023.01 (NeurIPS 2023 Spotlight) | Google Brain | — | 文本描述, 当前图像帧 | 视频序列（通过逆动力学提取动作） | 将序列决策问题转化为文本条件视频生成问题，视频作为跨环境通用接口 | [arXiv:2302.00111](https://arxiv.org/abs/2302.00111) | — |
| 7 | ACT | Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | 2023.04 (RSS 2023) | Stanford, Meta, UC Berkeley | ~80M | RGB图像（多相机）, 关节位置, 本体感知 | 动作块（未来k步动作序列） | 预测固定长度的动作序列而非单步动作，提高时序连贯性和样本效率 | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) | 是 |
| 8 | Diffusion Policy | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | 2023.03 (RSS 2023) | Columbia University, TRI, MIT | — | RGB图像, 机器人观测 | 机器人动作（去噪扩散过程） | 将视觉运动策略表示为条件去噪扩散过程，平均性能提升46.9% | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | 是 |
| 9 | RT-2 | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | 2023.07 | Google DeepMind | — | 相机图像, 自然语言指令 | 机器人动作（文本token形式） | 在VLM(PaLI-X/PaLM-E)上联合微调机器人数据和网络数据，将动作表示为语言token | [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) | 否 |
| 10 | RT-X | Open X-Embodiment: Robotic Learning Datasets and RT-X Models | 2023.10 (ICRA 2024) | Open X-Embodiment Collaboration (21家机构) | — | 多机器人观测（图像、本体感知等） | 跨具身机器人动作 | 21家机构、22种机器人、527种技能的标准化数据集，证明跨机器人正迁移 | [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) | 是 |
| 11 | RoboFlamingo | Vision-Language Foundation Models as Effective Robot Imitators | 2023.08 (ICLR 2024) | — | 3B/4B/9B（基于OpenFlamingo） | 图像, 自然语言指令 | 机器人操作动作 | 将预训练VLM(OpenFlamingo)适配用于机器人模仿学习，在CALVIN上达SOTA | [arXiv:2308.01390](https://arxiv.org/abs/2308.01390) | 是 |
| 12 | LEO | An Embodied Generalist Agent in 3D World | 2023.11 (ICML 2024) | BIGAI, 北京大学, CMU, 清华 | — | 2D图像（自我中心）, 3D点云, 语言指令 | 机器人动作, 3D描述, 导航命令 | 3D世界中的多模态通才Agent，融合2D/3D视觉编码器和LLM | [arXiv:2311.12871](https://arxiv.org/abs/2311.12871) | 是 |
| 13 | GR-1 | Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation | 2023.12 (ICLR 2024) | ByteDance | — | 语言指令, 观测图像序列, 机器人状态 | 机器人动作, 未来图像 | GPT风格Transformer进行大规模视频生成预训练+机器人数据微调，CALVIN 94.9%成功率 | [arXiv:2312.13139](https://arxiv.org/abs/2312.13139) | 是 |

---

## 二、探索阶段（2024年）

| 序号 | 工作名称 | 论文全称 | 提出时间 | 研究机构 | 模型大小 | 输入 | 输出 | 核心贡献 | 官方链接 | 是否开源 |
|:---:|:---:|:---|:---:|:---|:---:|:---|:---|:---|:---|:---:|
| 14 | 3D-VLA | 3D-VLA: A 3D Vision-Language-Action Generative World Model | 2024.03 | UMass Amherst | — | 3D点云, 图像, 语言指令 | 机器人动作, 目标图像, 目标点云 | 引入3D世界模型的VLA，在规划动作前推理未来场景 | [arXiv:2403.09631](https://arxiv.org/abs/2403.09631) | 是 |
| 15 | LLARVA | LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning | 2024.06 (CoRL 2024) | UC Berkeley | — | 图像, 语言指令, 本体感知 | 机器人动作, 视觉轨迹 | 引入视觉-动作指令微调和"视觉轨迹"中间表征 | [arXiv:2406.11815](https://arxiv.org/abs/2406.11815) | 是 |
| 16 | Octo | Octo: An Open-Source Generalist Robot Policy | 2024.05 | 多机构 (Octo Team) | 27M/93M | 图像（腕部/第三方相机）, 语言/目标图像, 本体感知 | 机器人动作 | 开源Transformer扩散策略，800k轨迹训练，可在消费级GPU上高效微调 | [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) | 是 |
| 17 | RT-H | RT-H: Action Hierarchies Using Language | 2024.03 (RSS 2024) | Google DeepMind, Stanford | — | 视觉观测, 语言指令 | 语言运动描述（中间层）, 机器人动作 | 用语言运动作为高级任务和低级动作间的中间层，支持人类语言纠正 | [arXiv:2403.01823](https://arxiv.org/abs/2403.01823) | — |
| 18 | OpenVLA | OpenVLA: An Open-Source Vision-Language-Action Model | 2024.06 | Stanford, UC Berkeley, MIT等 | 7B | 图像（SigLIP+DINOv2）, 语言指令 | token化机器人动作 | 开源7B VLA，以7倍少的参数超越RT-2-X(55B) 16.5%成功率 | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) | 是 |
| 19 | RoboMamba | RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation | 2024.06 (NeurIPS 2024) | — | — | 图像, 语言指令 | 机器人动作, SE(3)位姿 | 用Mamba(SSM)架构实现3倍推理加速，仅需0.1%参数微调 | [arXiv:2406.04339](https://arxiv.org/abs/2406.04339) | 是 |
| 20 | ECoT | Robotic Control via Embodied Chain-of-Thought Reasoning | 2024.07 (CoRL 2024) | Stanford, UC Berkeley | — | 图像, 语言指令, 机器人状态 | 多步推理, 机器人动作 | 训练VLA在执行动作前进行具身推理，OpenVLA成功率提升28% | [arXiv:2407.08693](https://arxiv.org/abs/2407.08693) | 是 |
| 21 | Gen2Act | Gen2Act: Human Video Generation in Novel Scenarios enables Generalizable Robot Manipulation | 2024.09 (CoRL 2025) | Google DeepMind, CMU, Stanford | — | 语言指令, 图像 | 生成的人类演示视频, 机器人动作 | 两阶段：先生成人类演示视频，再基于视频执行策略，实现零样本泛化 | [arXiv:2409.16283](https://arxiv.org/abs/2409.16283) | — |
| 22 | TinyVLA | TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation | 2024.09 (RA-L 2025) | — | — | 图像, 语言指令 | 机器人动作（扩散策略解码器） | 紧凑型VLA，推理速度比OpenVLA快20倍，无需大规模预训练 | [arXiv:2409.12514](https://arxiv.org/abs/2409.12514) | 是 |
| 23 | HPT | Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers | 2024.09 (NeurIPS 2024 Spotlight) | MIT CSAIL, Meta AI | 1B | 本体感知, 视觉（不同具身） | 机器人控制动作 | 预训练共享Transformer主干学习任务和具身无关表征，52个数据集200k+轨迹 | [arXiv:2409.20537](https://arxiv.org/abs/2409.20537) | 是 |
| 24 | RDT | RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation | 2024.10 | 清华大学 | 1.2B | RGB图像(3视角), 语言指令, 控制频率, 本体感知 | 机器人动作（预测64步） | 最大的扩散基础模型用于双臂操作，物理可解释统一动作空间 | [arXiv:2410.07864](https://arxiv.org/abs/2410.07864) | 是 |
| 25 | LAPA | Latent Action Pretraining from Videos | 2024.10 | KAIST, UW, Microsoft, NVIDIA, AI2 | 7B | RGB图像, 语言指令 | 潜在动作→机器人动作 | 首个无需动作标签的VLA预训练方法，从视频学习潜在动作，效率提升30倍 | [arXiv:2410.11758](https://arxiv.org/abs/2410.11758) | 是 |
| 26 | HiRT | HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers | 2024.09 | 清华大学, UC Berkeley | InstructBLIP 7B + 轻量策略 | RGB图像, 语言指令 | 机器人动作 | 层级架构：低频VLM语义理解+高频轻量策略实时控制，推理延迟降低58% | [arXiv:2410.05273](https://arxiv.org/abs/2410.05273) | — |
| 27 | π₀ | π₀: A Vision-Language-Action Flow Model for General Robot Control | 2024.10 | Physical Intelligence | — | RGB图像, 语言指令 | 连续机器人动作（flow matching） | 基于flow matching的VLA，继承VLM语义知识，跨平台通用机器人控制 | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) | 是 |
| 28 | RoboDual | Towards Synergistic, Generalized, and Efficient Dual-System for Robotic Manipulation | 2024.10 | 上海交大, 上海AI Lab, 港大, AgiBot | 专家20M + VLA | RGB图像, 语言提示 | 机器人动作（多步） | 通才VLA+专家扩散策略双系统，真实任务提升26.7%，频率提升3.8倍 | [arXiv:2410.08001](https://arxiv.org/abs/2410.08001) | 是 |
| 29 | CogACT | CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation | 2024.11 | Microsoft Research Asia, 清华, USTC | ~7B | RGB图像, 语言指令 | 连续动作序列（扩散Transformer） | 组件化VLA架构，解耦认知与动作预测，仿真成功率+35%，实机+55% | [arXiv:2411.19650](https://arxiv.org/abs/2411.19650) | 是 |
| 30 | GR-2 | GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation | 2024.10 | ByteDance Research | — | 视频帧, 语言指令 | 机器人动作, 视频生成 | 38M视频clips预训练捕获世界动态，100+任务97.7%平均成功率 | [arXiv:2410.06158](https://arxiv.org/abs/2410.06158) | 部分 |
| 31 | VPP | Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations | 2024.12 | 清华, UC Berkeley, 上海AI Lab, Robot Era | — | RGB图像, 语言指令 | 机器人动作 | 利用视频扩散模型的预测视觉表征学习控制，CALVIN提升18.6% | [arXiv:2412.14803](https://arxiv.org/abs/2412.14803) | 是 |
| 32 | RoboVLMs | Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models | 2024.12 | 清华, ByteDance, CASIA, 上海交大, NUS | — | RGB图像, 语言指令 | 机器人动作 | 系统性研究VLA设计选择，600+实验、8+VLM骨干、4种策略架构 | [arXiv:2412.14058](https://arxiv.org/abs/2412.14058) | 是 |
| 33 | TraceVLA | TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies | 2024.12 | UMD, Microsoft Research | 7B / 4B | RGB图像（叠加视觉轨迹）, 语言指令 | 机器人动作 | 视觉轨迹提示增强时空感知，实机任务性能提升3.5倍 | [arXiv:2412.10345](https://arxiv.org/abs/2412.10345) | 是 |
| 34 | FLIP | FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model | 2024.12 | NUS | — | RGB图像, 语言指令 | 图像流, 视频计划, 机器人动作 | 基于模型规划框架，图像流作为通用动作表示，合成长时域计划 | [arXiv:2412.08261](https://arxiv.org/abs/2412.08261) | — |
| 35 | DiVLA | Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning | 2024.12 | Midea Group等 | 2B–72B | RGB图像, 语言指令 | 机器人动作（扩散模型）+ 自生成推理 | 统一自回归模型（推理）与扩散模型（动作），推理注入模块 | [arXiv:2412.03293](https://arxiv.org/abs/2412.03293) | — |

---

## 三、快速发展阶段（2025年）

| 序号 | 工作名称 | 论文全称 | 提出时间 | 研究机构 | 模型大小 | 输入 | 输出 | 核心贡献 | 官方链接 | 是否开源 |
|:---:|:---:|:---|:---:|:---|:---:|:---|:---|:---|:---|:---:|
| 36 | SpatialVLA | SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Models | 2025.01 | 上海AI Lab, ShanghaiTech, TeleAI | 3.5B | 图像观测, 语言指令 | 机器人动作(7D) | 引入Ego3D位置编码注入3D空间信息，自适应动作网格离散化 | [arXiv:2501.15830](https://arxiv.org/abs/2501.15830) | 是 |
| 37 | UP-VLA | UP-VLA: A Unified Understanding and Prediction Model for Embodied Agent | 2025.01 (ICML 2025) | 清华大学, 上海齐智研究所 | — | 图像观测, 语言指令 | 机器人动作, 未来图像预测 | 统一多模态理解与未来预测的训练范式 | [arXiv:2501.18867](https://arxiv.org/abs/2501.18867) | 是 |
| 38 | π₀-FAST | FAST: Efficient Action Tokenization for Vision-Language-Action Models | 2025.01 | Physical Intelligence, UC Berkeley, Stanford | 基于π₀ (3B骨干) | 图像观测, 语言指令 | 机器人动作（FAST tokenization） | 频域动作序列token化(DCT)，训练速度提升5倍，保持精度 | [arXiv:2501.09747](https://arxiv.org/abs/2501.09747) | 是 |
| 39 | OpenVLA-OFT | Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success | 2025.02 | Stanford University | 基于OpenVLA (7B) | 图像（第三方/腕部相机）, 语言指令, 本体感知 | 连续机器人动作 | 优化微调方案(OFT)：并行解码+动作块+连续表示，推理加速25-50倍 | [arXiv:2502.19645](https://arxiv.org/abs/2502.19645) | 是 |
| 40 | UniAct | Universal Actions for Enhanced Embodied Foundation Models | 2025 (CVPR 2025) | — | 0.5B | 图像观测, 语言指令 | 通用动作→机器人特定命令 | 统一异构动作表示的通用动作空间，学习跨机器人通用原子行为 | [arXiv:2512.24321](https://arxiv.org/abs/2512.24321) | 是 |
| 41 | ARM4R | Pre-training Auto-regressive Robotic Models with 4D Representations | 2025.02 (ICML 2025) | UC Berkeley | — | 自我中心人类视频, 机器人演示 | 机器人本体状态和动作 | 4D表征（3D点跟踪随时间）从人类视频预训练机器人模型 | [arXiv:2502.13142](https://arxiv.org/abs/2502.13142) | 是 |
| 42 | π₀.₅ | π₀.₅: a Vision-Language-Action Model with Open-World Generalization | 2025.04 (CoRL 2025) | Physical Intelligence | — | 图像, 语言, 物体检测, 语义子任务 | 低级机器人动作, 中间子任务预测 | 异构任务协同训练实现开放世界泛化，可执行10-15分钟的长时域任务 | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) | — |
| 43 | NORA | NORA: A Small Open-Sourced Generalist Vision Language Action Model for Embodied Tasks | 2025.04 | — | 3B | 图像观测, 语言指令 | 机器人动作 | 紧凑3B VLA，基于Qwen-2.5-VL-3B+FAST+tokenizer，性能媲美大模型 | [arXiv:2504.19854](https://arxiv.org/abs/2504.19854) | 是 |
| 44 | DexVLA | DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control | 2025.02 | Midea Group, 华东师范大学 | 扩散专家1B | 视觉, 语言, 机器人状态 | 机器人动作（扩散动作专家） | 插件式扩散动作专家(1B)+具身课程学习，支持单臂/双臂/灵巧手 | [arXiv:2502.05855](https://arxiv.org/abs/2502.05855) | 是 |
| 45 | ChatVLA | ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model | 2025.02 (EMNLP 2025) | Midea Group, 华东师范大学, 上海大学, 清华等 | — | 视觉, 语言, 多模态数据 | 机器人动作 + 多模态理解输出 | 分阶段对齐训练+MoE解决VLA训练中的遗忘和任务干扰 | [arXiv:2502.14420](https://arxiv.org/abs/2502.14420) | — |
| 46 | Magma | Magma: A Foundation Model for Multimodal AI Agents | 2025.02 (CVPR 2025) | Microsoft Research, UMD, UW-Madison, KAIST, UW | 8B | 视觉, 语言, 时空信息 | 数字/物理环境中的动作 | VLM扩展空间时序智能(SoM+ToM)，统一数字和物理世界Agent | [arXiv:2502.13130](https://arxiv.org/abs/2502.13130) | 是 |
| 47 | DexGraspVLA | DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping | 2025.02 (AAAI 2026 Oral) | 北京大学, HKUST(GZ), UPenn | — | 视觉, 语言, 深度信息 | 灵巧抓取动作 | 层级VLM规划器+扩散控制器，零样本灵巧抓取成功率90%+ | [arXiv:2502.20900](https://arxiv.org/abs/2502.20900) | 是 |
| 48 | Hi Robot | Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models | 2025.02 | Physical Intelligence, Stanford, UC Berkeley | — | 视觉, 语言, 实时反馈 | 机器人动作（层级双系统） | 层级系统：高层VLM推理分解+低层π₀执行，支持实时反馈纠正，复杂任务85%成功率 | [arXiv:2502.19417](https://arxiv.org/abs/2502.19417) | — |
| 49 | Helix | Helix: A Vision-Language-Action Model for Generalist Humanoid Control | 2025.02/05 | Figure AI | — | 视觉, 语言, 本体感知 | 全身人形机器人高频连续控制 | 首个输出全人形上身高频连续控制的VLA，支持双机器人协作 | [arXiv:2505.03912](https://arxiv.org/abs/2505.03912) (OpenHelix) | 部分 |
| 50 | Gemini Robotics | Gemini Robotics: Bringing AI into the Physical World | 2025.03 | Google DeepMind | — (基于Gemini 2.0) | 视觉, 语言指令 | 电机控制命令 | 基于Gemini 2.0的VLA，在通用性、交互性和灵巧性方面取得重大进展 | [arXiv:2503.20020](https://arxiv.org/abs/2503.20020) | 否 |
| 51 | GR00T N1 | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | 2025.03 | NVIDIA | 2B / 3B | 视觉, 语言, 本体感知 | 机器人动作（扩散Transformer） | 开源人形机器人VLA，双系统架构(VLM+扩散Transformer)，混合真实/合成数据训练 | [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) | 是 |
| 52 | CoT-VLA | CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models | 2025.03 (CVPR 2025) | Stanford, MIT, NVIDIA等 | 7B | 视觉, 语言, 机器人观测 | 未来图像帧(视觉目标) + 动作序列 | 预测未来图像帧作为视觉思维链，实机任务提升17% | [arXiv:2503.22020](https://arxiv.org/abs/2503.22020) | 是 |
| 53 | HybridVLA | HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model | 2025.03 | 北京大学, BAAI, CUHK | 7B | 视觉, 语言, 机器人状态 | 机器人动作（协同扩散+自回归） | 在单一LLM中统一自回归和扩散动作预测，仿真+14%，实机+19% | [arXiv:2503.10631](https://arxiv.org/abs/2503.10631) | 是 |
| 54 | GO-1 | GO-1 (AgiBot World Colosseo) | 2025.03 | AgiBot, OpenDriveLab, 上海创新研究院 | — | 视觉, 语言, 机器人状态 | 潜在动作token (ViLLA框架) | 100+台G1机器人1M+轨迹训练，比RDT提升32%，比OXE策略提升30% | [arXiv:2503.06669](https://arxiv.org/abs/2503.06669) | 是 |
| 55 | GR00T N1 (上方已列) | — | — | — | — | — | — | — | — | — |
| 56 | ChatVLA-2 | ChatVLA-2: Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge | 2025.05 (NeurIPS 2025) | Midea Group, 华东师范大学 | — | 图像观测, 语言指令 | 机器人动作 + 推理输出 | MoE架构+两阶段训练保留VLM能力，支持开放世界具身推理 | [arXiv:2505.21906](https://arxiv.org/abs/2505.21906) | — |
| 57 | OneTwoVLA | OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning | 2025.05 | — | — | 图像观测, 语言指令 | 机器人动作 + 推理token | 自适应切换System 1/2，在关键时刻触发显式推理 | [arXiv:2505.11917](https://arxiv.org/abs/2505.11917) | 是 |
| 58 | SmolVLA | SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics | 2025.06 | Hugging Face (LeRobot) | 450M | 多相机视角, 机器人感觉运动状态, 语言指令 | 连续机器人动作 | 紧凑450M VLA，性能媲美10倍大模型，单GPU训练和消费级硬件部署 | [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) | 是 |
| 59 | UniVLA | UniVLA: Learning to Act Anywhere with Task-centric Latent Actions | 2025.05 | OpenDriveLab (BAAI Vision) | — | 视觉, 语言, 异构数据 | 潜在动作token | 任务中心潜在动作学习，仅需960 A100-hours (vs 21,500)，跨具身泛化 | [arXiv:2505.06111](https://arxiv.org/abs/2505.06111) | 是 |
| 60 | GraspVLA | GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data | 2025.05 (CoRL 2025) | 北京大学, 港大, BAAI, Galbot | — | 视觉, 语言, 合成抓取数据 | 机器人抓取动作 | 首个在十亿级合成数据(SynGrasp-1B)上预训练的抓取VLA | [arXiv:2505.03233](https://arxiv.org/abs/2505.03233) | 部分 |
| 61 | Robot-R1 | Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics | 2025.05 | KAIST, Yonsei, UC Berkeley | 7B | 场景图像, 环境元数据, 语言指令 | 关键点状态预测, 机器人动作 | RL增强具身推理，7B模型超越GPT-4o的空间/运动推理能力 | [arXiv:2506.00070](https://arxiv.org/abs/2506.00070) | — |
| 62 | 3D-CAVLA | 3D-CAVLA: Leveraging Depth and 3D Context to Generalize VLA Models for Unseen Tasks | 2025.05 (CVPR 2025 Workshop) | NYU | 基于LLaMA2-7B | 视觉, 语言, 深度, 3D上下文 | 机器人操作动作 | 链式思维推理+深度感知+任务导向ROI检测，LIBERO 98.1%，未见任务+8.8% | [arXiv:2505.05800](https://arxiv.org/abs/2505.05800) | 是 |
| 63 | WorldVLA | WorldVLA: Towards Autoregressive Action World Model | 2025.06 | 阿里巴巴达摩院 | 7B | 视觉, 语言, 机器人状态 | 机器人动作 + 未来图像预测 | 统一VLA与世界模型，注意力掩码策略缓解自回归误差传播 | [arXiv:2506.21539](https://arxiv.org/abs/2506.21539) | 是 |
| 64 | Hume | Hume: Introducing System-2 Thinking in Visual-Language-Action Model | 2025.05 | 上海交大, 上海AI Lab, 浙大, AgiBot, 复旦 | — | 视觉, 语言, 机器人状态 | 机器人动作（双系统架构） | 引入System-2思考：价值引导低频深度思考+System-1实时级联动作去噪 | [arXiv:2505.21432](https://arxiv.org/abs/2505.21432) | 是 |
| 65 | SP-VLA | SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration | 2025.06 | — | — | 图像, 语言指令 | 机器人动作 | 动作感知模型调度+时空语义双感知token剪枝，加速1.5-2.4倍 | [arXiv:2506.12723](https://arxiv.org/abs/2506.12723) | — |
| 66 | CoA | Chain-of-Action: Trajectory Autoregressive Modeling for Robotic Manipulation | 2025.06 (NeurIPS 2025) | — | — | 图像, 语言指令 | 机器人动作轨迹 | 逆向推理生成动作轨迹（从目标到当前状态），动作级思维链 | [arXiv:2506.09990](https://arxiv.org/abs/2506.09990) | 是 |
| 67 | BitVLA | BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation | 2025.06 | — | — | 图像, 语言指令 | 机器人动作 | 首次将1-bit量化应用于VLA，三值参数{-1,0,1}，仅29.8%内存 | [arXiv:2506.07530](https://arxiv.org/abs/2506.07530) | 是 |
| 68 | Fast ECoT | Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse | 2025.06 | UCL, U. of Freiburg, Cisco Research | — | 视觉, 语言, 机器人状态 | 机器人动作（加速ECoT推理） | 缓存复用高层推理+并行模块推理+异步调度，延迟降低7.5倍，无需重训 | [arXiv:2506.07639](https://arxiv.org/abs/2506.07639) | — |
| 69 | WorldVLA (上方已列) | — | — | — | — | — | — | — | — | — |
| 70 | TriVLA | TriVLA: A Triple-System-Based Unified VLA with Episodic World Modeling for General Robot Control | 2025.07 | — | — | 图像, 语言指令 | 机器人动作 | 三系统架构(策略学习+视觉语言+动态感知)+情景世界模型 | [arXiv:2507.01424](https://arxiv.org/abs/2507.01424) | — |
| 71 | DreamVLA | DreamVLA: A VLA Model Dreamed with Comprehensive World Knowledge | 2025.07 (NeurIPS 2025) | — | — | 图像, 语言指令 | 机器人动作 + 世界知识预测（动态区域、深度、语义） | 感知-预测-动作环路，预测中间世界知识表征而非图像 | [arXiv:2507.04447](https://arxiv.org/abs/2507.04447) | 是 |
| 72 | GR-3 | GR-3 Technical Report | 2025.07 | ByteDance Seed (硬件: Fourier Intelligence) | — | 语言指令, 环境观测, 机器人状态 | 双臂移动机器人动作块 | VLM+动作预测的大规模VLA，多面训练(网络数据+人类VR数据+机器人数据) | [arXiv:2507.15493](https://arxiv.org/abs/2507.15493) | — |
| 73 | FiS-VLA | — | — | — | — | — | — | 未找到该名称的公开论文 | — | — |

---

## 四、统计摘要

| 维度 | 统计 |
|:---|:---|
| 总计模型数 | 约 71 个（FiS-VLA未找到公开论文） |
| 开源比例 | ~60%+ 完全开源 |
| 模型参数范围 | 27M (Octo-Small) → 72B (DiVLA) |
| 最常见参数量 | 7B（OpenVLA, LAPA, TraceVLA, CogACT, CoT-VLA, HybridVLA, WorldVLA, Robot-R1等） |
| 主要输入模态 | 视觉(RGB/深度/点云) + 自然语言指令 + 本体感知 |
| 主要输出模态 | 连续/离散机器人动作，部分同时输出视频/图像预测、推理链 |
| 重要发展趋势 | 扩散模型动作生成、层级/双系统架构、思维链推理、世界模型集成、轻量化/量化、跨具身泛化 |

---

## 五、关键趋势总结

1. **动作生成范式**：从早期的直接回归 → 扩散策略(Diffusion Policy) → Flow Matching(π₀) → 自回归token化(FAST) → 混合方案(HybridVLA)
2. **架构演进**：单一模型 → 双系统(System 1/2) → 三系统(TriVLA) → MoE(ChatVLA)
3. **推理能力**：无推理 → ECoT显式推理 → CoT-VLA视觉思维链 → DreamVLA世界知识预测
4. **数据效率**：从需要大量机器人数据 → 利用互联网视频(LAPA/GR-2) → 合成数据(GraspVLA) → 人类视频(ARM4R)
5. **效率优化**：模型压缩(BitVLA 1-bit)、小型化(SmolVLA 450M)、推理加速(Fast ECoT 7.5×)、token剪枝(SP-VLA)
6. **产业化**：Google(Gemini Robotics)、NVIDIA(GR00T N1)、Physical Intelligence(π₀系列)、ByteDance(GR系列)、Midea(ChatVLA/DexVLA/DiVLA)、Figure AI(Helix)、AgiBot(GO-1)等企业深度参与

---

> **免责声明**：本表所有信息均来自各模型在 arXiv 上公开发表的论文、官方项目页面及公开资料。"—" 表示在公开来源中未能查实的信息。如有遗漏或错误，欢迎指正。
