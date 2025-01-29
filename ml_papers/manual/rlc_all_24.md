# ALL RLC 2024 Papers

### A Batch Sequential Halving Algorithm without Performance Degradation
Link: https://rlj.cs.umass.edu/2024/papers/Paper314.html
Keywords: 
Abstract: In this paper, we investigate the problem of pure exploration in the context of multi-armed bandits, with a specific focus on scenarios where arms are pulled in fixed-size batches. Batching has been shown to enhance computational efficiency, but it can potentially lead to a degradation compared to the original sequential algorithm's performance due to delayed feedback and reduced adaptability. We introduce a simple batch version of the Sequential Halving (SH) algorithm (Karnin et al., 2013) and provide theoretical evidence that batching does not degrade the performance of the original algorithm under practical conditions. Furthermore, we empirically validate our claim through experiments, demonstrating the robust nature of the SH algorithm in fixed-size batch settings.

Research goal: Bandit Theory
Empirical: no
Algorithms: Other (new)
Seeds: 100
Code: yes (in appendix)
Env: custom
Hyperparameters: yes

### A Natural Extension To Online Algorithms For Hybrid RL With Limited Coverage
Link: https://rlj.cs.umass.edu/2024/papers/Paper152.html
Keywords: 
Abstract: Hybrid Reinforcement Learning (RL), leveraging both online and offline data, has garnered recent interest, yet research on its provable benefits remains sparse. Additionally, many existing hybrid RL algorithms (Song et al., 2023; Nakamoto et al., 2023; Amortila et al., 2024) impose a stringent coverage assumption called single-policy concentrability on the offline dataset, requiring that the behavior policy visits every state-action pair that the optimal policy does. With such an assumption, no exploration of unseen state-action pairs is needed during online learning. We show that this is unnecessary, and instead study online algorithms designed to ''fill in the gaps'' in the offline dataset, exploring states and actions that the behavior policy did not explore. To do so, previous approaches focus on estimating the offline data distribution to guide online exploration (Li et al., 2023). We show that a natural extension to standard optimistic online algorithms -- warm-starting them by including the offline dataset in the experience replay buffer -- achieves similar provable gains from hybrid data even when the offline dataset does not have single-policy concentrability. We accomplish this by partitioning the state-action space into two, bounding the regret on each partition through an offline and an online complexity measure, and showing that the regret of this hybrid RL algorithm can be characterized by the best partition -- despite the algorithm not knowing the partition itself. As an example, we propose DISC-GOLF, a modification of an existing optimistic online algorithm with general function approximation called GOLF used in Jin et al. (2021); Xie et al. (2022), and show that it demonstrates provable gains over both online-only and offline-only reinforcement learning, with competitive bounds when specialized to the tabular, linear and block MDP cases. Numerical simulations further validate our theory that hybrid data facilitates more efficient exploration, supporting the potential of hybrid RL in various scenarios

Research goal: Theory
Empirical: hybrid
Algorithms: Other (new)
Seeds: 30
Code: yes
Env: partial/custom
Hyperparameters: yes

### A Recipe for Unbounded Data Augmentation in Visual Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper26.html
Keywords: 
Abstract: Q-learning algorithms are appealing for real-world applications due to their data- efficiency, but they are very prone to overfitting and training instabilities when trained from visual observations. Prior work, namely SVEA, finds that selective application of data augmentation can improve the visual generalization of RL agents without destabilizing training. We revisit its recipe for data augmentation, and find an assumption that limits its effectiveness to augmentations of a photometric nature. Addressing these limitations, we propose a generalized recipe, SADA, that works with wider varieties of augmentations. We benchmark its effectiveness on DMC-GB2 – our proposed extension of the popular DMControl Generalization Benchmark – as well as tasks from Meta-World and the Distracting Control Suite, and find that our method, SADA, greatly improves training stability and generalization of RL agents across a diverse set of augmentations. Visualizations, code and benchmark available at: https://aalmuzairee.github.io/SADA

Research goal: hybrid
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: yes
Env: partial/custom
Hyperparameters: in appendix

### A Simple Mixture Policy Parameterization for Improving Sample Efficiency of CVaR Optimization
Link: https://rlj.cs.umass.edu/2024/papers/Paper81.html
Keywords: 
Abstract: Reinforcement learning algorithms utilizing policy gradients (PG) to optimize Conditional Value at Risk (CVaR) face significant challenges with sample inefficiency, hindering their practical applications. This inefficiency stems from two main facts: a focus on tail-end performance that overlooks many sampled trajectories, and the potential of gradient vanishing when the lower tail of the return distribution is overly flat. To address these challenges, we propose a simple mixture policy parameterization. This method integrates a risk-neutral policy with an adjustable policy to form a risk-averse policy. By employing this strategy, all collected trajectories can be utilized for policy updating, and the issue of vanishing gradients is counteracted by stimulating higher returns through the risk-neutral component, thus lifting the tail and preventing flatness. Our empirical study reveals that this mixture parameterization is uniquely effective across a variety of benchmark domains. Specifically, it excels in identifying risk-averse CVaR policies in some Mujoco environments where the traditional CVaR-PG fails to learn a reasonable policy.

Research goal: Better algorithm
Empirical: yes
Algorithms: Other (new)
Seeds: 10
Code: no
Env: no/custom
Hyperparameters: in appendix

### A Super-human Vision-based Reinforcement Learning Agent for Autonomous Racing in Gran Turismo
Link: https://rlj.cs.umass.edu/2024/papers/Paper213.html
Keywords: 
Abstract: Racing autonomous cars faster than the best human drivers has been a longstanding grand challenge for the fields of Artificial Intelligence and robotics. Recently, an end-to-end deep reinforcement learning agent met this challenge in a high-fidelity racing simulator, Gran Turismo. However, this agent relied on global features that require instrumentation external to the car. This paper introduces, to the best of our knowledge, the first super-human car racing agent whose sensor input is purely local to the car, namely pixels from an ego-centric camera view and quantities that can be sensed from on-board the car, such as the car's velocity. By leveraging global features only at training time, the learned agent is able to outperform the best human drivers in time trial (one car on the track at a time) races using only local input features. The resulting agent is evaluated in Gran Turismo 7 on multiple tracks and cars. Detailed ablation experiments demonstrate the agent's strong reliance on visual inputs, making it the first vision-based super-human car racing agent. 

Research goal: rl for car racing
Empirical: yes
Algorithms: SAC
Seeds: 5 
Code: no
Env: custom/not accessible
Hyperparameters: in appendix

### A Tighter Convergence Proof of Reverse Experience Replay
Link: https://rlj.cs.umass.edu/2024/papers/Paper50.html
Keywords: 
Abstract: In reinforcement learning, Reverse Experience Replay (RER) is a recently proposed algorithm that attains better sample complexity than the classic experience replay method. RER requires the learning algorithm to update the parameters through consecutive state-action-reward tuples in reverse order. However, the most recent theoretical analysis only holds for a minimal learning rate and short consecutive steps, which converge slower than those large learning rate algorithms without RER. In view of this theoretical and empirical gap, we provide a tighter analysis that mitigate the limitation on the learning rate and the length of consecutive steps. Furthermore, we show theoretically that RER converges with a larger learning rate and a longer sequence. 

Research goal: theory
Empirical: no
Algorithms: Other
Seeds: 
Code: yes
Env: custom/not accessible
Hyperparameters: theory

### Agent-Centric Human Demonstrations Train World Models
Link: https://rlj.cs.umass.edu/2024/papers/Paper236.html
Keywords: 
Abstract: Previous work in interactive reinforcement learning considers human behavior directly in agent policy learning, but this requires estimating the distribution of human behavior over many samples to prevent bias. Our work shows that model-based systems can avoid this problem by using small amounts of human data to guide world-model learning rather than agent-policy learning. We show that this approach learns faster and produces useful policies more reliably than prior state-of-the-art. We evaluate our approach with expert human demonstrations in two environments: PinPad5, a fully observable environment which prioritizes task composition, and MemoryMaze, a partially observable environment which prioritizes exploration and memory. We show an order of magnitude speed-up in learning and reliability with only nine minutes of expert human demonstration data. 

Research goal: dreamer from human demos
Empirical: yes
Algorithms: Dreamer (new variatn)
Seeds: 10
Code: no
Env: custom/not accessible
Hyperparameters: in appendix

### An Optimal Tightness Bound for the Simulation Lemma
Link: https://rlj.cs.umass.edu/2024/papers/Paper106.html
Keywords: 
Abstract: We present a bound for value-prediction error with respect to model misspecification that is tight, including constant factors. This is a direct improvement of the ``simulation lemma,’’ a foundational result in reinforcement learning. We demonstrate that existing bounds are quite loose, becoming vacuous for large discount factors, due to the suboptimal treatment of compounding probability errors. By carefully considering this quantity on its own, instead of as a subcomponent of value error, we derive a bound that is sub-linear with respect to transition function misspecification. We then demonstrate broader applicability of this technique, improving a similar bound in the related subfield of hierarchical abstraction.  

Research goal: theory
Empirical: no
Algorithms: NA
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA

### An Open-Loop Baseline for Reinforcement Learning Locomotion Tasks
Link: https://rlj.cs.umass.edu/2024/papers/Paper18.html
Keywords: 
Abstract: In search of a simple baseline for Deep Reinforcement Learning in locomotion tasks, we propose a model-free open-loop strategy. By leveraging prior knowledge and the elegance of simple oscillators to generate periodic joint motions, it achieves respectable performance in five different locomotion environments, with a number of tunable parameters that is a tiny fraction of the thousands typically required by DRL algorithms. We conduct two additional experiments using open-loop oscillators to identify current shortcomings of these algorithms. Our results show that, compared to the baseline, DRL is more prone to performance degradation when exposed to sensor noise or failure. Furthermore, we demonstrate a successful transfer from simulation to reality using an elastic quadruped, where RL fails without randomization or reward engineering. Overall, the proposed baseline and associated experiments highlight the existing limitations of DRL for robotic applications, provide insights on how to address them, and encourage reflection on the costs of complexity and generality.   

Research goal: simpler baseline
Empirical: yes
Algorithms: Other
Seeds: NA
Code: yes
Env: Yes
Hyperparameters: Yes

### An Idiosyncrasy of Time-discretization in Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper164.html
Keywords: 
Abstract: Many reinforcement learning algorithms are built on an assumption that an agent interacts with an environment over fixed-duration, discrete time steps. However, physical systems are continuous in time, requiring a choice of time-discretization granularity when digitally controlling them. Furthermore, such systems do not wait for decisions to be made before advancing the environment state, necessitating the study of how the choice of discretization may affect a reinforcement learning algorithm. In this work, we consider the relationship between the definitions of the continuous-time and discrete-time returns. Specifically, we acknowledge an idiosyncrasy with naively applying a discrete-time algorithm to a discretized continuous-time environment, and note how a simple modification can better align the return definitions. This observation is of practical consideration when dealing with environments where time-discretization granularity is a choice, or situations where such granularity is inherently stochastic.   

Research goal: simpler baseline
Empirical: no
Algorithms: Other
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA

### An Idiosyncrasy of Time-discretization in Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper164.html
Keywords: 
Abstract: Many reinforcement learning algorithms are built on an assumption that an agent interacts with an environment over fixed-duration, discrete time steps. However, physical systems are continuous in time, requiring a choice of time-discretization granularity when digitally controlling them. Furthermore, such systems do not wait for decisions to be made before advancing the environment state, necessitating the study of how the choice of discretization may affect a reinforcement learning algorithm. In this work, we consider the relationship between the definitions of the continuous-time and discrete-time returns. Specifically, we acknowledge an idiosyncrasy with naively applying a discrete-time algorithm to a discretized continuous-time environment, and note how a simple modification can better align the return definitions. This observation is of practical consideration when dealing with environments where time-discretization granularity is a choice, or situations where such granularity is inherently stochastic.   

Research goal: simpler baseline
Empirical: no
Algorithms: Other
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA

### Aquatic Navigation: A Challenging Benchmark for Deep Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper131.html
Keywords: benchmarking, curriculum learning
Abstract: An exciting and promising frontier for Deep Reinforcement Learning (DRL) is its application to real-world robotic systems. While modern DRL approaches achieved remarkable successes in many robotic scenarios (including mobile robotics, surgical assistance, and autonomous driving) unpredictable and non-stationary environments can pose critical challenges to such methods. These features can significantly undermine fundamental requirements for a successful training process, such as the Markovian properties of the transition model. To address this challenge, we propose a new benchmarking environment for aquatic navigation using recent advances in the integration between game engines and DRL. In more detail, we show that our benchmarking environment is problematic even for state-of-the-art DRL approaches that may struggle to generate reliable policies in terms of generalization power and safety. Specifically, we focus on PPO, one of the most widely accepted algorithms, and we propose advanced training techniques (such as curriculum learning and learnable hyperparameters). Our extensive empirical evaluation shows that a well-designed combination of these ingredients can achieve promising results. Our simulation environment and training baselines are freely available to facilitate further research on this open problem and encourage collaboration in the field.

Research goal: benchmarking RL for aquatic navigation
Empirical: yes
Algorithms: PPO
Seeds: 10
Code: yes
Env: yes
Hyperparameters: no (only in code)

### Assigning Credit with Partial Reward Decoupling in Multi-Agent Proximal Policy Optimization
Link: https://rlj.cs.umass.edu/2024/papers/Paper45.html
Keywords: 
Abstract: Multi-agent proximal policy optimization (MAPPO) has recently demonstrated state-of-the-art performance on challenging multi-agent reinforcement learning tasks. However, MAPPO still struggles with the credit assignment problem, wherein the sheer difficulty in ascribing credit to individual agents' actions scales poorly with team size. In this paper, we propose a multi-agent reinforcement learning algorithm that adapts recent developments in credit assignment to improve upon MAPPO. Our approach leverages partial reward decoupling (PRD), which uses a learned attention mechanism to estimate which of a particular agent's teammates are relevant to its learning updates. We use this estimate to dynamically decompose large groups of agents into smaller, more manageable subgroups. We empirically demonstrate that our approach, PRD-MAPPO, decouples agents from teammates that do not influence their expected future reward, thereby streamlining credit assignment. We additionally show that PRD-MAPPO yields significantly higher data efficiency and asymptotic performance compared to both MAPPO and other state-of-the-art methods across several multi-agent tasks, including StarCraft II. Finally, we propose a version of PRD-MAPPO that is applicable to \textit{shared} reward settings, where PRD was previously not applicable, and empirically show that this also leads to performance improvements over MAPPO.

Research goal: 
Empirical: yes
Algorithms: PPO (new multi-agent variant)
Seeds: 5
Code: No
Env: No
Hyperparameters: in appendix

### Bad Habits: Policy Confounding and Out-of-Trajectory Generalization in RL
Link: https://rlj.cs.umass.edu/2024/papers/Paper216.html
Keywords: 
Abstract: Reinforcement learning agents tend to develop habits that are effective only under specific policies. Following an initial exploration phase where agents try out different actions, they eventually converge onto a particular policy. As this occurs, the distribution over state-action trajectories becomes narrower, leading agents to repeatedly experience the same transitions. This repetitive exposure fosters spurious correlations between certain observations and rewards. Agents may then pick up on these correlations and develop simplistic habits tailored to the specific set of trajectories dictated by their policy. The problem is that these habits may yield incorrect outcomes when agents are forced to deviate from their typical trajectories, prompted by changes in the environment. This paper presents a mathematical characterization of this phenomenon, termed policy confounding, and illustrates, through a series of examples, the circumstances under which it occurs. 

Research goal: 
Empirical: yes
Algorithms: PPO, DQN
Seeds: 10
Code: No
Env: No/custom
Hyperparameters: in appendix

### Bandits with Multimodal Structure
Link: https://rlj.cs.umass.edu/2024/papers/Paper350.html
Keywords: 
Abstract: Reinforcement learning agents tend to develop habits that are effective only under specific policies. Following an initial exploration phase where agents try out different actions, they eventually converge onto a particular policy. As this occurs, the distribution over state-action trajectories becomes narrower, leading agents to repeatedly experience the same transitions. This repetitive exposure fosters spurious correlations between certain observations and rewards. Agents may then pick up on these correlations and develop simplistic habits tailored to the specific set of trajectories dictated by their policy. The problem is that these habits may yield incorrect outcomes when agents are forced to deviate from their typical trajectories, prompted by changes in the environment. This paper presents a mathematical characterization of this phenomenon, termed policy confounding, and illustrates, through a series of examples, the circumstances under which it occurs. 

Research goal: 
Empirical: yes
Algorithms: Other
Seeds: 10
Code: yes
Env: custom
Hyperparameters: no

### Best Response Shaping
Link: https://rlj.cs.umass.edu/2024/papers/Paper108.html
Keywords: 
Abstract: We investigate the challenge of multi-agent deep reinforcement learning in partially competitive environments, where traditional methods struggle to foster reciprocity-based cooperation. LOLA and POLA agents learn reciprocity-based cooperative policies by differentiation through a few look-ahead optimization steps of their opponent. However, there is a key limitation in these techniques. Because they consider a few optimization steps, a learning opponent that takes many steps to optimize its return may exploit them. In response, we introduce a novel approach, Best Response Shaping (BRS), which differentiates through an opponent approximating the best response, termed the ""detective."" To condition the detective on the agent's policy for complex games we propose a state-aware differentiable conditioning mechanism, facilitated by a question answering (QA) method that extracts a representation of the agent based on its behaviour on specific environment states. To empirically validate our method, we showcase its enhanced performance against a Monte Carlo Tree Search (MCTS) opponent, which serves as an approximation to the best response in the Coin Game. This work expands the applicability of multi-agent RL in partially competitive environments and provides a new pathway towards achieving improved social welfare in general sum games. 

Research goal: 
Empirical: yes
Algorithms: Other (New)
Seeds: 3-10
Code: yes
Env: custom
Hyperparameters: yes

###  BetaZero: Belief-State Planning for Long-Horizon POMDPs using Learned Approximations
Link: https://rlj.cs.umass.edu/2024/papers/Paper27.html
Keywords: 
Abstract: Real-world planning problems, including autonomous driving and sustainable energy applications like carbon storage and resource exploration, have recently been modeled as partially observable Markov decision processes (POMDPs) and solved using approximate methods. To solve high-dimensional POMDPs in practice, state-of-the-art methods use online planning with problem-specific heuristics to reduce planning horizons and make the problems tractable. Algorithms that learn approximations to replace heuristics have recently found success in large-scale fully observable domains. The key insight is the combination of online Monte Carlo tree search with offline neural network approximations of the optimal policy and value function. In this work, we bring this insight to partially observable domains and propose BetaZero, a belief-state planning algorithm for high-dimensional POMDPs. BetaZero learns offline approximations that replace heuristics to enable online decision making in long-horizon problems. We address several challenges inherent in large-scale partially observable domains; namely challenges of transitioning in stochastic environments, prioritizing action branching with a limited search budget, and representing beliefs as input to the network. To formalize the use of all limited search information, we train against a novel -weighted visit counts policy. We test BetaZero on various well-established POMDP benchmarks found in the literature and a real-world problem of critical mineral exploration. Experiments show that BetaZero outperforms state-of-the-art POMDP solvers on a variety of tasks. 

Research goal: 
Empirical: yes
Algorithms: Other (New)
Seeds: 100
Code: yes
Env: custom
Hyperparameters: yes

###  Bidirectional-Reachable Hierarchical Reinforcement Learning with Mutually Responsive Policies
Link: https://rlj.cs.umass.edu/2024/papers/Paper104.html
Keywords: 
Abstract: Hierarchical reinforcement learning (HRL) addresses complex long-horizon tasks by skillfully decomposing them into subgoals. Therefore, the effectiveness of HRL is greatly influenced by subgoal reachability. Typical HRL methods only consider subgoal reachability from the unilateral level, where a dominant level enforces compliance to the subordinate level. However, we observe that when the dominant level becomes trapped in local exploration or generates unattainable subgoals, the subordinate level is negatively affected and cannot follow the dominant level's actions. This can potentially make both levels stuck in local optima, ultimately hindering subsequent subgoal reachability. Allowing real-time bilateral information sharing and error correction would be a natural cure for this issue, which motivates us to propose a mutual response mechanism. Based on this, we propose the Bidirectional-reachable Hierarchical Policy Optimization~(BrHPO)—a simple yet effective algorithm that also enjoys computation efficiency. Experiment results on a variety of long-horizon tasks showcase that BrHPO outperforms other state-of-the-art HRL baselines, coupled with a significantly higher exploration efficiency and robustness. 

Research goal: 
Empirical: yes
Algorithms: Other (New)
Seeds: 5
Code: yes
Env: custom/partial
Hyperparameters: in appendix

###  Boosting Soft Q-Learning by Bounding
Link: https://rlj.cs.umass.edu/2024/papers/Paper345.html
Keywords: 
Abstract: An agent’s ability to leverage past experience is critical for efficiently solving new tasks. Prior work has focused on using value function estimates to obtain zero-shot approximations for solutions to a new task. In soft -learning, we show how any value function estimate can also be used to derive double-sided bounds on the optimal value function. The derived bounds lead to new approaches for boosting training performance which we validate experimentally. Notably, we find that the proposed framework suggests an alternative method for updating the -function, leading to boosted performance. 

Research goal: 
Empirical: yes
Algorithms: DQN (variant)
Seeds: 30
Code: yes
Env: custom/no
Hyperparameters: in appendix

###  Bounding-Box Inference for Error-Aware Model-Based Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper356.html
Keywords: 
Abstract: In model-based reinforcement learning, simulated experiences from the learned model are often treated as equivalent to experience from the real environment. However, when the model is inaccurate, it can catastrophically interfere with policy learning. Alternatively, the agent might learn about the model's accuracy and selectively use it only when it can provide reliable predictions. We empirically explore model uncertainty measures for selective planning and show that best results require distribution insensitive inference to estimate the uncertainty over model-based updates. To that end, we propose and evaluate bounding-box inference, which operates on bounding-boxes around sets of possible states and other quantities. We find that bounding-box inference can reliably support effective selective planning. 

Research goal: 
Empirical: yes
Algorithms: Other (New)
Seeds: 50
Code: yes
Env: custom/no
Hyperparameters: in appendix

###  Can Differentiable Decision Trees Enable Interpretable Reward Learning from Human Feedback?
Link: https://rlj.cs.umass.edu/2024/papers/Paper237.html
Keywords: 
Abstract: Reinforcement Learning from Human Feedback (RLHF) has emerged as a popular paradigm for capturing human intent to alleviate the challenges of hand-crafting the reward values. Despite the increasing interest in RLHF, most works learn black box reward functions that while expressive are difficult to interpret and often require running the whole costly process of RL before we can even decipher if these frameworks are actually aligned with human preferences. We propose and evaluate a novel approach for learning expressive and interpretable reward functions from preferences using Differentiable Decision Trees (DDTs). Our experiments across several domains, including CartPole, Visual Gridworld environments and Atari games, provide evidence that the tree structure of our learned reward function is useful in determining the extent to which the reward function is aligned with human preferences. We also provide experimental evidence that not only shows that reward DDTs can often achieve competitive RL performance when compared with larger capacity deep neural network reward functions but also demonstrates the diagnostic utility of our framework in checking alignment of learned reward functions. We also observe that the choice between soft and hard (argmax) output of reward DDT reveals a tension between wanting highly shaped rewards to ensure good RL performance, while also wanting simpler, more interpretable rewards. Videos and code, are available at: https://sites.google.com/view/ddt-rlhf 

Research goal: 
Empirical: yes
Algorithms: Other (New)
Seeds: 10
Code: yes
Env: no
Hyperparameters: in appendix

###  Causal Contextual Bandits with Adaptive Context
Link: https://rlj.cs.umass.edu/2024/papers/Paper319.html
Keywords: 
Abstract: We study a variant of causal contextual bandits where the context is chosen based on an initial intervention chosen by the learner. At the beginning of each round, the learner selects an initial action, depending on which a stochastic context is revealed by the environment. Following this, the learner then selects a final action and receives a reward. Given rounds of interactions with the environment, the objective of the learner is to learn a policy (of selecting the initial and the final action) with maximum expected reward. In this paper we study the specific situation where every action corresponds to intervening on a node in some known causal graph. We extend prior work from the deterministic context setting to obtain simple regret minimization guarantees. This is achieved through an instance-dependent causal parameter, , which characterizes our upper bound. Furthermore, we prove that our simple regret is essentially tight for a large class of instances. A key feature of our work is that we use convex optimization to address the bandit exploration problem. We also conduct experiments to validate our theoretical results, and release our code at [github.com/adaptiveContextualCausalBandits/aCCB](https://github.com/adaptiveContextualCausalBandits/aCCB).  

Research goal: 
Empirical: yes
Algorithms: Other (New)
Seeds: 10000
Code: yes
Env: custom
Hyperparameters: yes

###  Co-Learning Empirical Games & World Models
Link: https://rlj.cs.umass.edu/2024/papers/Paper2.html
Keywords: 
Abstract: Game-based decision-making involves reasoning over both world dynamics and strategic interactions among the agents. Typically, models capturing these respective aspects are learned and used separately. We investigate the potential gain from co-learning these elements: a world model for dynamics and an empirical game for strategic interactions. Empirical games drive world models toward a broader consideration of possible game dynamics induced by a diversity of strategy profiles. Conversely, world models guide empirical games to efficiently discover new strategies through planning. We demonstrate these benefits first independently, then in combination as a new algorithm, Dyna-PSRO, that co-learns an empirical game and a world model. When compared to PSRO---a baseline empirical-game building algorithm, Dyna-PSRO is found to compute lower regret solutions on partially observable general-sum games. In our experiments, Dyna-PSRO also requires substantially fewer experiences than PSRO, a key algorithmic advantage for settings where collecting player-game interaction data is a cost-limiting factor.  

Research goal: 
Empirical: yes
Algorithms: Other (New)
Seeds: 5
Code: no
Env: custom
Hyperparameters: in appendix

### Combining Automated Optimisation of Hyperparameters and Reward Shape
Link: https://rlj.cs.umass.edu/2024/papers/Paper188.html
Keywords: 
Abstract: There has been significant progress in deep reinforcement learning (RL) in recent years. Nevertheless, finding suitable hyperparameter configurations and reward functions remains challenging even for experts, and performance heavily relies on these design choices. Also, most RL research is conducted on known benchmarks where knowledge about these choices already exists. However, novel practical applications often pose complex tasks for which no prior knowledge about good hyperparameters and reward functions is available, thus necessitating their derivation from scratch. Prior work has examined automatically tuning either hyperparameters or reward functions individually. We demonstrate empirically that an RL algorithm's hyperparameter configurations and reward function are often mutually dependent, meaning neither can be fully optimised without appropriate values for the other. We then propose a methodology for the combined optimisation of hyperparameters and the reward function. Furthermore, we include a variance penalty as an optimisation objective to improve the stability of learned policies. We conducted extensive experiments using Proximal Policy Optimisation and Soft Actor-Critic on four environments. Our results show that combined optimisation significantly improves over baseline performance in half of the environments and achieves competitive performance in the others, with only a minor increase in computational costs. This suggests that combined optimisation should be best practice.  

Research goal: 
Empirical: yes
Algorithms: PPO, SAC
Seeds: 5
Code: yes
Env: partial
Hyperparameters: in appendix

### Combining Reconstruction and Contrastive Methods for Multimodal Representations in RL
Link: https://rlj.cs.umass.edu/2024/papers/Paper208.html
Keywords: 
Abstract: Learning self-supervised representations using reconstruction or contrastive losses improves performance and sample complexity of image-based and multimodal reinforcement learning (RL). Here, different self-supervised loss functions have distinct advantages and limitations depending on the information density of the underlying sensor modality. Reconstruction provides strong learning signals but is susceptible to distractions and spurious information. While contrastive approaches can ignore those, they may fail to capture all relevant details and can lead to representation collapse. For multimodal RL, this suggests that different modalities should be treated differently, based on the amount of distractions in the signal. We propose Contrastive Reconstructive Aggregated representation Learning (CoRAL), a unified framework enabling us to choose the most appropriate self-supervised loss for each sensor modality and allowing the representation to better focus on relevant aspects. We evaluate CoRAL's benefits on a wide range of tasks with images containing distractions or occlusions, a new locomotion suite, and a challenging manipulation suite with visually realistic distractions. Our results show that learning a multimodal representation by combining contrastive and reconstruction-based losses can significantly improve performance and allow for solving tasks that are out of reach for more naive representation learning approaches and other recent baselines.  

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: yes
Env: custom
Hyperparameters: in appendix

### Constant Stepsize Q-learning: Distributional Convergence, Bias and Extrapolation
Link: https://rlj.cs.umass.edu/2024/papers/Paper144.html
Keywords: 
Abstract: Stochastic Approximation (SA) is a widely used algorithmic approach in various fields, including optimization and reinforcement learning (RL). Among RL algorithms, Q-learning is particularly popular due to its empirical success. In this paper, we study asynchronous Q-learning with constant stepsize, which is commonly used in practice for its fast convergence. By connecting the constant stepsize Q-learning to a time-homogeneous Markov chain, we show the distributional convergence of the iterates in Wasserstein distance and establish its exponential convergence rate. We also establish a Central Limit Theory for Q-learning iterates, demonstrating the asymptotic normality of the averaged iterates. Moreover, we provide an explicit expansion of the asymptotic bias of the averaged iterate in stepsize. Specifically, the bias is proportional to the stepsize up to higher-order terms and we provide an explicit expression for the linear coefficient. This precise characterization of the bias allows the application of Richardson-Romberg (RR) extrapolation technique to construct a new estimate that is provably closer to the optimal Q function. Numerical results corroborate our theoretical finding on the improvement of the RR extrapolation method.  

Research goal: 
Empirical: no
Algorithms: Q
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA

### Contextualized Hybrid Ensemble Q-learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper118.html
Keywords: 
Abstract: Combining Reinforcement Learning (RL) with a prior controller can yield the best out of two worlds: RL can solve complex nonlinear problems, while the control prior ensures safer exploration and speeds up training. Prior work largely blends both components with a fixed weight, neglecting that the RL agent's performance varies with the training progress and across regions in the state space. Therefore, we advocate for an adaptive strategy that dynamically adjusts the weighting based on the RL agent's current capabilities. We propose a new adaptive hybrid RL algorithm, Contextualized Hybrid Ensemble Q-learning (CHEQ). CHEQ combines three key ingredients: (i) a time-invariant formulation of the adaptive hybrid RL problem treating the adaptive weight as a context variable, (ii) a weight adaption mechanism based on the parametric uncertainty of a critic ensemble, and (iii) ensemble-based acceleration for data-efficient RL. Evaluating CHEQ on a car racing task reveals substantially stronger data efficiency, exploration safety, and transferability to unknown scenarios than state-of-the-art adaptive hybrid RL methods.  

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 10
Code: yes
Env: yes
Hyperparameters: in appendix

### Cost Aware Best Arm Identification
Link: https://rlj.cs.umass.edu/2024/papers/Paper193.html
Keywords: 
Abstract: In this paper, we study a best arm identification problem with dual objects. In addition to the classic reward, each arm is associated with a cost distribution and the goal is to identify the largest reward arm using the minimum expected cost. We call it Cost Aware Best Arm Identification (CABAI), which captures the separation of testing and implementation phases in product development pipelines and models the objective shift between phases, i.e., cost for testing and reward for implementation. We first derive an theoretic lower bound for CABAI and propose an algorithm called to match it asymptotically. To reduce the computation of , we further propose a low-complexity algorithm called CO, based on a square-root rule, which proves optimal in simplified two-armed models and generalizes surprisingly well in numerical experiments. Our results show (i) ignoring the heterogeneous action cost results in sub-optimality in practice, and (ii) low-complexity algorithms deliver near-optimal performance over a wide range of problems.   

Research goal: 
Empirical: no
Algorithms: Other (new)
Seeds: NA
Code: no
Env: NA
Hyperparameters: NA

### Cross-environment Hyperparameter Tuning
Link: https://rlj.cs.umass.edu/2024/papers/Paper330.html
Keywords: 
Abstract: This paper introduces a new benchmark, the Cross-environment Hyperparameter Setting Benchmark, that allows comparison of RL algorithms across environments using only a single hyperparameter setting, encouraging algorithmic development which is insensitive to hyperparameters. We demonstrate that the benchmark is robust to statistical noise and obtains qualitatively similar results across repeated applications, even when using a small number of samples. This robustness makes the benchmark computationally cheap to apply, allowing statistically sound insights at low cost. We provide two example instantiations of the CHS, on a set of six small control environments (SC-CHS) and on the entire DM Control suite of 28 environments (DMC-CHS). Finally, to demonstrate the applicability of the CHS to modern RL algorithms on challenging environments, we provide a novel empirical study of an open question in the continuous control literature. We show, with high confidence, that there is no meaningful difference in performance between Ornstein-Uhlenbeck noise and uncorrelated Gaussian noise for exploration with the DDPG algorithm on the DMC-CHS.   

Research goal: 
Empirical: yes
Algorithms: DDPG
Seeds: 3
Code: no
Env: partial
Hyperparameters: in appendix

### Cyclicity-Regularized Coordination Graphs
Link: https://rlj.cs.umass.edu/2024/papers/Paper40.html
Keywords: 
Abstract: In parallel with the rise of the successful value function factorization approach, numerous recent studies on Cooperative Multi-Agent Reinforcement Learning (MARL) have explored the application of Coordination Graphs (CG) to model the communication requirements among the agent population. These coordination problems often exhibit structural sparsity, which facilitates accurate joint value function learning with CGs. Value-based methods necessitate the computation of argmaxes over the exponentially large joint action space, leading to the adoption of the max-sum method from the distributed constraint optimization (DCOP) literature. However, it has been empirically observed that the performance of max-sum deteriorates with an increase in the number of agents, attributed to the increased cyclicity of the graph. While previous works have tackled this issue by sparsifying the graph based on a metric of edge importance, thereby demonstrating improved performance, we argue that neglecting topological considerations during the sparsification procedure can adversely affect action selection. Consequently, we advocate for the explicit consideration of graph cyclicity alongside edge importances. We demonstrate that this approach results in superior performance across various challenging coordination problems.   

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: no
Env: partial/custom
Hyperparameters: in appendix

### D5RL: Diverse Datasets for Data-Driven Deep Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper305.html
Keywords: 
Abstract: Offline reinforcement learning algorithms hold the promise of enabling data-driven RL methods that do not require costly or dangerous real-world exploration and benefit from large pre-collected datasets. This in turn can facilitate real-world applications, as well as a more standardized approach to RL research. Furthermore, offline RL methods can provide effective initializations for online finetuning to overcome challenges with exploration. However, evaluating progress on offline RL algorithms requires effective and challenging benchmarks that capture properties of real-world tasks, provide a range of task difficulties, and cover a range of challenges both in terms of the parameters of the domain (e.g., length of the horizon, sparsity of rewards) and the parameters of the data (e.g., narrow demonstration data or broad exploratory data). While considerable progress in offline RL in recent years has been enabled by simpler benchmark tasks, the most widely used datasets are increasingly saturating in performance and may fail to reflect properties of realistic tasks. We propose a new benchmark for offline RL that focuses on realistic simulations of robotic manipulation and locomotion environments, based on models of real-world robotic systems, and comprising a variety of data sources, including scripted data, play-style data collected by human teleoperators, and other data sources. Our proposed benchmark covers state-based and image-based domains, and supports both offline RL and online fine-tuning evaluation, with some of the tasks specifically designed to require both pre-training and fine-tuning. We hope that our proposed benchmark will facilitate further progress on both offline RL and fine-tuning algorithms. Website with code, examples, tasks, and data is available at \url{https://sites.google.com/view/d5rl/}    

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: no
Code: yes
Env: yes
Hyperparameters: in appendix

### Demystifying the Recency Heuristic in Temporal-Difference Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper126.html
Keywords: 
Abstract: The recency heuristic in reinforcement learning is the assumption that stimuli that occurred closer in time to an acquired reward should be more heavily reinforced. The recency heuristic is one of the key assumptions made by TD(), which reinforces recent experiences according to an exponentially decaying weighting. In fact, all other widely used return estimators for TD learning, such as -step returns, satisfy a weaker (i.e., non-monotonic) recency heuristic. Why is the recency heuristic effective for temporal credit assignment? What happens when credit is assigned in a way that violates this heuristic? In this paper, we analyze the specific mathematical implications of adopting the recency heuristic in TD learning. We prove that any return estimator satisfying this heuristic: 1) is guaranteed to converge to the correct value function, 2) has a relatively fast contraction rate, and 3) has a long window of effective credit assignment, yet bounded worst-case variance. We also give a counterexample where on-policy, tabular TD methods violating the recency heuristic diverge. Our results offer some of the first theoretical evidence that credit assignment based on the recency heuristic facilitates learning.     

Research goal: 
Empirical: no
Algorithms: Other (new)
Seeds: 400
Code: yes
Env: yes
Hyperparameters: no

### Dissecting Deep RL with High Update Ratios: Combatting Value Divergence
Link: https://rlj.cs.umass.edu/2024/papers/Paper125.html
Keywords: 
Abstract: We show that deep reinforcement learning algorithms can retain their ability to learn without resetting network parameters in settings where the number of gradient updates greatly exceeds the number of environment samples by combatting value function divergence. Under large update-to-data ratios, a recent study by Nikishin et al. (2022) suggested the emergence of a primacy bias, in which agents overfit early interactions and downplay later experience, impairing their ability to learn. In this work, we investigate the phenomena leading to the primacy bias. We inspect the early stages of training that were conjectured to cause the failure to learn and find that one fundamental challenge is a long-standing acquaintance: value function divergence. Overinflated Q-values are found not only on out-of-distribution but also in-distribution data and can be linked to overestimation on unseen action prediction propelled by optimizer momentum. We employ a simple unit-ball normalization that enables learning under large update ratios, show its efficacy on the widely used dm_control suite, and obtain strong performance on the challenging dog tasks, competitive with model-based approaches. Our results question, in parts, the prior explanation for sub-optimal learning due to overfitting early data.     

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: yes
Env: yes
Hyperparameters: in appendix

### Distributionally Robust Constrained Reinforcement Learning under Strong Duality
Link: https://rlj.cs.umass.edu/2024/papers/Paper226.html
Keywords: 
Abstract: We study the problem of Distributionally Robust Constrained RL (DRC-RL), where the goal is to maximize the expected reward subject to environmental distribution shifts and constraints. This setting captures situations where training and testing environments differ, and policies must satisfy constraints motivated by safety or limited budgets. Despite significant progress toward algorithm design for the separate problems of distributionally robust RL and constrained RL, there do not yet exist algorithms with end-to-end convergence guarantees for DRC-RL. We develop an algorithmic framework based on strong duality that enables the first efficient and provable solution in a class of environmental uncertainties. Further, our framework exposes an inherent structure of DRC-RL that arises from the combination of distributional robustness and constraints, which prevents a popular class of iterative methods from tractably solving DRC-RL, despite such frameworks being applicable for each of distributionally robust RL and constrained RL individually. Finally, we conduct experiments on a car racing benchmark to evaluate the effectiveness of the proposed algorithm.     

Research goal: theoretical
Empirical: no
Algorithms: Other (new)
Seeds: 3
Code: no
Env: custom
Hyperparameters: yes

### Dreaming of Many Worlds: Learning Contextual World Models aids Zero-Shot Generalization
Link: https://rlj.cs.umass.edu/2024/papers/Paper167.html
Keywords: 
Abstract: Zero-shot generalization (ZSG) to unseen dynamics is a major challenge for creating generally capable embodied agents. To address the broader challenge, we start with the simpler setting of contextual reinforcement learning (cRL), assuming observability of the context values that parameterize the variation in the system's dynamics, such as the mass or dimensions of a robot, without making further simplifying assumptions about the observability of the Markovian state. Toward the goal of ZSG to unseen variation in context, we propose the contextual recurrent state-space model (cRSSM), which introduces changes to the world model of Dreamer~(v3) \citep{hafner-arxiv23a}. This allows the world model to incorporate context for inferring latent Markovian states from the observations and modeling the latent dynamics. Our approach is evaluated on two tasks from the CARL benchmark suite, which is tailored to study contextual RL. Our experiments show that such systematic incorporation of the context improves the ZSG of the policies trained on the ``dreams'' of the world model. We further find qualitatively that our approach allows Dreamer to disentangle the latent state from context, allowing it to extrapolate its dreams to the many worlds of unseen contexts. The code for all our experiments is available at \url{https://github.com/sai-prasanna/dreaming_of_many_worlds}.      

Research goal: 
Empirical: yes
Algorithms: Dreamer (new variant)
Seeds: 10
Code: yes
Env: yes
Hyperparameters: in appendix

### Enabling Intelligent Interactions between an Agent and an LLM: A Reinforcement Learning Approach
Link: https://rlj.cs.umass.edu/2024/papers/Paper161.html
Keywords: 
Abstract: Large language models (LLMs) encode a vast amount of world knowledge acquired from massive text datasets. Recent studies have demonstrated that LLMs can assist an embodied agent in solving complex sequential decision making tasks by providing high-level instructions. However, interactions with LLMs can be time-consuming. In many practical scenarios, it requires a significant amount of storage space that can only be deployed on remote cloud servers. Additionally, using commercial LLMs can be costly since they may charge based on usage frequency. In this paper, we explore how to enable intelligent cost-effective interactions between a down stream task oriented agent and an LLM. We find that this problem can be naturally formulated by a Markov decision process (MDP), and propose When2Ask, a reinforcement learning based approach that learns when it is necessary to query LLMs for high-level instructions to accomplish a target task. On one side, When2Ask discourages unnecessary redundant interactions, while on the other side, it enables the agent to identify and follow useful instructions from the LLM. This enables the agent to halt an ongoing plan and transition to a more suitable one based on new environmental observations. Experiments on MiniGrid and Habitat environments that entail planning sub-goals demonstrate that When2Ask learns to solve target tasks with only a few necessary interactions with the LLM, significantly reducing interaction costs in testing environments compared with baseline methods. Our code is available at: https://github.com/ZJLAB-AMMI/LLM4RL.       

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: yes
Env: yes
Hyperparameters: partial

### Exploring Uncertainty in Distributional Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper122.html
Keywords: 
Abstract: Epistemic uncertainty, which stems from what a learning algorithm does not know, is the natural signal for exploration. Capturing and exploiting epistemic uncertainty for efficient exploration is conceptually straightforward for model-based methods. However, it is computationally ruinous, prompting a search for model-free approaches. One of the most seminal and venerable such is Bayesian Q-learning, which maintains and updates an approximation to the distribution of the long run returns associated with state-action pairs. However, this approximation can be rather severe. Recent work on distributional reinforcement learning (DRL) provides many powerful methods for modelling return distributions which offer the prospect of improving upon Bayesian Q-learning's parametric scheme, but have not been fully investigated for their exploratory potential. Here, we examine the characteristics of a number of DRL algorithms in the context of exploration and propose a novel Bayesian analogue of the categorical temporal-difference algorithm. We show that this works well, converging appropriately to a close approximation to the true return distribution.       

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 40
Code: no
Env: custom
Hyperparameters: in appendix

### Graph Neural Thompson Sampling
Link: https://rlj.cs.umass.edu/2024/papers/Paper12.html
Keywords: 
Abstract: We consider an online decision-making problem with a reward function defined over graph-structured data. We formally formulate the problem as an instance of graph action bandit. We then propose GNN-TS, a Graph Neural Network (GNN) powered Thompson Sampling (TS) algorithm which employs a GNN approximator for estimating the mean reward function and the graph neural tangent features for uncertainty estimation. We prove that, under certain boundness assumptions on the reward function, GNN-TS achieves a state-of-the-art regret bound which is (1) sub-linear of order in the number of interaction rounds, , and a notion of effective dimension , and (2) independent of the number of graph nodes. Empirical results validate that our proposed GNN-TS exhibits competitive performance and scales well on graph action bandit problems.       

Research goal: 
Empirical: no
Algorithms: Other (new)
Seeds: 10
Code: no
Env: custom
Hyperparameters: in appendix

### Guided Data Augmentation for Offline Reinforcement Learning and Imitation Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper33.html
Keywords: 
Abstract: In offline reinforcement learning (RL), an RL agent learns to solve a task using only a fixed dataset of previously collected data. While offline RL has been successful in learning real-world robot control policies, it typically requires large amounts of expert-quality data to learn effective policies that generalize to out-of-distribution states. Unfortunately, such data is often difficult and expensive to acquire in real-world tasks. Several recent works have leveraged data augmentation (DA) to inexpensively generate additional data, but most DA works apply augmentations in a random fashion and ultimately produce highly suboptimal augmented experience. In this work, we propose Guided Data Augmentation (GuDA), a human-guided DA framework that generates expert-quality augmented data. The key insight behind GuDA is that while it may be difficult to demonstrate the sequence of actions required to produce expert data, a user can often easily characterize when an augmented trajectory segment represents progress toward task completion. Thus, a user can restrict the space of possible augmentations to automatically reject suboptimal augmented data. To extract a policy from GuDA, we use off-the-shelf offline reinforcement learning and behavior cloning algorithms. We evaluate GuDA on a physical robot soccer task as well as simulated D4RL navigation tasks, a simulated autonomous driving task, and a simulated soccer task. Empirically, GuDA enables learning given a small initial dataset of potentially suboptimal experience and outperforms a random DA strategy as well as a model-based DA strategy.       

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 10
Code: yes
Env: yes
Hyperparameters: in appendix

### Harnessing Discrete Representations for Continual Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper84.html
Keywords: 
Abstract: Reinforcement learning (RL) agents make decisions using nothing but observations from the environment, and consequently, rely heavily on the representations of those observations. Though some recent breakthroughs have used vector-based categorical representations of observations, often referred to as discrete representations, there is little work explicitly assessing the significance of such a choice. In this work, we provide a thorough empirical investigation of the advantages of discrete representations in the context of world-model learning, model-free RL, and ultimately continual RL problems, where we find discrete representations to have the greatest impact. We find that, when compared to traditional continuous representations, world models learned over discrete representations accurately model more of the world with less capacity, and that agents trained with discrete representations learn better policies with less data. In the context of continual RL, these benefits translate into faster adapting agents. Additionally, our analysis suggests that it is the binary and sparse nature, rather than the “discreteness” of discrete representations that leads to these improvements.       

Research goal: 
Empirical: yes
Algorithms: Other
Seeds: 30
Code: yes
Env: yes
Hyperparameters: in appendix

### Human-compatible driving partners through data-regularized self-play reinforcement learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper338.html
Keywords: 
Abstract: A central challenge for autonomous vehicles is coordinating with humans. Therefore, incorporating realistic human agents is essential for scalable training and evaluation of autonomous driving systems in simulation. Simulation agents are typically developed by imitating large-scale, high-quality datasets of human driving. However, pure imitation learning agents empirically have high collision rates when executed in a multi-agent closed-loop setting. To build agents that are realistic and effective in closed-loop settings, we propose Human-Regularized PPO (HR-PPO), a multi-agent algorithm where agents are trained through self-play with a small penalty for deviating from a human reference policy. In contrast to prior work, our approach is RL-first and only uses 30 minutes of imperfect human demonstrations. We evaluate agents in a large set of multi-agent traffic scenes. Results show our HR-PPO agents are highly effective in achieving goals, with a success rate of 93%, an off-road rate of 3.5 %, and a collision rate of 3 %. At the same time, the agents drive in a human-like manner, as measured by their similarity to existing human driving logs. We also find that HR-PPO agents show considerable improvements on proxy measures for coordination with human driving, particularly in highly interactive scenarios. We open-source our code and trained agents at https://github.com/Emerge-Lab/nocturne_lab and share demonstrations of agent behaviors at https://sites.google.com/view/driving-partners.       

Research goal: 
Empirical: yes
Algorithms: PPO (new variant)
Seeds: 10
Code: yes
Env: yes
Hyperparameters: in appendix

### ICU-Sepsis: A Benchmark MDP Built from Real Medical Data
Link: https://rlj.cs.umass.edu/2024/papers/Paper338.html
Keywords: 
Abstract: A central challenge for autonomous vehicles is coordinating with humans. Therefore, incorporating realistic human agents is essential for scalable training and evaluation of autonomous driving systems in simulation. Simulation agents are typically developed by imitating large-scale, high-quality datasets of human driving. However, pure imitation learning agents empirically have high collision rates when executed in a multi-agent closed-loop setting. To build agents that are realistic and effective in closed-loop settings, we propose Human-Regularized PPO (HR-PPO), a multi-agent algorithm where agents are trained through self-play with a small penalty for deviating from a human reference policy. In contrast to prior work, our approach is RL-first and only uses 30 minutes of imperfect human demonstrations. We evaluate agents in a large set of multi-agent traffic scenes. Results show our HR-PPO agents are highly effective in achieving goals, with a success rate of 93%, an off-road rate of 3.5 %, and a collision rate of 3 %. At the same time, the agents drive in a human-like manner, as measured by their similarity to existing human driving logs. We also find that HR-PPO agents show considerable improvements on proxy measures for coordination with human driving, particularly in highly interactive scenarios. We open-source our code and trained agents at https://github.com/Emerge-Lab/nocturne_lab and share demonstrations of agent behaviors at https://sites.google.com/view/driving-partners.       

Research goal: 
Empirical: yes
Algorithms: DQN, PPO, SAC, Other
Seeds: 8
Code: yes
Env: custom
Hyperparameters: in appendix

### Imitation Learning from Observation through Optimal Transport
Link: https://rlj.cs.umass.edu/2024/papers/Paper241.html
Keywords: 
Abstract: Imitation Learning from Observation (ILfO) is a setting in which a learner tries to imitate the behavior of an expert, using only observational data and without the direct guidance of demonstrated actions. In this paper, we re-examine optimal transport for IL, in which a reward is generated based on the Wasserstein distance between the state trajectories of the learner and expert. We show that existing methods can be simplified to generate a reward function without requiring learned models or adversarial learning. Unlike many other state-of-the-art methods, our approach can be integrated with any RL algorithm and is amenable to ILfO. We demonstrate the effectiveness of this simple approach on a variety of continuous control tasks and find that it surpasses the state of the art in the IlfO setting, achieving expert-level performance across a range of evaluation domains even when observing only a single expert trajectory without actions.       

Research goal: 
Empirical: yes
Algorithms: Other
Seeds: 5
Code: no
Env: partial
Hyperparameters: in appendix

### Improving Thompson Sampling via Information Relaxation for Budgeted Multi-armed Bandits
Link: https://rlj.cs.umass.edu/2024/papers/Paper4.html
Keywords: 
Abstract: We consider a Bayesian budgeted multi-armed bandit problem, in which each arm consumes a different amount of resources when selected and there is a budget constraint on the total amount of resources that can be used. Budgeted Thompson Sampling (BTS) offers a very effective heuristic to this problem, but its arm-selection rule does not take into account the remaining budget information. We adopt \textit{Information Relaxation Sampling} framework that generalizes Thompson Sampling for classical -armed bandit problems, and propose a series of algorithms that are randomized like BTS but more carefully optimize their decisions with respect to the budget constraint. In a one-to-one correspondence with these algorithms, a series of performance benchmarks that improve the conventional benchmark are also suggested. Our theoretical analysis and simulation results show that our algorithms (and our benchmarks) make incremental improvements over BTS (respectively, the conventional benchmark) across various settings including a real-world example.       

Research goal: 
Empirical: no
Algorithms: Other
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA

### Inception: Efficiently Computable Misinformation Attacks on Markov Games
Link: https://rlj.cs.umass.edu/2024/papers/Paper339.html
Keywords: 
Abstract: We study security threats to Markov games due to information asymmetry and misinformation. We consider an attacker player who can spread misinformation about its reward function to influence the robust victim player's behavior. Given a fixed fake reward function, we derive the victim's policy under worst-case rationality and present polynomial-time algorithms to compute the attacker's optimal worst-case policy based on linear programming and backward induction. Then, we provide an efficient inception (""planting an idea in someone's mind"") attack algorithm to find the optimal fake reward function within a restricted set of reward functions with dominant strategies. Importantly, our methods exploit the universal assumption of rationality to compute attacks efficiently. Thus, our work exposes a security vulnerability arising from standard game assumptions under misinformation.       

Research goal: 
Empirical: no
Algorithms: Other
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA

### Informed POMDP: Leveraging Additional Information in Model-Based RL
Link: https://rlj.cs.umass.edu/2024/papers/Paper105.html
Keywords: 
Abstract: In this work, we generalize the problem of learning through interaction in a POMDP by accounting for eventual additional information available at training time. First, we introduce the informed POMDP, a new learning paradigm offering a clear distinction between the information at training and the observation at execution. Next, we propose an objective that leverages this information for learning a sufficient statistic of the history for the optimal control. We then adapt this informed objective to learn a world model able to sample latent trajectories. Finally, we empirically show a learning speed improvement in several environments using this informed world model in the Dreamer algorithm. These results and the simplicity of the proposed adaptation advocate for a systematic consideration of eventual additional information when learning in a POMDP using model-based RL.       

Research goal: 
Empirical: yes
Algorithms: Dreamer (new variant)
Seeds: no
Code: yes
Env: custom
Hyperparameters: only in code

### Inverse Reinforcement Learning with Multiple Planning Horizons
Link: https://rlj.cs.umass.edu/2024/papers/Paper138.html
Keywords: 
Abstract: In this work, we study an inverse reinforcement learning (IRL) problem where the experts are planning *under a shared reward function but with different, unknown planning horizons*. Without the knowledge of discount factors, the reward function has a larger feasible solution set, which makes it harder for existing IRL approaches to identify a reward function. To overcome this challenge, we develop algorithms that can learn a global multi-agent reward function with agent-specific discount factors that reconstruct the expert policies. We characterize the feasible solution space of the reward function and discount factors for both algorithms and demonstrate the generalizability of the learned reward function across multiple domains.        

Research goal: 
Empirical: no
Algorithms: Other
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA

### Investigating the Interplay of Prioritized Replay and Generalization
Link: https://rlj.cs.umass.edu/2024/papers/Paper265.html
Keywords: 
Abstract: Experience replay, the reuse of past data to improve sample efficiency, is ubiquitous in reinforcement learning. Though a variety of smart sampling schemes have been introduced to improve performance, uniform sampling by far remains the most common approach. One exception is Prioritized Experience Replay (PER), where sampling is done proportionally to TD errors, inspired by the success of prioritized sweeping in dynamic programming. The original work on PER showed improvements in Atari, but follow-up results were mixed. In this paper, we investigate several variations on PER, to attempt to understand where and when PER may be useful. Our findings in prediction tasks reveal that while PER can improve value propagation in tabular settings, behavior is significantly different when combined with neural networks. Certain mitigationslike delaying target network updates to control generalization and using estimates of expected TD errors in PER to avoid chasing stochasticitycan avoid large spikes in error with PER and neural networks but generally do not outperform uniform replay. In control tasks, none of the prioritized variants consistently outperform uniform replay. We present new insight into the interaction between prioritization, bootstrapping, and neural networks and propose several improvements for PER in tabular settings and noisy domains.        

Research goal: 
Empirical: yes
Algorithms: DQN, Other
Seeds: 30
Code: no
Env: partial
Hyperparameters: in appendix

### JoinGym: An Efficient Join Order Selection Environment
Link: https://rlj.cs.umass.edu/2024/papers/Paper14.html
Keywords: 
Abstract: Join order selection (JOS), the ordering of join operations to minimize query execution cost, is a core NP-hard combinatorial optimization problem in database query optimization. We present \textsc{JoinGym}, a lightweight and easy-to-use reinforcement learning (RL) environment that captures both left-deep and bushy variants of the JOS problem. Compared to prior works that execute queries online, \textsc{JoinGym} has much higher throughput and efficiently simulates the cost of joins offline by looking up the intermediate table's cardinality from a pre-computed dataset. We provide such a cardinality dataset for queries based on real IMDb workloads, which is the largest suite its kind and may be of independent interest. We extensively benchmark several RL algorithms and find that the best policies are competitive with or better than Postgres, a strong non-learning baseline. However, the learned policies can still catastrophically fail on a small fraction of queries which motivates future research using \textsc{JoinGym} to improve generalization and safety in long-tailed, partially observed, combinatorial optimization problems        

Research goal: 
Empirical: yes
Algorithms: DQN, TD3, SAC, PPO
Seeds: 10
Code: yes
Env: custom
Hyperparameters: in appendix

### Learning Abstract World Models for Value-preserving Planning with Options
Link: https://rlj.cs.umass.edu/2024/papers/Paper223.html
Keywords: 
Abstract: General-purpose agents require fine-grained controls and rich sensory inputs to perform a wide range of tasks. However, this complexity often leads to intractable decision-making. Traditionally, agents are provided with task-specific action and observation spaces to mitigate this challenge, but this reduces autonomy. Instead, agents must be capable of building state-action spaces at the correct abstraction level from their sensorimotor experiences. We leverage the structure of a given set of temporally extended actions to learn abstract Markov decision processes (MDPs) that operate at a higher level of temporal and state granularity. We characterize state abstractions necessary to ensure that planning with these skills, by simulating trajectories in the abstract MDP, results in policies with bounded value loss in the original MDP. We evaluate our approach in goal-based navigation environments that require continuous abstract states to plan successfully and show that abstract model learning improves the sample efficiency of planning and learning.        

Research goal: 
Empirical: yes
Algorithms: Dreamer
Seeds: no
Code: no
Env: yes
Hyperparameters: in appendix

### Learning Action-based Representations Using Invariance
Link: https://rlj.cs.umass.edu/2024/papers/Paper39.html
Keywords: 
Abstract: Robust reinforcement learning agents using high-dimensional observations must be able to identify relevant state features amidst many exogeneous distractors. A representation that captures controllability identifies these state elements by determining what affects agent control. While methods such as inverse dynamics and mutual information capture controllability for a limited number of timesteps, capturing long-horizon elements remains a challenging problem. Myopic controllability can capture the moment right before an agent crashes into a wall, but not the control-relevance of the wall while the agent is still some distance away. To address this we introduce action-bisimulation encoding, a method inspired by the bisimulation invariance pseudometric, that extends single-step controllability with a recursive invariance constraint. By doing this, action-bisimulation learns a multi-step controllability metric that smoothly discounts distant state features that are relevant for control. We demonstrate that action-bisimulation pretraining on reward-free, uniformly random data improves sample efficiency in several environments, including a photorealistic 3D simulation domain, Habitat. Additionally, we provide theoretical analysis and qualitative results demonstrating the information captured by action-bisimulation.        

Research goal: 
Empirical: yes
Algorithms: DQN, PPO, Other
Seeds: 5
Code: yes
Env: custom
Hyperparameters: in appendix

### Learning Discrete World Models for Heuristic Search
Link: https://rlj.cs.umass.edu/2024/papers/Paper225.html
Keywords: 
Abstract: Robust reinforcement learning agents using high-dimensional observations must be able to identify relevant state features amidst many exogeneous distractors. A representation that captures controllability identifies these state elements by determining what affects agent control. While methods such as inverse dynamics and mutual information capture controllability for a limited number of timesteps, capturing long-horizon elements remains a challenging problem. Myopic controllability can capture the moment right before an agent crashes into a wall, but not the control-relevance of the wall while the agent is still some distance away. To address this we introduce action-bisimulation encoding, a method inspired by the bisimulation invariance pseudometric, that extends single-step controllability with a recursive invariance constraint. By doing this, action-bisimulation learns a multi-step controllability metric that smoothly discounts distant state features that are relevant for control. We demonstrate that action-bisimulation pretraining on reward-free, uniformly random data improves sample efficiency in several environments, including a photorealistic 3D simulation domain, Habitat. Additionally, we provide theoretical analysis and qualitative results demonstrating the information captured by action-bisimulation.        

Research goal: 
Empirical: yes
Algorithms: Other
Seeds: no
Code: no
Env: custom
Hyperparameters: in appendix

### Learning to Navigate in Mazes with Novel Layouts using Abstract Top-down Maps
Link: https://rlj.cs.umass.edu/2024/papers/Paper341.html
Keywords: 
Abstract: Learning navigation capabilities in different environments has long been one of the major challenges in decision-making. In this work, we focus on zero-shot navigation ability using given abstract 2-D top-down maps. Like human navigation by reading a paper map, the agent reads the map as an image when navigating in a novel layout, after learning to navigate on a set of training maps. We propose a model-based reinforcement learning approach for this multi-task learning problem, where it jointly learns a hypermodel that takes top-down maps as input and predicts the weights of the transition network. We use the DeepMind Lab environment and customize layouts using generated maps. Our method can adapt better to novel environments in zero-shot and is more robust to noise.        

Research goal: 
Empirical: yes
Algorithms: DQN, Other
Seeds: no
Code: no
Env: partial
Hyperparameters: in appendix

### Learning to Optimize for Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper55.html
Keywords: 
Abstract: In recent years, by leveraging more data, computation, and diverse tasks, learned optimizers have achieved remarkable success in supervised learning, outperforming classical hand-designed optimizers. Reinforcement learning (RL) is essentially different from supervised learning, and in practice, these learned optimizers do not work well even in simple RL tasks. We investigate this phenomenon and identify two issues. First, the agent-gradient distribution is non-independent and identically distributed, leading to inefficient meta-training. Moreover, due to highly stochastic agent-environment interactions, the agent-gradients have high bias and variance, which increases the difficulty of learning an optimizer for RL. We propose pipeline training and a novel optimizer structure with a good inductive bias to address these issues, making it possible to learn an optimizer for reinforcement learning from scratch. We show that, although only trained in toy tasks, our learned optimizer can generalize to unseen complex tasks in Brax.        

Research goal: 
Empirical: yes
Algorithms:Other (new)
Seeds: 10
Code: yes
Env: partial
Hyperparameters: no

### Light-weight probing of unsupervised representations for Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper242.html
Keywords: 
Abstract: Unsupervised visual representation learning offers the opportunity to leverage large corpora of unlabeled trajectories to form useful visual representations, which can benefit the training of reinforcement learning (RL) algorithms. However, evaluating the fitness of such representations requires training RL algorithms which is computationally intensive and has high variance outcomes. Inspired by the vision community, we study whether linear probing can be a proxy evaluation task for the quality of unsupervised RL representation. Specifically, we probe for the observed reward in a given state and the action of an expert in a given state, both of which are generally applicable to many RL domains. Through rigorous experimentation, we show that the probing tasks are strongly rank correlated with the downstream RL performance on the Atari100k Benchmark, while having lower variance and up to 600x lower computational cost. This provides a more efficient method for exploring the space of pretraining algorithms and identifying promising pretraining recipes without the need to run RL evaluations for every setting. Leveraging this framework, we further improve existing self-supervised learning (SSL) recipes for RL, highlighting the importance of the forward model, the size of the visual backbone, and the precise formulation of the unsupervised objective        

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 10
Code: no
Env: partial
Hyperparameters: in appendix

### Mitigating the Curse of Horizon in Monte-Carlo Returns
Link: https://rlj.cs.umass.edu/2024/papers/Paper80.html
Keywords: 
Abstract: The standard framework in reinforcement learning (RL) dictates that an agent should use every observation collected from interactions with the environment when updating its value estimates. As this sequence of observations becomes longer, the agent is afflicted with the curse of horizon since the computational cost of its updates scales linearly with the length of the sequence. In this paper, we propose methods to mitigate this curse when computing value estimates with Monte-Carlo methods. This is accomplished by selecting a subsequence of observations on which the value estimates are computed. We empirically demonstrate on standard RL benchmarks that adopting an adaptive sampling scheme outperforms the default uniform sampling procedure.         

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 10
Code: no
Env: partial
Hyperparameters: partial

### Mixture of Experts in a Mixture of RL settings
Link: https://rlj.cs.umass.edu/2024/papers/Paper130.html
Keywords: 
Abstract: Mixtures of Experts (MoEs) have gained prominence in (self-)supervised learning due to their enhanced inference efficiency, adaptability to distributed training, and modularity. Previous research has illustrated that MoEs can significantly boost Deep Reinforcement Learning (DRL) performance by expanding the network's parameter count while reducing dormant neurons, thereby enhancing the model's learning capacity and ability to deal with non-stationarity. In this work, we shed more light on MoEs' ability to deal with non-stationarity and investigate MoEs in DRL settings with ``amplified'' non-stationarity via multi-task training, providing further evidence that MoEs improve learning capacity. In contrast to previous work, our multi-task results allow us to better understand the underlying causes for the beneficial effect of MoE in DRL training, the impact of the various MoE components, and insights into how best to incorporate them in actor-critic-based DRL networks. Finally, we also confirm results from previous work.         

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 10
Code: no
Env: partial
Hyperparameters: partial

