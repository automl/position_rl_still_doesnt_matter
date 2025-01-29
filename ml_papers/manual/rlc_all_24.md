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

### More Efficient Randomized Exploration for Reinforcement Learning via Approximate Sampling
Link: https://rlj.cs.umass.edu/2024/papers/Paper148.html
Keywords: 
Abstract: Thompson sampling (TS) is one of the most popular exploration techniques in reinforcement learning (RL). However, most TS algorithms with theoretical guarantees are difficult to implement and not generalizable to Deep RL. While approximate sampling-based exploration schemes are promising, most existing algorithms are specific to linear Markov Decision Processes (MDP) with suboptimal regret bounds, or only use the most basic samplers such as Langevin Monte Carlo. In this work, we propose an algorithmic framework that incorporates different approximate sampling methods with the recently proposed Feel-Good Thompson Sampling (FGTS) approach (Zhang, 2022; Dann et al., 2021), which was previously known to be intractable. When applied to linear MDPs, our regret analysis yields the best known dependency of regret on dimensionality, surpassing existing randomized algorithms. Additionally, we provide explicit sampling complexity for each employed sampler. Empirically, we show that in tasks where deep exploration is necessary, our proposed algorithms that combine FGTS and approximate sampling perform significantly better compared to other strong baselines. On several challenging games from the Atari 57 suite, our algorithms achieve performance that is either better than or on par with other strong baselines from the deep RL literature.         

Research goal: 
Empirical: hybrid
Algorithms: DQN
Seeds: 5
Code: yes
Env: yes
Hyperparameters: yes

### Multi-view Disentanglement for Reinforcement Learning with Multiple Cameras
Link: https://rlj.cs.umass.edu/2024/papers/Paper64.html
Keywords: 
Abstract: The performance of image-based Reinforcement Learning (RL) agents can vary depending on the position of the camera used to capture the images. Training on multiple cameras simultaneously, including a first-person egocentric camera, can leverage information from different camera perspectives to improve the performance of RL. However, hardware constraints may limit the availability of multiple cameras in real-world deployment. Additionally, cameras may become damaged in the real-world preventing access to all cameras that were used during training. To overcome these hardware constraints, we propose Multi-View Disentanglement (MVD), which uses multiple cameras to learn a policy that is robust to a reduction in the number of cameras to generalise to any single camera from the training set. Our approach is a self-supervised auxiliary task for RL that learns a disentangled representation from multiple cameras, with a shared representation that is aligned across all cameras to allow generalisation to a single camera, and a private representation that is camera-specific. We show experimentally that an RL agent trained on a single third-person camera is unable to learn an optimal policy in many control tasks; but, our approach, benefiting from multiple cameras during training, is able to solve the task using only the same single third-person camera.         

Research goal: 
Empirical: yes
Algorithms: SAC
Seeds: 5
Code: yes
Env: partial
Hyperparameters: in appendix

### MultiHyRL: Robust Hybrid RL for Obstacle Avoidance against Adversarial Attacks on the Observation Space
Link: https://rlj.cs.umass.edu/2024/papers/Paper263.html
Keywords: 
Abstract: Reinforcement learning (RL) holds promise for the next generation of autonomous vehicles, but it lacks formal robustness guarantees against adversarial attacks in the observation space for safety-critical tasks. In particular, for obstacle avoidance tasks, attacks on the observation space can significantly alter vehicle behavior, as demonstrated in this paper. Traditional approaches to enhance the robustness of RL-based control policies, such as training under adversarial conditions or employing worst-case scenario planning, are limited by their policy's parameterization and cannot address the challenges posed by topological obstructions in the presence of noise. We introduce a new hybrid RL algorithm featuring hysteresis-based switching to guarantee robustness against these attacks for vehicles operating in environments with multiple obstacles. This hysteresis-based RL algorithm for coping with multiple obstacles, referred to as MultiHyRL, addresses the 2D bird's-eye view obstacle avoidance problem, featuring a complex observation space that combines local (images) and global (vectors) observations. Numerical results highlight its robustness to adversarial attacks in various challenging obstacle avoidance settings where Proximal Policy Optimization (PPO), a traditional RL method, fails.         

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: no
Code: yes
Env: custom
Hyperparameters: in appendix

### Multistep Inverse Is Not All You Need
Link: https://rlj.cs.umass.edu/2024/papers/Paper117.html
Keywords: 
Abstract: Reinforcement learning (RL) holds promise for the next generation of autonomous vehicles, but it lacks formal robustness guarantees against adversarial attacks in the observation space for safety-critical tasks. In particular, for obstacle avoidance tasks, attacks on the observation space can significantly alter vehicle behavior, as demonstrated in this paper. Traditional approaches to enhance the robustness of RL-based control policies, such as training under adversarial conditions or employing worst-case scenario planning, are limited by their policy's parameterization and cannot address the challenges posed by topological obstructions in the presence of noise. We introduce a new hybrid RL algorithm featuring hysteresis-based switching to guarantee robustness against these attacks for vehicles operating in environments with multiple obstacles. This hysteresis-based RL algorithm for coping with multiple obstacles, referred to as MultiHyRL, addresses the 2D bird's-eye view obstacle avoidance problem, featuring a complex observation space that combines local (images) and global (vectors) observations. Numerical results highlight its robustness to adversarial attacks in various challenging obstacle avoidance settings where Proximal Policy Optimization (PPO), a traditional RL method, fails.         

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 10
Code: yes
Env: custom
Hyperparameters: in appendix

### Non-adaptive Online Finetuning for Offline Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper28.html
Keywords: 
Abstract: Offline reinforcement learning (RL) has emerged as an important framework for applying RL to real-life applications. However, the complete lack of online interactions causes technical difficulties. The online finetuning setting which incorporates a limited form of online interactions, often available in practice, has been developed to address these challenges. Unfortunately, existing theoretical frameworks for online finetuning either assume high online sample complexity or require deploying fully adaptive algorithms (i.e., unlimited policy changes), which restrict their application to real-world settings where online interactions and policy updates are expensive and limited. In this paper, we develop a new theoretical framework for online finetuning. Instead of competing with the optimal policy (which inherits the high sample complexity and adaptivity requirements of online RL), we aim to learn a policy that improves as much as possible over an existing reference policy using a pre-specified number of online samples and a non-adaptive data-collection strategy. Our formulation reveals surprising nuances and suggests novel principles that distinguish finetuning from purely online and offline RL.         

Research goal: 
Empirical: no
Algorithms: NA
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA

### Non-stationary Bandits and Meta-Learning with a Small Set of Optimal Arms
Link: https://rlj.cs.umass.edu/2024/papers/Paper358.html
Keywords: 
Abstract: Offline reinforcement learning (RL) has emerged as an important framework for applying RL to real-life applications. However, the complete lack of online interactions causes technical difficulties. The online finetuning setting which incorporates a limited form of online interactions, often available in practice, has been developed to address these challenges. Unfortunately, existing theoretical frameworks for online finetuning either assume high online sample complexity or require deploying fully adaptive algorithms (i.e., unlimited policy changes), which restrict their application to real-world settings where online interactions and policy updates are expensive and limited. In this paper, we develop a new theoretical framework for online finetuning. Instead of competing with the optimal policy (which inherits the high sample complexity and adaptivity requirements of online RL), we aim to learn a policy that improves as much as possible over an existing reference policy using a pre-specified number of online samples and a non-adaptive data-collection strategy. Our formulation reveals surprising nuances and suggests novel principles that distinguish finetuning from purely online and offline RL.         

Research goal: 
Empirical: no
Algorithms: NA
Seeds: 5
Code: yes
Env: custom
Hyperparameters: yes

### OCAtari: Object-Centric Atari 2600 Reinforcement Learning Environments
Link: https://rlj.cs.umass.edu/2024/papers/Paper46.html
Keywords:
Abstract: Cognitive science and psychology suggest that object-centric representations of complex scenes are a promising step towards enabling efficient abstract reasoning from low-level perceptual features. Yet, most deep reinforcement learning approaches only rely on pixel-based representations that do not capture the compositional properties of natural scenes. For this, we need environments and datasets that allow us to work and evaluate object-centric approaches. In our work, we extend the Atari Learning Environments, the most-used evaluation framework for deep RL approaches, by introducing OCAtari, that performs resource-efficient extractions of the object-centric states for these games. Our framework allows for object discovery, object representation learning, as well as object-centric RL. We evaluate OCAtari's detection capabilities and resource efficiency.         

Research goal: 
Empirical: yes
Algorithms: PPO
Seeds: 3
Code: yes
Env: custom/yes
Hyperparameters: in appendix

### Offline Diversity Maximization under Imitation Constraints
Link: https://rlj.cs.umass.edu/2024/papers/Paper169.html
Keywords:
Abstract: There has been significant recent progress in the area of unsupervised skill discovery, utilizing various information-theoretic objectives as measures of diversity. Despite these advances, challenges remain: current methods require significant online interaction, fail to leverage vast amounts of available task-agnostic data and typically lack a quantitative measure of skill utility. We address these challenges by proposing a principled offline algorithm for unsupervised skill discovery that, in addition to maximizing diversity, ensures that each learned skill imitates state-only expert demonstrations to a certain degree. Our main analytical contribution is to connect Fenchel duality, reinforcement learning, and unsupervised skill discovery to maximize a mutual information objective subject to KL-divergence state occupancy constraints. Furthermore, we demonstrate the effectiveness of our method on the standard offline benchmark D4RL and on a custom offline dataset collected from a 12-DoF quadruped robot for which the policies trained in simulation transfer well to the real robotic system.         

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 3
Code: no
Env: yes
Hyperparameters: no

### Offline Reinforcement Learning from Datasets with Structured Non-Stationarity
Link: https://rlj.cs.umass.edu/2024/papers/Paper287.html
Keywords:
Abstract: Current Reinforcement Learning (RL) is often limited by the large amount of data needed to learn a successful policy. Offline RL aims to solve this issue by using transitions collected by a different behavior policy. We address a novel Offline RL problem setting in which, while collecting the dataset, the transition and reward functions gradually change between episodes but stay constant within each episode. We propose a method based on Contrastive Predictive Coding that identifies this non-stationarity in the offline dataset, accounts for it when training a policy, and predicts it during evaluation. We analyze our proposed method and show that it performs well in simple continuous control tasks and challenging, high-dimensional locomotion tasks. We show that our method often achieves the oracle performance and performs better than baselines.         

Research goal: 
Empirical: yes
Algorithms: TD3, PPO
Seeds: 20
Code: yes
Env: partial
Hyperparameters: in appendix

### On the consistency of hyper-parameter selection in value-based deep reinforcement learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper128.html
Keywords:
Abstract: Deep reinforcement learning (deep RL) has achieved tremendous success on various domains through a combination of algorithmic design and careful selection of hyper-parameters. Algorithmic improvements are often the result of iterative enhancements built upon prior approaches, while hyper-parameter choices are typically inherited from previous methods or fine-tuned specifically for the proposed technique. Despite their crucial impact on performance, hyper-parameter choices are frequently overshadowed by algorithmic advancements. This paper conducts an extensive empirical study focusing on the reliability of hyper-parameter selection for value-based deep reinforcement learning agents. Our findings not only help establish which hyper-parameters are most critical to tune, but also help clarify which tunings remain consistent across different training regimes.         

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: yes
Env: yes
Hyperparameters: in appendix

### On Welfare-Centric Fair Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper133.html
Keywords:
Abstract: We propose a welfare-centric fair reinforcement-learning setting, in which an agent enjoys vector-valued reward from a set of beneficiaries. Given a welfare function W(·), the task is to select a policy π̂ that approximately optimizes the welfare of theirvalue functions from start state s0 , i.e., π̂ ≈ argmaxπ W Vπ1 (s0 ), Vπ2 (s0 ), . . . , Vπg (s0 ) . We find that welfare-optimal policies are stochastic and start-state dependent. Whether individual actions are mistakes depends on the policy, thus mistake bounds, regret analysis, and PAC-MDP learning do not readily generalize to our setting. We develop the adversarial-fair KWIK (Kwik-Af) learning model, wherein at each timestep, an agent either takes an exploration action or outputs an exploitation policy, such that the number of exploration actions is bounded and each exploitation policy is ε-welfare optimal. Finally, we reduce PAC-MDP to Kwik-Af, introduce the Equitable Explicit Explore Exploit (E4) learner, and show that it Kwik-Af learns.         

Research goal: 
Empirical: no
Algorithms: Other (new)
Seeds: no
Code: no
Env: custom
Hyperparameters: no

### Online Planning in POMDPs with State-Requests
Link: https://rlj.cs.umass.edu/2024/papers/Paper23.html
Keywords:
Abstract: In key real-world problems, full state information can sometimes be obtained but only at a high cost, such as by activating more precise yet energy-intensive sensors, or by consulting a human, thereby compelling the agent to operate under partial observability. For this scenario, we propose AEMS-SR (Anytime Error Minimization Search with State Requests), a principled online planning algorithm tailored for POMDPs with state requests. By representing the search space as a graph instead of a tree, AEMS-SR avoids the exponential growth of the search space originating from state requests. Theoretical analysis demonstrates AEMS-SR's -optimality, ensuring solution quality, while empirical evaluations illustrate its effectiveness compared with AEMS and POMCP, two SOTA online planning algorithms. AEMS-SR enables efficient planning in domains characterized by partial observability and costly state requests offering practical benefits across various applications         

Research goal: 
Empirical: no
Algorithms: Other (new)
Seeds: no
Code: no
Env: custom
Hyperparameters: no

### Optimizing Rewards while meeting ω-regular Constraints
Link: https://rlj.cs.umass.edu/2024/papers/Paper359.html
Keywords:
Abstract: This paper addresses the problem of synthesizing policies for Markov Decision Processes (MDPs) with hard -regular constraints, which include and are more general than safety, reachability, liveness, and fairness. The objective is to derive a policy that not only makes the MDP adhere to the given -regular constraint with certainty but also maximizes the expected reward. We first show that there are no optimal policies for the general constrained MDP (CMDP) problem with -regular constraints, which contrasts with simpler problem of CMDPs with safety requirements. Next we show that, despite its complexity, the optimal policy can be approximated within any desired level of accuracy in polynomial time. This approximation ensures both the fulfillment of the -regular constraint with probability and the attainment of a -optimal reward for any given . The proof identifies specific classes of policies capable of achieving these objectives and may be of independent interest. Furthermore, we introduce an approach to tackle the CMDP problem by transforming it into a classical MDP reward optimization problem, thereby enabling the application of conventional reinforcement learning. We show that proximal policy optimization is an effective approach to identifying near-optimal policies that satisfy -regular constraints. This result is demonstrated across multiple environments and constraint types.          

Research goal: 
Empirical: no
Algorithms: PPO, SAC
Seeds: no
Code: no
Env: custom
Hyperparameters: in appendix

### PASTA: Pretrained Action-State Transformer Agents
Link: https://rlj.cs.umass.edu/2024/papers/Paper191.html
Keywords:
Abstract: Self-supervised learning has brought about a revolutionary paradigm shift in various computing domains, including NLP, vision, and biology. Recent approaches involve pretraining transformer models on vast amounts of unlabeled data, serving as a starting point for efficiently solving downstream tasks. In reinforcement learning, researchers have recently adapted these approaches, developing models pretrained on expert trajectories. However, existing methods mostly rely on intricate pretraining objectives tailored to specific downstream applications. This paper conducts a comprehensive investigation of models, referred to as pre-trained action-state transformer agents (PASTA). Our study covers a unified framework and covers an extensive set of general downstream tasks including behavioral cloning, offline Reinforcement Learning (RL), sensor failure robustness, and dynamics change adaptation. We systematically compare various design choices and offer valuable insights that will aid practitioners in developing robust models. Key findings highlight improved performance of component-level tokenization, the use of fundamental pretraining objectives such as next token prediction or masked language modeling, and simultaneous training of models across multiple domains. In this study, the developed models contain fewer than 7M parameters allowing a broad community to use these models and reproduce our experiments. We hope that this study will encourage further research into the use of transformers with first principle design choices to represent RL trajectories and contribute to robust policy learning.          

Research goal: 
Empirical: yes
Algorithms: Other (new), SAC
Seeds: 5
Code: no
Env: partial
Hyperparameters: in appendix

### Physics-Informed Model and Hybrid Planning for Efficient Dyna-Style Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper99.html
Keywords:
Abstract: Applying reinforcement learning (RL) to real-world applications requires addressing a trade-off between asymptotic performance, sample efficiency, and inference time. In this work, we demonstrate how to address this triple challenge by leveraging partial physical knowledge about the system dynamics. Our approach involves learning a physics-informed model to boost sample efficiency and generating imaginary trajectories from this model to learn a model-free policy and Q-function. Furthermore, we propose a hybrid planning strategy, combining the learned policy and Q-function with the learned model to enhance time efficiency in planning. Through practical demonstrations, we illustrate that our method improves the compromise between sample efficiency, time efficiency, and performance over state-of-the-art methods.          

Research goal: 
Empirical: yes
Algorithms: Other (new), TD3, TD-MPC
Seeds: 10
Code: yes
Env: no
Hyperparameters: in appendix

### PID Accelerated Temporal Difference Algorithms
Link: https://rlj.cs.umass.edu/2024/papers/Paper270.html
Keywords:
Abstract: Long-horizon tasks, which have a large discount factor, pose a challenge for most conventional reinforcement learning (RL) algorithms. Algorithms such as Value Iteration and Temporal Difference (TD) learning have a slow convergence rate and become inefficient in these tasks. When the transition distributions are given, PID~VI was recently introduced to accelerate the convergence of Value Iteration using ideas from control theory. Inspired by this, we introduce PID TD Learning and PID Q-Learning algorithms for the RL setting, in which only samples from the environment are available. We give a theoretical analysis of the convergence of PID TD Learning and its acceleration compared to the conventional TD Learning. We also introduce a method for adapting PID gains in the presence of noise and empirically verify its effectiveness.          

Research goal: 
Empirical: no
Algorithms: Other (new)
Seeds: 80
Code: no
Env: no
Hyperparameters: in appendix

### Planning to Go Out-of-Distribution in Offline-to-Online Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper67.html
Keywords:
Abstract: Offline pretraining with a static dataset followed by online fine-tuning (offline-to-online, or OtO) is a paradigm that is well matched to a real-world RL deployment process. In this scenario, we aim to find the best-performing policy within a limited budget of online interactions. Previous work in the OtO setting has focused on correcting for bias introduced by the policy-constraint mechanisms of offline RL algorithms. Such constraints keep the learned policy close to the behavior policy that collected the dataset, but we show this can unnecessarily limit policy performance if the behavior policy is far from optimal. Instead, we forgo policy constraints and frame OtO RL as an exploration problem that aims to maximize the benefit of the online data-collection. We first study the major online RL exploration methods based on intrinsic rewards and UCB in the OtO setting, showing that intrinsic rewards add training instability through reward-function modification, and UCB methods are myopic and it is unclear which learned-component's ensemble to use for action selection. We then introduce an algorithm for \textbf{p}lanning to go out of distribution (PTGOOD) that avoids these issues. PTGOOD uses a non-myopic planning procedure that targets exploration in relatively high-reward regions of the state-action space unlikely to be visited by the behavior policy. By leveraging concepts from the Conditional Entropy Bottleneck, PTGOOD encourages data collected online to provide new information relevant to improving the final deployment policy without altering rewards. We show empirically in different control tasks that PTGOOD significantly improves agent returns during online fine-tuning and finds the optimal policy in as few as 10k online steps in the Walker control task and in as few as 50k in complex control tasks such as Humanoid. We find that PTGOOD avoids the suboptimal policy convergence that many of our baselines exhibit in several environments.          

Research goal: 
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: no
Env: yes
Hyperparameters: in appendix

### Policy Architectures for Compositional Generalization in Control
Link: https://rlj.cs.umass.edu/2024/papers/Paper327.html
Keywords:
Abstract: Many tasks in control, robotics, and planning can be specified using desired goal configurations for various entities in the environment. Learning goal-conditioned policies is a natural paradigm to solve such tasks. However, current approaches struggle to learn and generalize as task complexity increases, such as variations in number of environment entities or compositions of goals. In this work, we introduce a framework for modeling entity-based compositional structure in tasks, and create suitable policy designs that can leverage this structure. Our policies, which utilize architectures like Deep Sets and Self Attention, are flexible and can be trained end-to-end without requiring any action primitives. When trained using standard reinforcement and imitation learning methods on a suite of simulated robot manipulation tasks, we find that these architectures achieve significantly higher success rates with less data. We also find these architectures enable broader and compositional generalization, producing policies that extrapolate to different numbers of entities than seen in training, and stitch together (i.e. compose) learned skills in novel ways.          

Research goal: 
Empirical: yes
Algorithms: DDPG
Seeds: 5
Code: yes
Env: yes
Hyperparameters: in appendix

### Policy Gradient Algorithms with Monte Carlo Tree Learning for Non-Markov Decision Processes
Link: https://rlj.cs.umass.edu/2024/papers/Paper168.html
Keywords:
Abstract: Policy gradient (PG) is a reinforcement learning (RL) approach that optimizes a parameterized policy model for an expected return using gradient ascent. While PG can work well even in non-Markovian environments, it may encounter plateaus or peakiness issues. As another successful RL approach, algorithms based on Monte Carlo Tree Search (MCTS), which include AlphaZero, have obtained groundbreaking results, especially in the game-playing domain. They are also effective when applied to non-Markov decision processes. However, the standard MCTS is a method for decision-time planning, which differs from the online RL setting. In this work, we first introduce Monte Carlo Tree Learning (MCTL), an adaptation of MCTS for online RL setups. We then explore a combined policy approach of PG and MCTL to leverage their strengths. We derive conditions for asymptotic convergence with the results of a two-timescale stochastic approximation and propose an algorithm that satisfies these conditions and converges to a reasonable solution. Our numerical experiments validate the effectiveness of the proposed methods.          

Research goal: theory
Empirical: no
Algorithms: Other
Seeds: no
Code: no
Env: custom
Hyperparameters: in appendix

### Policy Gradient Algorithms with Monte Carlo Tree Learning for Non-Markov Decision Processes
Link: https://rlj.cs.umass.edu/2024/papers/Paper168.html
Keywords:
Abstract: Policy gradient (PG) is a reinforcement learning (RL) approach that optimizes a parameterized policy model for an expected return using gradient ascent. While PG can work well even in non-Markovian environments, it may encounter plateaus or peakiness issues. As another successful RL approach, algorithms based on Monte Carlo Tree Search (MCTS), which include AlphaZero, have obtained groundbreaking results, especially in the game-playing domain. They are also effective when applied to non-Markov decision processes. However, the standard MCTS is a method for decision-time planning, which differs from the online RL setting. In this work, we first introduce Monte Carlo Tree Learning (MCTL), an adaptation of MCTS for online RL setups. We then explore a combined policy approach of PG and MCTL to leverage their strengths. We derive conditions for asymptotic convergence with the results of a two-timescale stochastic approximation and propose an algorithm that satisfies these conditions and converges to a reasonable solution. Our numerical experiments validate the effectiveness of the proposed methods.          

Research goal: theory
Empirical: no
Algorithms: Other
Seeds: 10
Code: no
Env: custom
Hyperparameters: in appendix

### Policy Gradient with Active Importance Sampling
Link: https://rlj.cs.umass.edu/2024/papers/Paper90.html
Keywords:
Abstract: mportance sampling (IS) represents a fundamental technique for a large surge of off-policy reinforcement learning approaches. Policy gradient (PG) methods, in particular, significantly benefit from IS, enabling the effective reuse of previously collected samples, thus increasing sample efficiency. However, classically, IS is employed in RL as a passive tool for re-weighting historical samples. However, the statistical community employs IS as an active tool combined with the use of behavioral distributions that allow the reduction of the estimate variance even below the sample mean one. In this paper, we focus on this second setting by addressing the behavioral policy optimization (BPO) problem. We look for the best behavioral policy from which to collect samples to reduce the policy gradient variance as much as possible. We provide an iterative algorithm that alternates between the cross-entropy estimation of the minimum-variance behavioral policy and the actual policy optimization, leveraging on defensive IS. We theoretically analyze such an algorithm, showing that it enjoys a convergence rate of order to a stationary point, but depending on a more convenient variance term w.r.t. standard PG methods. We then provide a practical version that is numerically validated, showing the advantages in the policy gradient estimation variance and on the learning speed.           

Research goal: theory
Empirical: no
Algorithms: Other
Seeds: no
Code: no
Env: no
Hyperparameters: in appendix

### Policy-Guided Diffusion
Link: https://rlj.cs.umass.edu/2024/papers/Paper233.html
Keywords:
Abstract: mportance sampling (IS) represents a fundamental technique for a large surge of off-policy reinforcement learning approaches. Policy gradient (PG) methods, in particular, significantly benefit from IS, enabling the effective reuse of previously collected samples, thus increasing sample efficiency. However, classically, IS is employed in RL as a passive tool for re-weighting historical samples. However, the statistical community employs IS as an active tool combined with the use of behavioral distributions that allow the reduction of the estimate variance even below the sample mean one. In this paper, we focus on this second setting by addressing the behavioral policy optimization (BPO) problem. We look for the best behavioral policy from which to collect samples to reduce the policy gradient variance as much as possible. We provide an iterative algorithm that alternates between the cross-entropy estimation of the minimum-variance behavioral policy and the actual policy optimization, leveraging on defensive IS. We theoretically analyze such an algorithm, showing that it enjoys a convergence rate of order to a stationary point, but depending on a more convenient variance term w.r.t. standard PG methods. We then provide a practical version that is numerically validated, showing the advantages in the policy gradient estimation variance and on the learning speed.           

Research goal:
Empirical: yes
Algorithms: Other (new)
Seeds: 4
Code: yes
Env: partial/custom
Hyperparameters: in appendix

### Reinforcement Learning from Delayed Observations via World Models
Link: https://rlj.cs.umass.edu/2024/papers/Paper280.html
Keywords:
Abstract: In standard reinforcement learning settings, agents typically assume immediate feedback about the effects of their actions after taking them. However, in practice, this assumption may not hold true due to physical constraints and can significantly impact the performance of learning algorithms. In this paper, we address observation delays in partially observable environments. We propose leveraging world models, which have shown success in integrating past observations and learning dynamics, to handle observation delays. By reducing delayed POMDPs to delayed MDPs with world models, our methods can effectively handle partial observability, where existing approaches achieve sub-optimal performance or degrade quickly as observability decreases. Experiments suggest that one of our methods can outperform a naive model-based approach by up to 250%. Moreover, we evaluate our methods on visual delayed environments, for the first time showcasing delay-aware reinforcement learning continuous control with visual observations.           

Research goal:
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: yes
Env: yes
Hyperparameters: yes

### Reinforcement Learning from Human Feedback without Reward Inference
Link: https://rlj.cs.umass.edu/2024/papers/Paper150.html
Keywords:
Abstract: In this paper, we study reinforcement learning from human feedback (RLHF) under an episodic Markov decision process with a general trajectory-wise reward model. We developed a model-free RLHF best policy identification algorithm, called , without explicit reward model inference, which is a critical intermediate step in the contemporary RLHF paradigms for training large language models (LLM). The algorithm identifies the optimal policy directly from human preference information in a backward manner, employing a dueling bandit sub-routine that constantly duels actions to identify the superior one. adopts a reward-free exploration and best-arm-identification-like adaptive stopping criteria to equalize the visitation among all states in the same decision step while moving to the previous step as soon as the optimal action is identifiable, leading to a provable, instance-dependent sample complexity which resembles the result in classic RL, where is the instance-dependent constant and is the batch size. Moreover, can be transformed into an explore-then-commit algorithm with logarithmic regret and generalized to discounted MDPs using a frame-based approach. Our results show: (i) sample-complexity-wise, RLHF is not significantly harder than classic RL and (ii) end-to-end RLHF may deliver improved performance by avoiding pitfalls in reward inferring such as overfit and distribution shift.            

Research goal:
Empirical: no
Algorithms: Other (new)
Seeds: 5
Code: no
Env: custom
Hyperparameters: no


### Representation Alignment from Human Feedback for Cross-Embodiment Reward Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper228.html
Keywords:
Abstract: In this paper, we study reinforcement learning from human feedback (RLHF) under an episodic Markov decision process with a general trajectory-wise reward model. We developed a model-free RLHF best policy identification algorithm, called , without explicit reward model inference, which is a critical intermediate step in the contemporary RLHF paradigms for training large language models (LLM). The algorithm identifies the optimal policy directly from human preference information in a backward manner, employing a dueling bandit sub-routine that constantly duels actions to identify the superior one. adopts a reward-free exploration and best-arm-identification-like adaptive stopping criteria to equalize the visitation among all states in the same decision step while moving to the previous step as soon as the optimal action is identifiable, leading to a provable, instance-dependent sample complexity which resembles the result in classic RL, where is the instance-dependent constant and is the batch size. Moreover, can be transformed into an explore-then-commit algorithm with logarithmic regret and generalized to discounted MDPs using a frame-based approach. Our results show: (i) sample-complexity-wise, RLHF is not significantly harder than classic RL and (ii) end-to-end RLHF may deliver improved performance by avoiding pitfalls in reward inferring such as overfit and distribution shift.            

Research goal:
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: yes
Env: yes
Hyperparameters: in appendix


### Resource Usage Evaluation of Discrete Model-Free Deep Reinforcement Learning Algorithms
Link: https://rlj.cs.umass.edu/2024/papers/Paper304.html
Keywords:
Abstract: Deep Reinforcement Learning (DRL) has become popular due to promising results in chatbot, healthcare, and autonomous driving applications. However, few DRL algorithms are rigorously evaluated in terms of their space or time efficiency, making them difficult to develop and deploy in practice. In current literature, existing performance comparisons mostly focus on inference accuracy, without considering real-world limitations such as maximum runtime and memory. Furthermore, many works do not make their code publicly accessible for others to use. This paper addresses this gap by presenting the most comprehensive resource usage evaluation and performance comparison of DRL algorithms known to date. This work focuses on publicly-accessible discrete model-free DRL algorithms because of their practicality in real-world problems where efficient implementations are necessary. Although there are other state-of-the art algorithms, few were presently deployment-ready for training on a large number of environments. In total, sixteen DRL algorithms were trained in 23 different environments (468 seeds total), which collectively required 256 GB and 830 CPU days to run all experiments and 1.8 GB to store all models. Overall, our results validate several known challenges in DRL, including exploration and memory inefficiencies, the classic exploration-exploitation trade-off, and large resource utilizations. To address these challenges, this paper suggests numerous opportunities for future work to help improve the capabilities of modern algorithms. The findings of this paper are intended to aid researchers and practitioners in improving and employing DRL algorithms in time-sensitive and resource-constrained applications such as economics, cybersecurity, robotics, and the Internet of Things.             

Research goal:
Empirical: yes
Algorithms: Other (new)
Seeds: 20
Code: yes
Env: yes
Hyperparameters: yes


### Revisiting Sparse Rewards for Goal-Reaching Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper231.html
Keywords:
Abstract: Deep Reinforcement Learning (DRL) has become popular due to promising results in chatbot, healthcare, and autonomous driving applications. However, few DRL algorithms are rigorously evaluated in terms of their space or time efficiency, making them difficult to develop and deploy in practice. In current literature, existing performance comparisons mostly focus on inference accuracy, without considering real-world limitations such as maximum runtime and memory. Furthermore, many works do not make their code publicly accessible for others to use. This paper addresses this gap by presenting the most comprehensive resource usage evaluation and performance comparison of DRL algorithms known to date. This work focuses on publicly-accessible discrete model-free DRL algorithms because of their practicality in real-world problems where efficient implementations are necessary. Although there are other state-of-the art algorithms, few were presently deployment-ready for training on a large number of environments. In total, sixteen DRL algorithms were trained in 23 different environments (468 seeds total), which collectively required 256 GB and 830 CPU days to run all experiments and 1.8 GB to store all models. Overall, our results validate several known challenges in DRL, including exploration and memory inefficiencies, the classic exploration-exploitation trade-off, and large resource utilizations. To address these challenges, this paper suggests numerous opportunities for future work to help improve the capabilities of modern algorithms. The findings of this paper are intended to aid researchers and practitioners in improving and employing DRL algorithms in time-sensitive and resource-constrained applications such as economics, cybersecurity, robotics, and the Internet of Things.             

Research goal:
Empirical: yes
Algorithms: SAC
Seeds: 30
Code: yes
Env: custom/partial
Hyperparameters: in appendix


### Reward Centering
Link: https://rlj.cs.umass.edu/2024/papers/Paper261.html
Keywords:
Abstract: We show that discounted methods for solving continuing reinforcement learning problems can perform significantly better if they center their rewards by subtracting out the rewards' empirical average. The improvement is substantial at commonly used discount factors and increases further as the discount factor approaches one. In addition, we show that if a _problem's_ rewards are shifted by a constant, then standard methods perform much worse, whereas methods with reward centering are unaffected. Estimating the average reward is straightforward in the on-policy setting; we propose a slightly more sophisticated method for the off-policy setting. Reward centering is a general idea, so we expect almost every reinforcement-learning algorithm to benefit by the addition of reward centering.             

Research goal:
Empirical: yes
Algorithms: DQN, PPO, Other
Seeds: 10
Code: yes
Env: custom/no
Hyperparameters: in appendix

### RL for Consistency Models: Faster Reward Guided Text-to-Image Generation
Link: https://rlj.cs.umass.edu/2024/papers/Paper210.html
Keywords:
Abstract: Reinforcement learning (RL) has improved guided image generation with diffusion models by directly optimizing rewards that capture image quality, aesthetics, and instruction following capabilities. However, the resulting generative policies inherit the same iterative sampling process of diffusion models that causes slow generation. To overcome this limitation, consistency models proposed learning a new class of generative models that directly map noise to data, resulting in a model that can generate an image in as few as one sampling iteration. In this work, to optimize text-to-image generative models for task specific rewards and enable fast training and inference, we propose a framework for fine-tuning consistency models via RL. Our framework, called Reinforcement Learning for Consistency Model (RLCM), frames the iterative inference process of a consistency model as an RL procedure. Comparing to RL finetuned diffusion models, RLCM trains significantly faster, improves the quality of the generation measured under the reward objectives, and speeds up the inference procedure by generating high quality images with as few as two inference steps. Experimentally, we show that RLCM can adapt text-to-image consistency models to objectives that are challenging to express with prompting, such as image compressibility, and those derived from human feedback, such as aesthetic quality. Our code is available at https://rlcm.owenoertell.com             

Research goal:
Empirical: yes
Algorithms: Other
Seeds: 3
Code: yes
Env: custom
Hyperparameters: in appendix

### Robotic Manipulation Datasets for Offline Compositional Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper124.html
Keywords:
Abstract: Offline reinforcement learning (RL) is a promising direction that allows RL agents to pre-train on large datasets, avoiding the recurrence of expensive data collection. To advance the field, it is crucial to generate large-scale datasets. Compositional RL is particularly appealing for generating such large datasets, since 1) it permits creating many tasks from few components, 2) the task structure may enable trained agents to solve new tasks by combining relevant learned components, and 3) the compositional dimensions provide a notion of task relatedness. This paper provides four offline RL datasets for simulated robotic manipulation created using the tasks from CompoSuite (Mendez et al., 2022). Each dataset is collected from an agent with a different degree of performance, and consists of million transitions. We provide training and evaluation settings for assessing an agent's ability to learn compositional task policies. Our benchmarking experiments show that current offline RL methods can learn the training tasks to some extent and that compositional methods outperform non-compositional methods. Yet, current methods are unable to extract the compositional structure to generalize to unseen tasks highlighting a need for future research in offline compositional RL.             

Research goal:
Empirical: yes
Algorithms: Other
Seeds: 3
Code: yes
Env: partial
Hyperparameters: in appendix


### ROER: Regularized Optimal Experience Replay
Link: https://rlj.cs.umass.edu/2024/papers/Paper198.html
Keywords:
Abstract: Experience replay serves as a key component in the success of online reinforcement learning (RL). Prioritized experience replay (PER) reweights experiences by the temporal difference (TD) error empirically enhancing the performance. However, few works have explored the motivation of using TD error. In this work, we provide an alternative perspective on TD-error-based reweighting. We show the connections between the experience prioritization and occupancy optimization. By using a regularized RL objective with divergence regularizer and employing its dual form, we show that an optimal solution to the objective is obtained by shifting the distribution of off-policy data in the replay buffer towards the on-policy optimal distribution using TD-error-based occupancy ratios. Our derivation results in a new pipeline of TD error prioritization. We specifically explore the KL divergence as the regularizer and obtain a new form of prioritization scheme, the regularized optimal experience replay (ROER). We evaluate the proposed prioritization scheme with the Soft Actor-Critic (SAC) algorithm in continuous control MuJoCo and DM Control benchmark tasks where our proposed scheme outperforms baselines in 6 out of 11 tasks while the results of the rest match with or do not deviate far from the baselines. Further, using pretraining, ROER achieves noticeable improvement on difficult Antmaze environment where baselines fail, showing applicability to offline-to-online fine-tuning.             

Research goal:
Empirical: yes
Algorithms: Other (new)
Seeds: 20
Code: yes
Env: yes
Hyperparameters: in appendix

### ROIL: Robust Offline Imitation Learning without Trajectories
Link: https://rlj.cs.umass.edu/2024/papers/Paper83.html
Keywords:
Abstract: We study the problem of imitation learning via inverse reinforcement learning where the agent attempts to learn an expert's policy from a dataset of collected state, action tuples. We derive a new Robust model-based Offline Imitation Learning method (ROIL) that mitigates covariate shift by avoiding estimating the expert's occupancy frequency. Frequently in offline settings, there is insufficient data to reliably estimate the expert's occupancy frequency and this leads to models that do not generalize well. Our proposed approach, ROIL, is a method that is guaranteed to recover the expert's occupancy frequency and is efficiently solvable as an LP. We demonstrate ROIL's ability to achieve minimal regret in large environments under covariate shift, such as when the state visitation frequency of the demonstrations does not come from the expert.             

Research goal:
Empirical: no
Algorithms: Other
Seeds: 10
Code: no
Env: no
Hyperparameters: no


### Sample Complexity of Offline Distributionally Robust Linear Markov Decision Processes
Link: https://rlj.cs.umass.edu/2024/papers/Paper189.html
Keywords:
Abstract: In offline reinforcement learning (RL), the absence of active exploration calls for attention on the model robustness to tackle the sim-to-real gap, where the discrepancy between the simulated and deployed environments can significantly undermine the performance of the learned policy. To endow the learned policy with robustness in a sample-efficient manner in the presence of high-dimensional state-action space, this paper considers the sample complexity of distributionally robust linear Markov decision processes (MDPs) with an uncertainty set characterized by the total variation distance using offline data. We develop a pessimistic model-based algorithm and establish its sample complexity bound under minimal data coverage assumptions, which outperforms prior art by at least , where is the feature dimension. We further improve the performance guarantee of the proposed algorithm by incorporating a carefully-designed variance estimator             

Research goal:
Empirical: no
Algorithms: NA
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA


### Semi-Supervised One-Shot Imitation Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper328.html
Keywords:
Abstract: One-shot Imitation Learning (OSIL) aims to imbue AI agents with the ability to learn a new task from a single demonstration. To supervise the learning, OSIL requires a prohibitively large number of paired expert demonstrations: trajectories corresponding to different variations of the same semantic task. To overcome this limitation, we introduce the semi-supervised OSIL problem setting, where the learning agent is presented with a large dataset of tasks with only one demonstration each (unpaired dataset), along with a small dataset of tasks with multiple demonstrations (paired dataset). This presents a more realistic and practical embodiment of few-shot learning and requires the agent to effectively leverage weak supervision. Subsequently, we develop an algorithm applicable to this semi-supervised OSIL setting. Our approach first learns an embedding space where different tasks cluster uniquely. We utilize this embedding space and the clustering it supports to self-generate pairings between trajectories in the large unpaired dataset. Through empirical results, we demonstrate that OSIL models trained on such self-generated pairings (labels) are competitive with OSIL models trained with ground-truth labels, presenting a major advancement in the label-efficiency of OSIL.             

Research goal:
Empirical: yes
Algorithms: Other (new)
Seeds: 3
Code: no (commited to release after acceptance but did not do so for final version)
Env: custom
Hyperparameters: in appendix


### Sequential Decision-Making for Inline Text Autocomplete
Link: https://rlj.cs.umass.edu/2024/papers/Paper119.html
Keywords:
Abstract: Autocomplete suggestions are fundamental to modern text entry systems, with applications in domains such as messaging and email composition. Typically, autocomplete suggestions are generated from a language model with a confidence threshold. However, this threshold does not directly take into account the cognitive burden imposed on the user by surfacing suggestions, such as the effort to switch contexts from typing to reading the suggestion, and the time to decide whether to accept the suggestion. In this paper, we study the problem of improving inline autocomplete suggestions in text entry systems via a sequential decision-making formulation, and use reinforcement learning (RL) to learn suggestion policies through repeated interactions with a target user over time. This formulation allows us to factor cognitive burden into the objective of training an autocomplete model, through a reward function based on text entry speed. We acquired theoretical and experimental evidence that, under certain objectives, the sequential decision-making formulation of the autocomplete problem provides a better suggestion policy than myopic single-step reasoning. However, aligning these objectives with real users requires further exploration. In particular, we hypothesize that the objectives under which sequential decision-making can improve autocomplete systems are not tailored solely to text entry speed, but more broadly to metrics such as user satisfaction and convenience.              

Research goal:
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: no
Env: custom
Hyperparameters: in appendix


### Shield Decomposition for Safe Reinforcement Learning in General Partially Observable Multi-Agent Environments
Link: https://rlj.cs.umass.edu/2024/papers/Paper254.html
Keywords:
Abstract: As Reinforcement Learning is increasingly used in safety-critical systems, it is important to restrict RL agents to only take safe actions. Shielding is a promising approach to this task; however, in multi-agent domains, shielding has previously been restricted to environments where all agents observe the same information. Most real-world tasks do not satisfy this strong assumption. We discuss the theoretical foundations of multi-agent shielding in environments with general partial observability and develop a novel shielding method which is effective in such domains. Through a series of experiments, we show that agents that use our shielding method are able to safely and successfully solve a variety of RL tasks, including tasks in which prior methods cannot be applied.               

Research goal:
Empirical: yes
Algorithms: Other (new)
Seeds: 50
Code: no
Env: custom
Hyperparameters: in appendix


### SplAgger: Split Aggregation for Meta-Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper48.html
Keywords:
Abstract: A core ambition of reinforcement learning (RL) is the creation of agents capable of rapid learning in novel tasks. Meta-RL aims to achieve this by directly learning such agents. Black box methods do so by training off-the-shelf sequence models end-to-end. By contrast, task inference methods explicitly infer a posterior distribution over the unknown task, typically using distinct objectives and sequence models designed to enable task inference. Recent work has shown that task inference methods are not necessary for strong performance. However, it remains unclear whether task inference sequence models are beneficial even when task inference objectives are not. In this paper, we present evidence that task inference sequence models are indeed still beneficial. In particular, we investigate sequence models with permutation invariant aggregation, which exploit the fact that, due to the Markov property, the task posterior does not depend on the order of data. We empirically confirm the advantage of permutation invariant sequence models without the use of task inference objectives. However, we also find, surprisingly, that there are multiple conditions under which permutation variance remains useful. Therefore, we propose SplAgger, which uses both permutation variant and invariant components to achieve the best of both worlds, outperforming all baselines evaluated on continuous control and memory environments. Code is provided at https://github.com/jacooba/hyper.               

Research goal:
Empirical: yes
Algorithms: Other (new)
Seeds: 3
Code: yes
Env: yes
Hyperparameters: in appendix


### Stabilizing Extreme Q-learning by Maclaurin Expansion
Link: https://rlj.cs.umass.edu/2024/papers/Paper171.html
Keywords:
Abstract: In offline reinforcement learning, in-sample learning methods have been widely used to prevent performance degradation caused by evaluating out-of-distribution actions from the dataset. Extreme Q-learning (XQL) employs a loss function based on the assumption that Bellman error follows a Gumbel distribution, enabling it to model the soft optimal value function in an in-sample manner. It has demonstrated strong performance in both offline and online reinforcement learning settings. However, issues remain, such as the instability caused by the exponential term in the loss function and the risk of the error distribution deviating from the Gumbel distribution. Therefore, we propose Maclaurin Expanded Extreme Q-learning to enhance stability. In this method, applying Maclaurin expansion to the loss function in XQL enhances stability against large errors. This approach involves adjusting the modeled value function between the value function under the behavior policy and the soft optimal value function, thus achieving a trade-off between stability and optimality depending on the order of expansion. It also enables adjustment of the error distribution assumption from a normal distribution to a Gumbel distribution. Our method significantly stabilizes learning in online RL tasks from DM Control, where XQL was previously unstable. Additionally, it improves performance in several offline RL tasks from D4RL.               

Research goal:
Empirical: yes
Algorithms: Other (new)
Seeds: 5
Code: no
Env: yes
Hyperparameters: in appendix


### States as goal-directed concepts: an epistemic approach to state-representation learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper275.html
Keywords:
Abstract: Goals fundamentally shape how we experience the world. For example, when we are hungry, we tend to view objects in our environment according to whether or not they are edible (or tasty). Alternatively, when we are cold, we view the very same objects according to their ability to produce heat. Computational theories of learning in cognitive systems, such as reinforcement learning, use state-representations to describe how agents determine behaviorally-relevant features of their environment. However, these approaches typically assume ground-truth state representations that are known to the agent, and reward functions that need to be learned. Here we suggest an alternative approach in which state-representations are not assumed veridical, or even pre-defined, but rather emerge from the agent's goals through interaction with its environment. We illustrate this novel perspective using a rodent odor-guided choice task and discuss its potential role in developing a unified theory of experience based learning in natural and artificial agents.               

Research goal:
Empirical: no
Algorithms: NA
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA


### Surprise-Adaptive Intrinsic Motivation for Unsupervised Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper77.html
Keywords:
Abstract: Both entropy-minimizing and entropy-maximizing objectives for unsupervised reinforcement learning (RL) have been shown to be effective in different environments, depending on the environment's level of natural entropy. However, neither method alone results in an agent that will consistently learn intelligent behavior across environments. In an effort to find a single entropy-based method that will encourage emergent behaviors in any environment, we propose an agent that can adapt its objective online, depending on the entropy conditions it faces in the environment, by framing the choice as a multi-armed bandit problem. We devise a novel intrinsic feedback signal for the bandit, which captures the agent's ability to control the entropy in its environment. We demonstrate that such agents can learn to optimize task returns through entropy control alone in didactic environments for both high- and low-entropy regimes and learn skillful behaviors in certain benchmark tasks.               

Research goal:
Empirical: yes
Algorithms: Other
Seeds: 5
Code: yes
Env: yes
Hyperparameters: in appendix


### SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper111.html
Keywords:
Abstract: Learning to make temporal predictions is a key component of reinforcement learning algorithms. The dominant paradigm for learning predictions from an online stream of data is Temporal Difference (TD) learning. In this work we introduce a new TD algorithm---SwiftTD---that learns more accurate predictions than existing algorithms. SwiftTD combines True Online TD() with per-feature step-size parameters, step-size optimization, a bound on the update to the eligibility vector, and step-size decay. Per-feature step-size parameters and step-size optimization improve credit assignment by increasing the step-size parameters of important signals and reducing them for irrelevant signals. The bound on the update to the eligibility vector prevents overcorrections. Step-size decay reduces step-size parameters if they are too large. We benchmark SwiftTD on the Atari Prediction Benchmark and show that even with linear function approximation it can learn accurate predictions. We further show that SwiftTD performs well across a wide range of its hyperparameters. Finally, we show that SwiftTD can be used in the last layer of neural networks to improve their performance.               

Research goal:
Empirical: no
Algorithms: NA
Seeds: NA
Code: NA
Env: NA
Hyperparameters: NA


### The Cliff of Overcommitment with Policy Gradient Step Sizes
Link: https://rlj.cs.umass.edu/2024/papers/Paper115.html
Keywords:
Abstract: Policy gradient methods form the basis for many successful reinforcement learning algorithms, but their success depends heavily on selecting an appropriate step size and many other hyperparameters. While many adaptive step size methods exist, none are both free of hyperparameter tuning and able to converge quickly to an optimal policy. It is unclear why these methods are insufficient, so we aim to uncover what needs to be addressed to make an effective adaptive step size for policy gradient methods. Through extensive empirical investigation, the results reveal that when the step size is above optimal, the policy overcommits to sub-optimal actions leading to longer training times. These findings suggest the need for a new kind of policy optimization that can prevent or recover from entropy collapses.               

Research goal:
Empirical: yes
Algorithms: NA
Seeds: 20
Code: no
Env: custom
Hyperparameters: in appendix


### The Limits of Pure Exploration in POMDPs: When the Observation Entropy is Enough
Link: https://rlj.cs.umass.edu/2024/papers/Paper95.html
Keywords:
Abstract: The problem of pure exploration in Markov decision processes has been cast as maximizing the entropy over the state distribution induced by the agent's policy, an objective that has been extensively studied. However, little attention has been dedicated to state entropy maximization under partial observability, despite the latter being ubiquitous in applications, e.g., finance and robotics, in which the agent only receives noisy observations of the true state governing the system's dynamics. How can we address state entropy maximization in those domains? In this paper, we study the simple approach of maximizing the entropy over observations in place of true latent states. First, we provide lower and upper bounds to the approximation of the true state entropy that only depends on some properties of the observation function. Then, we show how knowledge of the latter can be exploited to compute a principled regularization of the observation entropy to improve performance. With this work, we provide both a flexible approach to bring advances in state entropy maximization to the POMDP setting and a theoretical characterization of its intrinsic limits.               

Research goal:
Empirical: no
Algorithms: NA
Seeds: 16
Code: no
Env: custom
Hyperparameters: in appendix

