# RLC Awards 2024

### A Super-human Vision-based Reinforcement Learning Agent for Autonomous Racing in Gran Turismo
Link: https://rlj.cs.umass.edu/2024/papers/Paper213.html
Keywords: reinforcement learning, rl for games
Abstract: Racing autonomous cars faster than the best human drivers has been a longstanding grand challenge for the fields of Artificial Intelligence and robotics. Recently, an end-to-end deep reinforcement learning agent met this challenge in a high-fidelity racing simulator, Gran Turismo. However, this agent relied on global features that require instrumentation external to the car. This paper introduces, to the best of our knowledge, the first super-human car racing agent whose sensor input is purely local to the car, namely pixels from an ego-centric camera view and quantities that can be sensed from on-board the car, such as the car's velocity. By leveraging global features only at training time, the learned agent is able to outperform the best human drivers in time trial (one car on the track at a time) races using only local input features. The resulting agent is evaluated in Gran Turismo 7 on multiple tracks and cars. Detailed ablation experiments demonstrate the agent's strong reliance on visual inputs, making it the first vision-based super-human car racing agent.

Research goal: RL for racing games
Empirical: yes
Algorithms: SAC
Seeds: 5
Code: no
Env: not accessible
Hyperparameters: in appendix

### On Welfare-Centric Fair Reinforcement Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper133.html
Keywords: fair RL, safe RL
Abstract: We propose a welfare-centric fair reinforcement-learning setting, in which an agent enjoys vector-valued reward from a set of beneficiaries. Given a welfare function W(·), the task is to select a policy π̂ that approximately optimizes the welfare of theirvalue functions from start state s0 , i.e., π̂ ≈ argmaxπ W Vπ1 (s0 ), Vπ2 (s0 ), . . . , Vπg (s0 ) . We find that welfare-optimal policies are stochastic and start-state dependent. Whether individual actions are mistakes depends on the policy, thus mistake bounds, regret analysis, and PAC-MDP learning do not readily generalize to our setting. We develop the adversarial-fair KWIK (Kwik-Af) learning model, wherein at each timestep, an agent either takes an exploration action or outputs an exploitation policy, such that the number of exploration actions is bounded and each exploitation policy is ε-welfare optimal. Finally, we reduce PAC-MDP to Kwik-Af, introduce the Equitable Explicit Explore Exploit (E4) learner, and show that it Kwik-Af learns.

Research goal: a setting for fair RL
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -


### SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning
Link: https://rlj.cs.umass.edu/2024/papers/Paper111.html
Keywords: online learning, reinforcement learning
Abstract: Learning to make temporal predictions is a key component of reinforcement learning algorithms. The dominant paradigm for learning predictions from an online stream of data is Temporal Difference (TD) learning. In this work we introduce a new TD algorithm---SwiftTD---that learns more accurate predictions than existing algorithms. SwiftTD combines True Online TD() with per-feature step-size parameters, step-size optimization, a bound on the update to the eligibility vector, and step-size decay. Per-feature step-size parameters and step-size optimization improve credit assignment by increasing the step-size parameters of important signals and reducing them for irrelevant signals. The bound on the update to the eligibility vector prevents overcorrections. Step-size decay reduces step-size parameters if they are too large. We benchmark SwiftTD on the Atari Prediction Benchmark and show that even with linear function approximation it can learn accurate predictions. We further show that SwiftTD performs well across a wide range of its hyperparameters. Finally, we show that SwiftTD can be used in the last layer of neural networks to improve their performance.

Research goal: improving TD learning
Empirical: yes
Algorithms: TD
Seeds: 15
Code: no
Env: no
Hyperparameters: yes

### An Open-Loop Baseline for Reinforcement Learning Locomotion Tasks
Link: https://rlj.cs.umass.edu/2024/papers/Paper18.html
Keywords: reinforcement learning, robotics, baselines
Abstract: In search of a simple baseline for Deep Reinforcement Learning in locomotion tasks, we propose a model-free open-loop strategy. By leveraging prior knowledge and the elegance of simple oscillators to generate periodic joint motions, it achieves respectable performance in five different locomotion environments, with a number of tunable parameters that is a tiny fraction of the thousands typically required by DRL algorithms. We conduct two additional experiments using open-loop oscillators to identify current shortcomings of these algorithms. Our results show that, compared to the baseline, DRL is more prone to performance degradation when exposed to sensor noise or failure. Furthermore, we demonstrate a successful transfer from simulation to reality using an elastic quadruped, where RL fails without randomization or reward engineering. Overall, the proposed baseline and associated experiments highlight the existing limitations of DRL for robotic applications, provide insights on how to address them, and encourage reflection on the costs of complexity and generality.

Research goal: baseline for locomotion
Empirical: yes
Algorithms: SAC, PPO, DDOG, ARS
Seeds: 10
Code: in appendix
Env: yes
Hyperparameters: no

### Bad Habits: Policy Confounding and Out-of-Trajectory Generalization in RL
Link: https://rlj.cs.umass.edu/2024/papers/Paper216.html
Keywords: reinforcement learning, generalization
Abstract: Reinforcement learning agents tend to develop habits that are effective only under specific policies. Following an initial exploration phase where agents try out different actions, they eventually converge onto a particular policy. As this occurs, the distribution over state-action trajectories becomes narrower, leading agents to repeatedly experience the same transitions. This repetitive exposure fosters spurious correlations between certain observations and rewards. Agents may then pick up on these correlations and develop simplistic habits tailored to the specific set of trajectories dictated by their policy. The problem is that these habits may yield incorrect outcomes when agents are forced to deviate from their typical trajectories, prompted by changes in the environment. This paper presents a mathematical characterization of this phenomenon, termed policy confounding, and illustrates, through a series of examples, the circumstances under which it occurs.

Research goal: understanding generalization
Empirical: yes
Algorithms: DQN, PPO
Seeds: 10
Code: no
Env: no
Hyperparameters: yes

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

### Posterior Sampling for Continuing Environments
Link: https://rlj.cs.umass.edu/2024/papers/Paper277.html
Keywords: continual learning, reinforcement learning
Abstract: Existing posterior sampling algorithms for continuing reinforcement learning (RL) rely on maintaining state-action visitation counts, making them unsuitable for complex environments with high-dimensional state spaces. We develop the first extension of posterior sampling for RL (PSRL) that is suited for a continuing agent-environment interface and integrates naturally into scalable agent designs. Our approach, continuing PSRL (CPSRL), determines when to resample a new model of the environment from the posterior distribution based on a simple randomization scheme. We establish an bound on the Bayesian regret in the tabular setting, where is the number of environment states, is the number of actions, and denotes the {\it reward averaging time}, which is a bound on the duration required to accurately estimate the average reward of any policy. Our work is the first to formalize and rigorously analyze this random resampling approach. Our simulations demonstrate CPSRL's effectiveness in high-dimensional state spaces where traditional algorithms fail.

Research goal: benchmarking RL for aquatic navigation
Empirical: toy
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -