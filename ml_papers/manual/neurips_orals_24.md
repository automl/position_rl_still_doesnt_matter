# NeurIPS Orals 2024

### Reinforcement Learning Under Latent Dynamics: Toward Statistical and Algorithmic Modularity
Link: https://openreview.net/forum?id=qf2uZAdy1N
Keywords: Reinforcement Learning, Representation Learning, Latent Dynamics, Function Approximation
Abstract: Real-world applications of reinforcement learning often involve environments where agents operate on complex, high-dimensional observations, but the underlying (``latent'') dynamics are comparatively simple. However, beyond restrictive settings such as tabular latent dynamics, the fundamental statistical requirements and algorithmic principles for reinforcement learning under latent dynamics are poorly understood. This paper addresses the question of reinforcement learning under general latent dynamics from a statistical and algorithmic perspective. On the statistical side, our main negative result shows that most well-studied settings for reinforcement learning with function approximation become intractable when composed with rich observations; we complement this with a positive result, identifying latent pushforward coverability as a general condition that enables statistical tractability. Algorithmically, we develop provably efficient observable-to-latent reductions ---that is, reductions that transform an arbitrary algorithm for the latent MDP into an algorithm that can operate on rich observations--- in two settings: one where the agent has access to hindsight observations of the latent dynamics (Lee et al., 2023) and one where the agent can estimate self-predictive latent models (Schwarzer et al., 2020). Together, our results serve as a first step toward a unified statistical and algorithmic theory for reinforcement learning under latent dynamics.

Research goal: statistical requirements for RL under latent dynamics
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Aligner: Efficient Alignment by Learning to Correct
Link: https://openreview.net/forum?id=kq166jACVP
Keywords: Large Language Models, Alignment, Reinforcement Learning from Human Feedback
Abstract: With the rapid development of large language models (LLMs) and ever-evolving practical requirements, finding an efficient and effective alignment method has never been more critical. However, the tension between the complexity of current alignment methods and the need for rapid iteration in deployment scenarios necessitates the development of a model-agnostic alignment approach that can operate under these constraints. In this paper, we introduce Aligner, a novel and simple alignment paradigm that learns the correctional residuals between preferred and dispreferred answers using a small model. Designed as a model-agnostic, plug-and-play module, Aligner can be directly applied to various open-source and API-based models with only one-off training, making it suitable for rapid iteration. Notably, Aligner can be applied to any powerful, large-scale upstream models. Moreover, it can even iteratively bootstrap the upstream models using corrected responses as synthetic human preference data, breaking through the model's performance ceiling. Our experiments demonstrate performance improvements by deploying the same Aligner model across 11 different LLMs, evaluated on the 3H dimensions (helpfulness, harmlessness, and honesty). Specifically, Aligner-7B has achieved an average improvement of 68.9% in helpfulness and 22.8% in harmlessness across the tested LLMs while also effectively reducing hallucination. In the Alpaca-Eval leaderboard, stacking Aligner-2B on GPT-4 Turbo improved its LC Win Rate from 55.0% to 58.3%, surpassing GPT-4 Omni's 57.5% Win Rate (community report).

Research goal: better llm alignment
Empirical: non-rl
Algorithms: -
Seeds: -
Code: yes
Env: -
Hyperparameters: -

### Learning Formal Mathematics From Intrinsic Motivation
Link: https://openreview.net/forum?id=uNKlTQ8mBD
Keywords: reasoning, reinforcement learning, formal mathematics, logic
Abstract: How did humanity coax mathematics from the aether? We explore the Platonic view that mathematics can be discovered from its axioms---a game of conjecture and proof. We describe an agent that jointly learns to pose challenging problems for itself (conjecturing) and solve them (theorem proving). Given a mathematical domain axiomatized in dependent type theory, we first combine methods for constrained decoding and type-directed synthesis to sample valid conjectures from a language model. Our method guarantees well-formed conjectures by construction, even as we start with a randomly initialized model. We use the same model to represent a policy and value function for guiding proof search. Our agent targets generating hard but provable conjectures --- a moving target, since its own theorem proving ability also improves as it trains. We propose novel methods for hindsight relabeling on proof search trees to significantly improve the agent's sample efficiency in both tasks. Experiments on 3 axiomatic domains (propositional logic, arithmetic and group theory) demonstrate that our agent can bootstrap from only the axioms, self-improving in generating true and challenging conjectures and in finding proofs.

Research goal: automated proofs
Empirical: yes
Algorithms: -
Seeds: 3
Code: yes
Env: in appendix
Hyperparameters: no

### Span-Based Optimal Sample Complexity for Weakly Communicating and General Average Reward MDPs
Link: https://openreview.net/forum?id=pGEY8JQ3qx
Keywords: reinforcement learning theory, average reward, sample complexity
Abstract: We study the sample complexity of learning an -optimal policy in an average-reward Markov decision process (MDP) under a generative model. For weakly communicating MDPs, we establish the complexity bound , where is the span of the bias function of the optimal policy and is the cardinality of the state-action space. Our result is the first that is minimax optimal (up to log factors) in all parameters, and , improving on existing work that either assumes uniformly bounded mixing times for all policies or has suboptimal dependence on the parameters. We also initiate the study of sample complexity in general (multichain) average-reward MDPs. We argue a new transient time parameter is necessary, establish an complexity bound, and prove a matching (up to log factors) minimax lower bound. Both results are based on reducing the average-reward MDP to a discounted MDP, which requires new ideas in the general setting. To optimally analyze this reduction, we develop improved bounds for -discounted MDPs, showing that and samples suffice to learn -optimal policies in weakly communicating and in general MDPs, respectively. Both these results circumvent the well-known minimax lower bound of for -discounted MDPs, and establish a quadratic rather than cubic horizon dependence for a fixed MDP instance.

Research goal: sample complexity in average reward MDPs
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Maximum Entropy Inverse Reinforcement Learning of Diffusion Models with Energy-Based Models
Link: https://openreview.net/forum?id=V0oJaLqY4E
Keywords: diffusion models, inverse reinforcement learning, dynamic programming, reinforcement learning, generative modeling
Abstract: We present a maximum entropy inverse reinforcement learning (IRL) approach for improving the sample quality of diffusion generative models, especially when the number of generation time steps is small. Similar to how IRL trains a policy based on the reward function learned from expert demonstrations, we train (or fine-tune) a diffusion model using the log probability density estimated from training data. Since we employ an energy-based model (EBM) to represent the log density, our approach boils down to the joint training of a diffusion model and an EBM. Our IRL formulation, named Diffusion by Maximum Entropy IRL (DxMI), is a minimax problem that reaches equilibrium when both models converge to the data distribution. The entropy maximization plays a key role in DxMI, facilitating the exploration of the diffusion model and ensuring the convergence of the EBM. We also propose Diffusion by Dynamic Programming (DxDP), a novel reinforcement learning algorithm for diffusion models, as a subroutine in DxMI. DxDP makes the diffusion model update in DxMI efficient by transforming the original problem into an optimal control formulation where value functions replace back-propagation in time. Our empirical studies show that diffusion models fine-tuned using DxMI can generate high-quality samples in as few as 4 and 10 steps. Additionally, DxMI enables the training of an EBM without MCMC, stabilizing EBM training dynamics and enhancing anomaly detection performance.

Research goal: inverse RL for learning diffusion models
Empirical: yes
Algorithms: -
Seeds: 5
Code: yes
Env: -
Hyperparameters: partial

### Improving Environment Novelty Quantification for Effective Unsupervised Environment Design
Link: https://openreview.net/forum?id=UdxpjKO2F9
Keywords: Unsupervised Environment Design, Novelty-driven Autocurricula
Abstract: Unsupervised Environment Design (UED) formalizes the problem of autocurricula through interactive training between a teacher agent and a student agent. The teacher generates new training environments with high learning potential, curating an adaptive curriculum that strengthens the student's ability to handle unseen scenarios. Existing UED methods mainly rely on regret, a metric that measures the difference between the agent's optimal and actual performance, to guide curriculum design. Regret-driven methods generate curricula that progressively increase environment complexity for the student but overlook environment novelty — a critical element for enhancing an agent's generalizability. Measuring environment novelty is especially challenging due to the underspecified nature of environment parameters in UED, and existing approaches face significant limitations. To address this, this paper introduces the Coverage-based Evaluation of Novelty In Environment (CENIE) framework. CENIE proposes a scalable, domain-agnostic, and curriculum-aware approach to quantifying environment novelty by leveraging the student's state-action space coverage from previous curriculum experiences. We then propose an implementation of CENIE that models this coverage and measures environment novelty using Gaussian Mixture Models. By integrating both regret and novelty as complementary objectives for curriculum design, CENIE facilitates effective exploration across the state-action space while progressively increasing curriculum complexity. Empirical evaluations demonstrate that augmenting existing regret-based UED algorithms with CENIE achieves state-of-the-art performance across multiple benchmarks, underscoring the effectiveness of novelty-driven autocurricula for robust generalization.

Research goal: novelty for curriculum generation
Empirical: yes
Algorithms: PPO
Seeds: 5
Code: no
Env: no
Hyperparameters: in appendix

### Enhancing Preference-based Linear Bandits via Human Response Time
Link: https://openreview.net/forum?id=aIPwlkdOut
Keywords: human response time, preference learning, linear bandits, dueling bandits, psychology, economics
Abstract: Interactive preference learning systems infer human preferences by presenting queries as pairs of options and collecting binary choices. Although binary choices are simple and widely used, they provide limited information about preference strength. To address this, we leverage human response times, which are inversely related to preference strength, as an additional signal. We propose a computationally efficient method that combines choices and response times to estimate human utility functions, grounded in the EZ diffusion model from psychology. Theoretical and empirical analyses show that for queries with strong preferences, response times complement choices by providing extra information about preference strength, leading to significantly improved utility estimation. We incorporate this estimator into preference-based linear bandits for fixed-budget best-arm identification. Simulations on three real-world datasets demonstrate that using response times significantly accelerates preference learning compared to choice-only approaches. Additional materials, such as code, slides, and talk video, are available at https://shenlirobot.github.io/pages/NeurIPS24.html.

Research goal: integrating human response times into solving bandits
Empirical: bandit
Algorithms: -
Seeds: 300
Code: yes
Env: -
Hyperparameters: -

### The Sample-Communication Complexity Trade-off in Federated Q-Learning
Link: https://openreview.net/forum?id=6YIpvnkjUK
Keywords: Federated Q learning, Communication Efficiency
Abstract: We consider the problem of Federated Q-learning, where agents aim to collaboratively learn the optimal Q-function of an unknown infinite horizon Markov Decision Process with finite state and action spaces. We investigate the trade-off between sample and communication complexity for the widely used class of intermittent communication algorithms. We first establish the converse result, where we show that any Federated Q-learning that offers a linear speedup with respect to number of agents in sample complexity needs to incur a communication cost of at least , where is the discount factor. We also propose a new Federated Q-learning algorithm, called Fed-DVR-Q, which is the first Federated Q-learning algorithm to simultaneously achieve order-optimal sample and communication complexities. Thus, together these results provide a complete characterization of the sample-communication complexity trade-off in Federated Q-learning.

Research goal: sample complexity vs communication in federated Q-Learning
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Statistical Efficiency of Distributional Temporal Difference Learning
Link: https://openreview.net/forum?id=eWUM5hRYgH
Keywords: Distributional Reinforcement Learning, Distributional Temporal Difference Learning, Sample Complexity
Abstract: Distributional reinforcement learning (DRL) has achieved empirical success in various domains. One of the core tasks in the field of DRL is distributional policy evaluation, which involves estimating the return distribution for a given policy . The distributional temporal difference learning has been accordingly proposed, which is an extension of the temporal difference learning (TD) in the classic RL area. In the tabular case, Rowland et al. [2018] and Rowland et al. [2023] proved the asymptotic convergence of two instances of distributional TD, namely categorical temporal difference learning (CTD) and quantile temporal difference learning (QTD), respectively. In this paper, we go a step further and analyze the finite-sample performance of distributional TD. To facilitate theoretical analysis, we propose a non-parametric distributional TD learning (NTD). For a -discounted infinite-horizon tabular Markov decision process, we show that for NTD we need iterations to achieve an -optimal estimator with high probability, when the estimation error is measured by the -Wasserstein distance. This sample complexity bound is minimax optimal (up to logarithmic factors) in the case of the -Wasserstein distance. To achieve this, we establish a novel Freedman's inequality in Hilbert spaces, which would be of independent interest. In addition, we revisit CTD, showing that the same non-asymptotic convergence bounds hold for CTD in the case of the -Wasserstein distance.

Research goal: sample complexity in distributional rl
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### RL-GPT: Integrating Reinforcement Learning and Code-as-policy
Link: https://openreview.net/forum?id=LEzx6QRkRH
Keywords: Agent, Large Language Models (LLMs), Reinforcement Learning (RL)
Abstract: Large Language Models (LLMs) have demonstrated proficiency in utilizing various tools by coding, yet they face limitations in handling intricate logic and precise control. In embodied tasks, high-level planning is amenable to direct coding, while low-level actions often necessitate task-specific refinement, such as Reinforcement Learning (RL). To seamlessly integrate both modalities, we introduce a two-level hierarchical framework, RL-GPT, comprising a slow agent and a fast agent. The slow agent analyzes actions suitable for coding, while the fast agent executes coding tasks. This decomposition effectively focuses each agent on specific tasks, proving highly efficient within our pipeline. Our approach outperforms traditional RL methods and existing GPT agents, demonstrating superior efficiency. In the Minecraft game, it rapidly obtains diamonds within a single day on an RTX3090. Additionally, it achieves SOTA performance across all designated MineDojo tasks.

Research goal: use llms to code rl policies
Empirical: yes
Algorithms: Dreamer, PPO
Seeds: 1
Code: no
Env: yes
Hyperparameters: partial in appendix

### Policy Learning from Tutorial Books via Understanding, Rehearsing and Introspecting
Link: https://openreview.net/forum?id=Ddak3nSqQM
Keywords: Reinforcement Learning, Large Language Model, Agent, Retrieval Augmented Generation
Abstract: When humans need to learn a new skill, we can acquire knowledge through written books, including textbooks, tutorials, etc. However, current research for decision-making, like reinforcement learning (RL), has primarily required numerous real interactions with the target environment to learn a skill, while failing to utilize the existing knowledge already summarized in the text. The success of Large Language Models (LLMs) sheds light on utilizing such knowledge behind the books. In this paper, we discuss a new policy learning problem called Policy Learning from tutorial Books (PLfB) upon the shoulders of LLMs’ systems, which aims to leverage rich resources such as tutorial books to derive a policy network. Inspired by how humans learn from books, we solve the problem via a three-stage framework: Understanding, Rehearsing, and Introspecting (URI). In particular, it first rehearses decision-making trajectories based on the derived knowledge after understanding the books, then introspects in the imaginary dataset to distill a policy network. We build two benchmarks for PLfB~based on Tic-Tac-Toe and Football games. In experiment, URI's policy achieves at least 44% net win rate against GPT-based agents without any real data; In Football game, which is a complex scenario, URI's policy beat the built-in AIs with a 37% while using GPT-based agent can only achieve a 6% winning rate. The project page: https://plfb-football.github.io.

Research goal: learning policies from written tutorials
Empirical: yes
Algorithms: -
Seeds: 3
Code: no
Env: no
Hyperparameters: in appendix

### Decompose, Analyze and Rethink: Solving Intricate Problems with Human-like Reasoning Cycle
Link: https://openreview.net/forum?id=NPKZF1WDjZ
Keywords: Reasoning Tree, Large Language Models, Question Decomposition, Rationale Updating
Abstract: In this paper, we introduce DeAR (Decompose-Analyze-Rethink), a framework that iteratively builds a reasoning tree to tackle intricate problems within a single large language model (LLM). Unlike approaches that extend or search for rationales, DeAR is featured by 1) adopting a tree-based question decomposition manner to plan the organization of rationales, which mimics the logical planning inherent in human cognition; 2) globally updating the rationales at each reasoning step through natural language feedback. Specifically, the Decompose stage decomposes the question into simpler sub-questions, storing them as new nodes; the Analyze stage generates and self-checks rationales for sub-questions at each node evel; and the Rethink stage updates parent-node rationales based on feedback from their child nodes. By generating and updating the reasoning process from a more global perspective, DeAR constructs more adaptive and accurate logical structures for complex problems, facilitating timely error correction compared to rationale-extension and search-based approaches such as Tree-of-Thoughts (ToT) and Graph-of-Thoughts (GoT). We conduct extensive experiments on three reasoning benchmarks, including ScienceQA, StrategyQA, and GSM8K, which cover a variety of reasoning tasks, demonstrating that our approach significantly reduces logical errors and enhances performance across various LLMs. Furthermore, we validate that DeAR is an efficient method that achieves a superior trade-off between accuracy and reasoning time compared to ToT and GoT.

Research goal: better reasoning in llms
Empirical: yes
Algorithms: -
Seeds: -
Code: yes
Env: -
Hyperparameters: -

### Learning rigid-body simulators over implicit shapes for large-scale scenes and vision
Link: https://openreview.net/forum?id=QDYts5dYgq
Keywords: graph networks, learned simulation, physics, rigid body simulation, scaling
Abstract: Simulating large scenes with many rigid objects is crucial for a variety of applications, such as robotics, engineering, film and video games. Rigid interactions are notoriously hard to model: small changes to the initial state or the simulation parameters can lead to large changes in the final state. Recently, learned simulators based on graph networks (GNNs) were developed as an alternative to hand-designed simulators like MuJoCo and Bullet. They are able to accurately capture dynamics of real objects directly from real-world observations. However, current state-of-the-art learned simulators operate on meshes and scale poorly to scenes with many objects or detailed shapes. Here we present SDF-Sim, the first learned rigid-body simulator designed for scale. We use learned signed-distance functions (SDFs) to represent the object shapes and to speed up distance computation. We design the simulator to leverage SDFs and avoid the fundamental bottleneck of the previous simulators associated with collision detection. For the first time in literature, we demonstrate that we can scale the GNN-based simulators to scenes with hundreds of objects and up to 1.1 million nodes, where mesh-based approaches run out of memory. Finally, we show that SDF-Sim can be applied to real world scenes by extracting SDFs from multi-view images.

Research goal: learning environments involving rigid bodies
Empirical: yes
Algorithms: -
Seeds: 3
Code: no
Env: -
Hyperparameters: -