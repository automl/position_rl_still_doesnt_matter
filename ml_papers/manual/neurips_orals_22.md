# NeurIPS 2022 Orals


### Minimax Regret for Cascading Bandits
Link: https://openreview.net/forum?id=EgMbj9yWrMI
Keywords: online learning-to-rank, cascading bandits, linear stochastic bandits
Abstract: Cascading bandits is a natural and popular model that frames the task of learning to rank from Bernoulli click feedback in a bandit setting. For the case of unstructured rewards, we prove matching upper and lower bounds for the problem-independent (i.e., gap-free) regret, both of which strictly improve the best known. A key observation is that the hard instances of this problem are those with small mean rewards, i.e., the small click-through rates that are most relevant in practice. Based on this, and the fact that small mean implies small variance for Bernoullis, our key technical result shows that variance-aware confidence sets derived from the Bernstein and Chernoff bounds lead to optimal algorithms (up to log terms), whereas Hoeffding-based algorithms suffer order-wise suboptimal regret. This sharply contrasts with the standard (non-cascading) bandit setting, where the variance-aware algorithms only improve constants. In light of this and as an additional contribution, we propose a variance-aware algorithm for the structured case of linear rewards and show its regret strictly improves the state-of-the-art.

Research goal: Optimal bias-variance tradeoff in offline RL as opposed to manual tuning
Empirical: other
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting
Link: https://openreview.net/forum?id=XvI6h-s4un
Keywords: Reinforcement Learning, Language Models, Reward Maximization, Distribution Matching, Energy Based Models, Controlled Text Generation
Abstract: The availability of large pre-trained models is changing the landscape of Machine Learning research and practice, moving from a "training from scratch" to a "fine-tuning'' paradigm. While in some applications the goal is to "nudge'' the pre-trained distribution towards preferred outputs, in others it is to steer it towards a different distribution over the sample space. Two main paradigms have emerged to tackle this challenge: Reward Maximization (RM) and, more recently, Distribution Matching (DM). RM applies standard Reinforcement Learning (RL) techniques, such as Policy Gradients, to gradually increase the reward signal. DM prescribes to first make explicit the target distribution that the model is fine-tuned to approximate. Here we explore the theoretical connections between the two paradigms and show that methods such as KL-control developed in the RM paradigm can also be construed as belonging to DM. We further observe that while DM differs from RM, it can suffer from similar training difficulties, such as high gradient variance. We leverage connections between the two paradigms to import the concept of baseline into DM methods. We empirically validate the benefits of adding a baseline on an array of controllable language generation tasks such as constraining topic, sentiment, and gender distributions in texts sampled from a language model. We observe superior performance in terms of constraint satisfaction, stability, and sample efficiency.

Research goal: Preventing forgetting when finetuning LLMs
Empirical: yes
Algorithms: Reinforce
Seeds: 3
Code: yes
Env: yes
Hyperparameters: in appendix

### Shield Decentralization for Safe Multi-Agent Reinforcement Learning
Link: https://openreview.net/forum?id=JO9o3DgV9l2
Keywords: safety, shielding, reinforcement learning, synthesis, multi agent
Abstract: Learning safe solutions is an important but challenging problem in multi-agent reinforcement learning (MARL). Shielded reinforcement learning is one approach for preventing agents from choosing unsafe actions. Current shielded reinforcement learning methods for MARL make strong assumptions about communication and full observability. In this work, we extend the formalization of the shielded reinforcement learning problem to a decentralized multi-agent setting. We then present an algorithm for decomposition of a centralized shield, allowing shields to be used in such decentralized, communication-free environments. Our results show that agents equipped with decentralized shields perform comparably to agents with centralized shields in several tasks, allowing shielding to be used in environments with decentralized training and execution for the first time.

Research goal: Learning safe policies in MARL
Empirical: yes
Algorithms: -
Seeds: 10
Code: yes
Env: yes
Hyperparameters: in appendix

### On the Complexity of Adversarial Decision Making
Link: https://openreview.net/forum?id=pgBpQYss2ba
Keywords: online learning, adversarial learning, complexity
Abstract: A central problem in online learning and decision making---from bandits to reinforcement learning---is to understand what modeling assumptions lead to sample-efficient learning guarantees. We consider a general adversarial decision making framework that encompasses (structured) bandit problems with adversarial rewards and reinforcement learning problems with adversarial dynamics. Our main result is to show---via new upper and lower bounds---that the Decision-Estimation Coefficient, a complexity measure introduced by Foster et al. in the stochastic counterpart to our setting, is necessary and sufficient to obtain low regret for adversarial decision making. However, compared to the stochastic setting, one must apply the Decision-Estimation Coefficient to the convex hull of the class of models (or, hypotheses) under consideration. This establishes that the price of accommodating adversarial rewards or dynamics is governed by the behavior of the model class under convexification, and recovers a number of existing results --both positive and negative. En route to obtaining these guarantees, we provide new structural results that connect the Decision-Estimation Coefficient to variants of other well-known complexity measures, including the Information Ratio of Russo and Van Roy and the Exploration-by-Optimization objective of Lattimore and György.

Research goal: Regret bounds in adversarial decision making
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Dynamic Inverse Reinforcement Learning for Characterizing Animal Behavior
Link: https://openreview.net/forum?id=nosngu5XwY9
Keywords: Neuroscience, decision-making, inverse reinforcement learning
Abstract: Understanding decision-making is a core goal in both neuroscience and psychology, and computational models have often been helpful in the pursuit of this goal. While many models have been developed for characterizing behavior in binary decision-making and bandit tasks, comparatively little work has focused on animal decision-making in more complex tasks, such as navigation through a maze. Inverse reinforcement learning (IRL) is a promising approach  for understanding such behavior, as it aims to infer the unknown reward function of an agent from its observed trajectories through state space. However, IRL has yet to be widely applied in neuroscience. One potential reason for this is that existing IRL frameworks assume that an agent's reward function is fixed over time. To address this shortcoming, we introduce dynamic inverse reinforcement learning (DIRL), a novel IRL framework that allows for time-varying intrinsic rewards. Our method parametrizes the unknown reward function as a time-varying linear combination of spatial reward maps (which we refer to as "goal maps"). We develop an efficient inference method for recovering this dynamic reward function from behavioral data.  We demonstrate DIRL in simulated experiments and then apply it to a dataset of mice exploring a labyrinth. Our method returns interpretable reward functions for two separate cohorts of mice, and provides a novel characterization of exploratory behavior. We expect DIRL to have broad applicability in neuroscience, and to facilitate the design of biologically-inspired reward functions for training artificial agents.

Research goal: Learning decision making from animals
Empirical: yes
Algorithms: -
Seeds: 1
Code: yes
Env: no
Hyperparameters: in appendix

### Near-Optimal Collaborative Learning in Bandits
Link: https://openreview.net/forum?id=2xfJ26BuFP
Keywords: collaborative learning, multi-armed bandit, centralized learning, communication, elimination based-algorithm, data-driven sampling
Abstract: This paper introduces a general multi-agent bandit model in which each agent is facing a finite set of arms and may communicate with other agents through a central controller in order to identify -in pure exploration- or play -in regret minimization- its optimal arm. The twist is that the optimal arm for each agent is the arm with largest expected mixed reward, where the mixed reward of an arm is a weighted sum of the rewards of this arm for all agents. This makes communication between agents often necessary. This general setting allows to recover and extend several recent models for collaborative bandit learning, including the recently proposed federated learning with personalization [Shi et al., 2021]. In this paper, we provide new lower bounds on the sample complexity of pure exploration and on the regret. We then propose a near-optimal algorithm for pure exploration. This algorithm is based on phased elimination with two novel ingredients: a data-dependent sampling scheme within each phase, aimed at matching a relaxation of the lower bound.

Research goal: Mulit-agent multi-armed bandits
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Efficient Methods for Non-stationary Online Learning
Link: https://openreview.net/forum?id=5Ap96waLr8A
Keywords: non-stationary online learning, dynamic regret, adaptive regret, online ensemble, projection complexity
Abstract: Non-stationary online learning has drawn much attention in recent years. In particular, \emph{dynamic regret} and \emph{adaptive regret} are proposed as two principled performance measures for online convex optimization in non-stationary environments. To optimize them, a two-layer online ensemble is usually deployed due to the inherent uncertainty of the non-stationarity, in which a group of base-learners are maintained and a meta-algorithm is employed to track the best one on the fly. However, the two-layer structure raises the concern about the computational complexity--those methods typically maintain base-learners simultaneously for a round online game and thus perform multiple projections onto the feasible domain per round, which becomes the computational bottleneck when the domain is complicated. In this paper, we present efficient methods for optimizing dynamic regret and adaptive regret, which reduce the number of projections per round from to .  Moreover, our obtained algorithms require only one gradient query and one function evaluation at each round. Our technique hinges on the reduction mechanism developed in parameter-free online learning and requires non-trivial twists on non-stationary online methods. Empirical studies verify our theoretical findings.

Research goal: Easier non-stationary online learning
Empirical: toy
Algorithms: -
Seeds: 5
Code: yes
Env: yes
Hyperparameters: partial

### Adaptively Exploiting d-Separators with Causal Bandits
Link: https://openreview.net/forum?id=-e2SBzFDE8x
Keywords: bandit, causal bandit, adaptive, d-separation, online
Abstract: Multi-armed bandit problems provide a framework to identify the optimal intervention over a sequence of repeated experiments. Without additional assumptions, minimax optimal performance (measured by cumulative regret) is well-understood. With access to additional observed variables that d-separate the intervention from the outcome (i.e., they are a d-separator), recent "causal bandit" algorithms provably incur less regret. However, in practice it is desirable to be agnostic to whether observed variables are a d-separator. Ideally, an algorithm should be adaptive; that is, perform nearly as well as an algorithm with oracle knowledge of the presence or absence of a d-separator. In this work, we formalize and study this notion of adaptivity, and provide a novel algorithm that simultaneously achieves (a) optimal regret when a d-separator is observed, improving on classical minimax algorithms, and (b) significantly smaller regret than recent causal bandit algorithms when the observed variables are not a d-separator. Crucially, our algorithm does not require any oracle knowledge of whether a d-separator is observed. We also generalize this adaptivity to other conditions, such as the front-door criterion.

Research goal: Adaptive multi-armed bandits
Empirical: toy
Algorithms: -
Seeds: -
Code: yes
Env: -
Hyperparameters: -

### Skills Regularized Task Decomposition for Multi-task Offline Reinforcement Learning
Link: https://openreview.net/forum?id=uuaMrewU9Kk
Keywords: mutli-task reinforcement learning, offline reinforcement learning, task inference, skill embedding
Abstract: Reinforcement learning (RL) with diverse offline datasets can have the advantage of leveraging the relation of multiple tasks and the common skills learned across those tasks, hence allowing us to deal with real-world complex problems efficiently in a data-driven way.  In offline RL where only offline data is used and online interaction with the environment is restricted, it is yet difficult to achieve the optimal policy for multiple tasks, especially when the data quality varies for the tasks. In this paper, we present a skill-based multi-task RL technique on heterogeneous datasets that are generated by behavior policies of different quality. To learn the shareable knowledge across those datasets effectively, we employ a task decomposition method for which common skills are jointly learned and used as guidance to reformulate a task in shared and achievable subtasks. In this joint learning, we use Wasserstein Auto-Encoder (WAE) to represent both skills and tasks on the same latent space and use the quality-weighted loss as a regularization term to induce tasks to be decomposed into subtasks that are more consistent with high-quality skills than others. To improve the performance of offline RL agents learned on the latent space, we also augment datasets with imaginary trajectories relevant to high-quality skills for each task. Through experiments, we show that our multi-task offline RL approach is robust to different-quality datasets and it outperforms other state-of-the-art algorithms for several robotic manipulation tasks and drone navigation tasks.

Research goal: Offline learning across multiple datasets
Empirical: yes
Algorithms: TD3
Seeds: 3
Code: only in appendix
Env: no
Hyperparameters: in appendix

### Modeling Human Exploration Through Resource-Rational Reinforcement Learning
Link: https://openreview.net/forum?id=W1MUJv5zaXP
Keywords: Exploration, Meta-Learning, Cognitive Science, Resource-Rationality
Abstract: Equipping artificial agents with useful exploration mechanisms remains a challenge to this day. Humans, on the other hand, seem to manage the trade-off between exploration and exploitation effortlessly. In the present article, we put forward the hypothesis that they accomplish this by making optimal use of limited computational resources. We study this hypothesis by meta-learning reinforcement learning algorithms that sacrifice performance for a shorter description length (defined as the number of bits required to implement the given algorithm). The emerging class of models captures human exploration behavior better than previously considered approaches, such as Boltzmann exploration, upper confidence bound algorithms, and Thompson sampling. We additionally demonstrate that changing the description length in our class of models produces the intended effects: reducing description length captures the behavior of brain-lesioned patients while increasing it mirrors cognitive development during adolescence.

Research goal: Imitating human exploration by meta-learning
Empirical: yes
Algorithms: TD3
Seeds: -
Code: no
Env: no
Hyperparameters: in appendix

### Finite-Time Last-Iterate Convergence for Learning in Multi-Player Games
Link: https://openreview.net/forum?id=snUOkDdJypm
Keywords: learning in games, smooth monotone games, last-iterate convergence rate, Nash equilibrium
Abstract: We study the question of last-iterate convergence rate of the extragradient algorithm by Korpelevich [1976] and the optimistic gradient algorithm by Popov [1980] in multi-player games. We show that both algorithms with constant step-size have last-iterate convergence rate of to a Nash equilibrium in terms of the gap function in smooth monotone games, where each player's action set is an arbitrary convex set. Previous results only study the unconstrained setting, where each player's action set is the entire Euclidean space.  Our results address an open question raised in several recent work by Hsieh et al. [2019], Golowich et al. [2020a,b], who ask for last-iterate convergence rate of either the extragradient or the optimistic gradient algorithm in the constrained setting. Our convergence rates for both algorithms are tight and match the lower bounds by Golowich et al. [2020a,b]. At the core of our results lies a new notion -- the tangent residual, which we use to measure the proximity to equilibrium. We use the tangent residual (or a slight variation of the tangent residual) as the the potential function in our analysis of the extragradient algorithm (or the optimistic gradient algorithm) and prove that it is non-increasing between two consecutive iterates.

Research goal: convergence rates in multi-player games
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge
Link: https://openreview.net/forum?id=rc8o_j8I8PX
Keywords: Embodied Agents, Minecraft, Open-ended Learning, Multitask Learning, Internet Knowledge Base, Reinforcement Learning, Large Pre-training
Abstract: Autonomous agents have made great strides in specialist domains like Atari games and Go. However, they typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize across a wide spectrum of tasks and capabilities. Inspired by how humans continually learn and adapt in the open world, we advocate a trinity of ingredients for building generalist agents: 1) an environment that supports a multitude of tasks and goals, 2) a large-scale database of multimodal knowledge, and 3) a flexible and scalable agent architecture. We introduce MineDojo, a new framework built on the popular Minecraft game that features a simulation suite with thousands of diverse open-ended tasks and an internet-scale knowledge base with Minecraft videos, tutorials, wiki pages, and forum discussions. Using MineDojo's data, we propose a novel agent learning algorithm that leverages large pre-trained video-language models as a learned reward function. Our agent is able to solve a variety of open-ended tasks specified in free-form language without any manually designed dense shaping reward. We open-source the simulation suite, knowledge bases, algorithm implementation, and pretrained models (https://minedojo.org) to promote research towards the goal of generally capable embodied agents.

Research goal: benchmarking in open-ended worlds
Empirical: yes
Algorithms: PPO
Seeds: 3
Code: yes
Env: yes
Hyperparameters: in appendix

### Computationally Efficient Horizon-Free Reinforcement Learning for Linear Mixture MDPs
Link: https://openreview.net/forum?id=H4GmqyYMxFP
Keywords: -
Abstract: Recent studies have shown that episodic reinforcement learning (RL) is not more difficult than bandits, even with a long planning horizon and unknown state transitions. However, these results are limited to either tabular Markov decision processes (MDPs) or computationally inefficient algorithms for linear mixture MDPs. In this paper, we propose the first computationally efficient horizon-free algorithm for linear mixture MDPs, which achieves the optimal regret up to logarithmic factors. Our algorithm adapts a weighted least square estimator for the unknown transitional dynamic, where the weight is both \emph{variance-aware} and \emph{uncertainty-aware}. When applying our weighted least square estimato to heterogeneous linear bandits, we can obtain an regret in the first rounds, where is the dimension of the context and s the variance of the reward in the -th round. This also improves upon the best known algorithms in this setting when 's are known.

Research goal: solving linear mixture MDPs
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Learn to Match with No Regret: Reinforcement Learning in Markov Matching Markets
Link: https://openreview.net/forum?id=R3JMyR4MvoU
Keywords: matching, optimal matching, sequential matching, dynamic matching
Abstract: We study a Markov matching market involving a planner and a set of strategic agents on the two sides of the market. At each step, the agents are presented with a dynamical context, where the contexts determine the utilities. The planner controls the transition of the contexts to maximize the cumulative social welfare, while the agents aim to find a myopic stable matching at each step. Such a setting captures a range of applications including ridesharing platforms. We formalize the problem by proposing a reinforcement learning framework that integrates optimistic value iteration with maximum weight matching. The proposed algorithm addresses the coupled challenges of sequential exploration, matching stability, and function approximation. We prove that the algorithm achieves sublinear regret. 

Research goal: RL for matching markets
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Avalon: A Benchmark for RL Generalization Using Procedurally Generated Worlds
Link: https://openreview.net/forum?id=TzNuIdrHoU
Keywords: reinforcement learning, benchmark, generalization, simulator, embodied agents, virtual reality
Abstract: Despite impressive successes, deep reinforcement learning (RL) systems still fall short of human performance on generalization to new tasks and environments that differ from their training. As a benchmark tailored for studying RL generalization, we introduce Avalon, a set of tasks in which embodied agents in highly diverse procedural 3D worlds must survive by navigating terrain, hunting or gathering food, and avoiding hazards. Avalon is unique among existing RL benchmarks in that the reward function, world dynamics, and action space are the same for every task, with tasks differentiated solely by altering the environment; its 20 tasks, ranging in complexity from eat and throw to hunt and navigate, each create worlds in which the agent must perform specific skills in order to survive. This setup enables investigations of generalization within tasks, between tasks, and to compositional tasks that require combining skills learned from previous tasks. Avalon includes a highly efficient simulator, a library of baselines, and a benchmark with scoring metrics evaluated against hundreds of hours of human performance, all of which are open-source and publicly available. We find that standard RL baselines make progress on most tasks but are still far from human performance, suggesting Avalon is challenging enough to advance the quest for generalizable RL.

Research goal: benchmarking in diverse environments
Empirical: yes
Algorithms: PPO, Dreamer, IMPALA
Seeds: 5
Code: yes
Env: yes
Hyperparameters: in appendix

### A Policy-Guided Imitation Approach for Offline Reinforcement Learning
Link: https://openreview.net/forum?id=CKbqDtZnSc
Keywords: Offline RL
Abstract: Offline reinforcement learning (RL) methods can generally be categorized into two types: RL-based and Imitation-based. RL-based methods could in principle enjoy out-of-distribution generalization but suffer from erroneous off-policy evaluation. Imitation-based methods avoid off-policy evaluation but are too conservative to surpass the dataset. In this study, we propose an alternative approach, inheriting the training stability of imitation-style methods while still allowing logical out-of-distribution generalization. We decompose the conventional reward-maximizing policy in offline RL into a guide-policy and an execute-policy. During training, the guide-poicy and execute-policy are learned using only data from the dataset, in a supervised and decoupled manner. During evaluation, the guide-policy guides the execute-policy by telling where it should go so that the reward can be maximized, serving as the \textit{Prophet}. By doing so, our algorithm allows \textit{state-compositionality} from the dataset, rather than \textit{action-compositionality} conducted in prior imitation-style methods. We dumb this new approach Policy-guided Offline RL (\texttt{POR}). \texttt{POR} demonstrates the state-of-the-art performance on D4RL, a standard benchmark for offline RL. We also highlight the benefits of \texttt{POR} in terms of improving with supplementary suboptimal data and easily adapting to new tasks by only changing the guide-poicy.

Research goal: hybrid imitation learning and offline RL
Empirical: yes
Algorithms: BC, BCQ, CQL, BEAR, TD3
Seeds: 5
Code: yes
Env: yes
Hyperparameters: in appendix

### Markovian Interference in Experiments
Link: https://openreview.net/forum?id=CKbqDtZnSc
Keywords: Causal inference, Off-policy Evaluation, Experimentation, Interference, Reinforcement Learning
Abstract: We consider experiments in dynamical systems where interventions on some experimental units impact other units through a limiting constraint (such as a limited supply of products). Despite outsize practical importance, the best estimators for this `Markovian' interference problem are largely heuristic in nature, and their bias is not well understood. We formalize the problem of inference in such experiments as one of policy evaluation. Off-policy estimators, while unbiased, apparently incur a large penalty in variance relative to state-of-the-art heuristics. We introduce an on-policy estimator: the Differences-In-Q's (DQ) estimator. We show that the DQ estimator can in general have exponentially smaller variance than off-policy evaluation. At the same time, its bias is second order in the impact of the intervention. This yields a striking bias-variance tradeoff so that the DQ estimator effectively dominates state-of-the-art alternatives. From a theoretical perspective, we introduce three separate novel techniques that are of independent interest in the theory of Reinforcement Learning (RL). Our empirical evaluation includes a set of experiments on a city-scale ride-hailing simulator.  

Research goal: solving dynamic experimental decision making settings
Empirical: yes
Algorithms: TSRI, OPE
Seeds: 100
Code: in appendix
Env: no
Hyperparameters: in appendix

### Bellman Residual Orthogonalization for Offline Reinforcement Learning
Link: https://openreview.net/forum?id=x26Mpsf45P3
Keywords: offline RL, weight function, confidence intervals, policy optimization
Abstract: We propose and analyze a reinforcement learning principle that approximates the Bellman equations by enforcing their validity only along a user-defined space of test functions.  Focusing on applications to model-free offline RL with function approximation, we exploit this principle to derive confidence intervals for off-policy evaluation, as well as to optimize over policies within a prescribed policy class.  We prove an oracle inequality on our policy optimization procedure in terms of a trade-off between the value and uncertainty of an arbitrary comparator policy.  Different choices of test function spaces allow us to tackle different problems within a common framework.  We characterize the loss of efficiency in moving from on-policy to off-policy data using our procedures, and establish connections to concentrability coefficients studied in past work.  We examine in depth the implementation of our methods with linear function approximation, and provide theoretical guarantees with polynomial-time implementations even when Bellman closure does not hold.

Research goal: offline RL by validating policy using user-specified test functions
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Model-Based Offline Reinforcement Learning with Pessimism-Modulated Dynamics Belief
Link: https://openreview.net/forum?id=x26Mpsf45P3
Keywords: Offline reinforcement learning, model-based reinforcement learning, Bayesian learning
Abstract: Model-based offline reinforcement learning (RL) aims to find highly rewarding policy, by leveraging a previously collected static dataset and a dynamics model. While the dynamics model learned through reuse of the static dataset, its generalization ability hopefully promotes policy learning if properly utilized. To that end, several works propose to quantify the uncertainty of predicted dynamics, and explicitly apply it to penalize reward. However, as the dynamics and the reward are  intrinsically different factors in context of MDP, characterizing the impact of dynamics uncertainty through reward penalty may incur unexpected tradeoff between model utilization and risk avoidance. In this work, we instead maintain a belief distribution over dynamics, and evaluate/optimize policy through biased sampling from the belief. The sampling procedure, biased towards pessimism, is derived based on an alternating Markov game formulation of offline RL. We formally show that the biased sampling naturally induces an updated dynamics belief with policy-dependent reweighting factor, termed Pessimism-Modulated Dynamics Belief. To improve policy, we devise an iterative regularized policy optimization algorithm for the game, with guarantee of monotonous improvement under certain condition. To make practical, we further devise an offline RL algorithm to approximately find the solution. Empirical results show that the proposed approach achieves state-of-the-art performance on a wide range of benchmark tasks.

Research goal: offline RL with belief over environment dynamics
Empirical: yes
Algorithms: BC, BEAR, BRAC, CQL, MOReL, EDAC, PMDB
Seeds: 4
Code: yes
Env: yes
Hyperparameters: in appendix

### Oracle-Efficient Online Learning for Smoothed Adversaries
Link: https://openreview.net/forum?id=SbHxPRHPc2u
Keywords: Online learning, Computational Efficiency, Smoothed Analysis
Abstract: We study the design of computationally efficient online learning algorithms under smoothed analysis. In this setting, at every step, an adversary generates a sample from an adaptively chosen distribution whose density is upper bounded by times the uniform density. Given access to an offline optimization (ERM) oracle, we give the first computationally efficient online algorithms whose sublinear regret depends only on the pseudo/VC dimension of the class and the smoothness parameter . In particular, we achieve \emph{oracle-efficient} regret bounds of   for learning real-valued functions and for learning binary-valued functions. Our results establish that online learning is computationally as easy as offline learning, under the smoothed analysis framework. This contrasts the computational separation between online learning with worst-case adversaries and offline learning established by [HK16]. Our algorithms also achieve improved bounds for some settings with binary-valued functions and worst-case adversaries.  These include an oracle-efficient algorithm withregret that refines the earlier bound of [DS16] for finite domains, and an oracle-efficient algorithm with regret for the transductive setting.  

Research goal: better online learning with smooth adversaries
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Multi-Game Decision Transformers
Link: https://openreview.net/forum?id=0gouO5saq6K
Keywords: Reinforcement Learning, Generalist Agent, Multi-Environment RL, Upside-Down RL, Decision Transformers
Abstract: A longstanding goal of the field of AI is a method for learning a highly capable, generalist agent from diverse experience. In the subfields of vision and language, this was largely achieved by scaling up transformer-based models and training them on large, diverse datasets. Motivated by this progress, we investigate whether the same strategy can be used to produce generalist reinforcement learning agents. Specifically, we show that a single transformer-based model – with a single set of weights – trained purely offline can play a suite of up to 46 Atari games simultaneously at close-to-human performance. When trained and evaluated appropriately, we find that the same trends observed in language and vision hold, including scaling of performance with model size and rapid adaptation to new games via fine-tuning. We compare several approaches in this multi-game setting, such as online and offline RL methods and behavioral cloning, and find that our Multi-Game Decision Transformer models offer the best scalability and performance. We release the pre-trained models and code to encourage further research in this direction.

Research goal: train generalist agents
Empirical: yes
Algorithms: DT, BC, DQN, CQL, CPC, BERT, ACL
Seeds: 1
Code: yes
Env: yes
Hyperparameters: in appendix

### Giving Feedback on Interactive Student Programs with Meta-Exploration
Link: https://openreview.net/forum?id=_AsEqoBu3s
Keywords: meta-reinforcement learning, education, exploration
Abstract: Developing interactive software, such as websites or games, is a particularly engaging way to learn computer science. However, teaching and giving feedback on such software is time-consuming — standard approaches require instructors to manually grade student-implemented interactive programs. As a result, online platforms that serve millions, like Code.org, are unable to provide any feedback on assignments for implementing interactive programs, which critically hinders students’ ability to learn. One approach toward automatic grading is to learn an agent that interacts with a student’s program and explores states indicative of errors via reinforcement learning. However, existing work on this approach only provides binary feedback of whether a program is correct or not, while students require finer-grained feedback on the specific errors in their programs to understand their mistakes. In this work, we show that exploring to discover errors can be cast as a meta-exploration problem. This enables us to construct a principled objective for discovering errors and an algorithm for optimizing this objective, which provides fine-grained feedback. We evaluate our approach on a set of over 700K real anonymized student programs from a Code.org interactive assignment. Our approach provides feedback with 94.3% accuracy, improving over existing approaches by 17.7% and coming within 1.5% of human-level accuracy. Project web page: https://ezliu.github.io/dreamgrader.

Research goal: Rl for student feedback using meta-exploration
Empirical: yes
Algorithms: -
Seeds: 3
Code: yes
Env: -
Hyperparameters: in appendix

### Minimax-Optimal Multi-Agent RL in Markov Games With a Generative Model
Link: https://openreview.net/forum?id=W8nyVJruVg
Keywords: Markov games, sample complexity, Nash equilibrium, coarse correlated equilibrium, adversarial learning, Follow-the-Regularized-Leader
Abstract: Developing interactive software, such as websites or games, is a particularly engaging way to learn computer science. However, teaching and giving feedback on such software is time-consuming — standard approaches require instructors to manually grade student-implemented interactive programs. As a result, online platforms that serve millions, like Code.org, are unable to provide any feedback on assignments for implementing interactive programs, which critically hinders students’ ability to learn. One approach toward automatic grading is to learn an agent that interacts with a student’s program and explores states indicative of errors via reinforcement learning. However, existing work on this approach only provides binary feedback of whether a program is correct or not, while students require finer-grained feedback on the specific errors in their programs to understand their mistakes. In this work, we show that exploring to discover errors can be cast as a meta-exploration problem. This enables us to construct a principled objective for discovering errors and an algorithm for optimizing this objective, which provides fine-grained feedback. We evaluate our approach on a set of over 700K real anonymized student programs from a Code.org interactive assignment. Our approach provides feedback with 94.3% accuracy, improving over existing approaches by 17.7% and coming within 1.5% of human-level accuracy. Project web page: https://ezliu.github.io/dreamgrader.

Research goal: learning Nash equlibria in the presence of generative models
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -


### Leveraging Factored Action Spaces for Efficient Offline Reinforcement Learning in Healthcare
Link: https://openreview.net/forum?id=Jd70afzIvJ4
Keywords: reinforcement learning, offline rl, action space factorization, bias-variance trade-off, domain knowledge, healthcare, sepsis
Abstract: Many reinforcement learning (RL) applications have combinatorial action spaces, where each action is a composition of sub-actions. A standard RL approach ignores this inherent factorization structure, resulting in a potential failure to make meaningful inferences about rarely observed sub-action combinations; this is particularly problematic for offline settings, where data may be limited. In this work, we propose a form of linear Q-function decomposition induced by factored action spaces. We study the theoretical properties of our approach, identifying scenarios where it is guaranteed to lead to zero bias when used to approximate the Q-function. Outside the regimes with theoretical guarantees, we show that our approach can still be useful because it leads to better sample efficiency without necessarily sacrificing policy optimality, allowing us to achieve a better bias-variance trade-off. Across several offline RL problems using simulators and real-world datasets motivated by healthcare, we demonstrate that incorporating factored action spaces into value-based RL can result in better-performing policies. Our approach can help an agent make more accurate inferences within underexplored regions of the state-action space when applying RL to observational datasets. 

Research goal: RL for healthcare via factored action spaces
Empirical: yes
Algorithms: Q-iteration
Seeds: 10
Code: yes
Env: -
Hyperparameters: in appendix
