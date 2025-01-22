# NeurIPS 2021 Orals

### Bellman-consistent Pessimism for Offline Reinforcement Learning
Link: https://openreview.net/forum?id=e8WWUBeafM
Keywords: offline reinforcement learning, Bellman-consistent pessimism, sample complexity bounds, linear MDP, function approximation
Abstract: The use of pessimism, when reasoning about datasets lacking exhaustive exploration has recently gained prominence in offline reinforcement learning. Despite the robustness it adds to the algorithm, overly pessimistic reasoning can be equally damaging in precluding the discovery of good policies, which is an issue for the popular bonus-based pessimism. In this paper, we introduce the notion of Bellman-consistent pessimism for general function approximation: instead of calculating a point-wise lower bound for the value function, we implement pessimism at the initial state over the set of functions consistent with the Bellman equations. Our theoretical guarantees only require Bellman closedness as standard in the exploratory setting, in which case bonus-based pessimism fails to provide guarantees.  Even in the special case of linear function approximation where stronger expressivity assumptions hold, our result improves upon a recent bonus-based approach by 
 in its sample complexity (when the action space is finite). Remarkably, our algorithms automatically adapt to the best bias-variance tradeoff in the hindsight, whereas most prior approaches require tuning extra hyperparameters a priori.

Research goal: Optimal bias-variance tradeoff in offline RL as opposed to manual tuning
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Deep Reinforcement Learning at the Edge of the Statistical Precipice
Link: https://openreview.net/forum?id=uqv8-U4lKBe
Keywords: Reinforcement Learning, Evaluation, Benchmarking, Scientific Progress, Reliability
Abstract: Deep reinforcement learning (RL) algorithms are predominantly evaluated by comparing their relative performance on a large suite of tasks. Most published results on deep RL benchmarks compare point estimates of aggregate performance such as mean and median scores across tasks, ignoring the statistical uncertainty implied by the use of a finite number of training runs. Beginning with the Arcade Learning Environment (ALE), the shift towards computationally-demanding benchmarks has led to the practice of evaluating only a small number of runs per task, exacerbating the statistical uncertainty in point estimates. In this paper, we argue that reliable evaluation in the few run deep RL regime cannot ignore the uncertainty in results without running the risk of slowing down progress in the field. We illustrate this point using a case study on the Atari 100k benchmark, where we find substantial discrepancies between conclusions drawn from point estimates alone versus a more thorough statistical analysis. With the aim of increasing the field's confidence in reported results with a handful of runs, we advocate for reporting interval estimates of aggregate performance and propose performance profiles to account for the variability in results, as well as present more robust and efficient aggregate metrics, such as interquartile mean scores, to achieve small uncertainty in results. Using such statistical tools, we scrutinize performance evaluations of existing algorithms on other widely used RL benchmarks including the ALE, Procgen, and the DeepMind Control Suite, again revealing discrepancies in prior comparisons. Our findings call for a change in how we evaluate performance in deep RL, for which we present a more rigorous evaluation methodology, accompanied with an open-source library rliable, to prevent unreliable results from stagnating the field. This work received an outstanding paper award at NeurIPS 2021.

Research goal: More reliable evaluation in RL research
Empirical: yes
Algorithms: DQN
Seeds: 100
Code: yes
Env: not explicitly, though unsure if Atari 100k is versioned
Hyperparameters: yes


### Replacing Rewards with Examples: Example-Based Policy Search via Recursive Classification
Link: https://openreview.net/forum?id=UVQNdLIELSU
Keywords: reinforcement learning, example-based control
Abstract: Reinforcement learning (RL) algorithms assume that users specify tasks by manually writing down a reward function. However, this process can be laborious and demands considerable technical expertise. Can we devise RL algorithms that instead enable users to specify tasks simply by providing examples of successful outcomes? In this paper, we derive a control algorithm that maximizes the future probability of these successful outcome examples. Prior work has approached similar problems with a two-stage process, first learning a reward function and then optimizing this reward function using another reinforcement learning algorithm. In contrast, our method directly learns a value function from transitions and successful outcomes, without learning this intermediate reward function. Our method therefore requires fewer hyperparameters to tune and lines of code to debug. We show that our method satisfies a new data-driven Bellman equation, where examples take the place of the typical reward function term. Experiments show that our approach outperforms prior methods that learn explicit reward functions.

Research goal: RL with examples instead of a reward function
Empirical: yes
Algorithms: SAC, RCE, SQIL, VICE, DAC, ORIL, PURL
Seeds: 5
Code: yes
Env: dated version on envs in appendix
Hyperparameters: partial in appendix

### An Exponential Lower Bound for Linearly Realizable MDP with Constant Suboptimality Gap
Link: https://openreview.net/forum?id=WnJXcebN7hX
Keywords: Reinforcement Learning, Linear Function Approximation, Lower Bound
Abstract: A fundamental question in the theory of reinforcement learning is: suppose the optimal function lies in the linear span of a given dimensional feature mapping, is sample-efficient reinforcement learning (RL) possible? The recent and remarkable result of Weisz et al. (2020) resolves this question in the negative, providing an exponential (in d) sample size lower bound, which holds even if the agent has access to a generative model of the environment. One may hope that such a lower can be circumvented with an even stronger assumption that there is a \emph{constant gap} between the optimal value of the best action and that of the second-best action (for all states); indeed, the construction in Weisz et al. (2020) relies on having an exponentially small gap. This work resolves this subsequent question, showing that an exponential sample complexity lower bound still holds even if a constant gap is assumed.  Perhaps surprisingly, this result implies an exponential separation between the online RL setting and the generative model setting, where sample-efficient RL is in fact possible in the latter setting with a constant gap. Complementing our negative hardness result, we give two positive results showing that provably sample-efficient RL is possible either under an additional low-variance assumption or under a novel hypercontractivity assumption.

Research goal: Theoretical analysis of sample efficiency
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Interesting Object, Curious Agent: Learning Task-Agnostic Exploration
Link: https://openreview.net/forum?id=knKJgksd7kA
Keywords: reinforcement learning, exploration, intrinsic motivation, continual learning
Abstract: Common approaches for task-agnostic exploration learn tabula-rasa --the agent assumes isolated environments and no prior knowledge or experience. However, in the real world, agents learn in many environments and always come with prior experiences as they explore new ones. Exploration is a lifelong process. In this paper, we propose a paradigm change in the formulation and evaluation of task-agnostic exploration. In this setup, the agent first learns to explore across many environments without any extrinsic goal in a task-agnostic manner. Later on, the agent effectively transfers the learned exploration policy to better explore new environments when solving tasks. In this context, we evaluate several baseline exploration strategies and present a simple yet effective approach to learning task-agnostic exploration policies. Our key idea is that there are two components of exploration: (1) an agent-centric component encouraging exploration of unseen parts of the environment based on an agent’s belief; (2) an environment-centric component encouraging exploration of inherently interesting objects. We show that our formulation is effective and provides the most consistent exploration across several training-testing environment pairs. We also introduce benchmarks and metrics for evaluating task-agnostic exploration strategies. The source code is available at https://github.com/sparisi/cbet/.

Research goal: Meta-learning exploration
Empirical: yes
Algorithms: IMPALA
Seeds: 7
Code: yes
Env: no
Hyperparameters: in appendix

### Sequential Causal Imitation Learning with Unobserved Confounders
Link: https://openreview.net/forum?id=Kvb0482Ysaf
Keywords: causality, reinforcement learning, imitation
Abstract: "Monkey see monkey do" is an age-old adage, referring to naive imitation without a deep understanding of a system's underlying mechanics. Indeed, if a demonstrator has access to information unavailable to the imitator (monkey), such as a different set of sensors, then no matter how perfectly the imitator models its perceived environment (See), attempting to directly reproduce the demonstrator's behavior (Do) can lead to poor outcomes. Imitation learning in the presence of a mismatch between demonstrator and imitator has been studied in the literature under the rubric of causal imitation learning  (Zhang et. al. 2020), but existing solutions are limited to single-stage decision-making. This paper investigates the problem of causal imitation learning in sequential settings, where the imitator must make multiple decisions per episode. We develop a graphical criterion that is both necessary and sufficient for determining the feasibility of causal imitation, providing conditions when an imitator can match a demonstrator's performance despite differing capabilities. Finally, we provide an efficient algorithm for determining imitability, and corroborate our theory with simulations.

Research goal: Imitation learning in sequential settings with mismatch
Empirical: toy
Algorithms: -
Seeds: -
Code: no
Env: -
Hyperparameters: no

### On the Expressivity of Markov Reward 
Link: https://openreview.net/forum?id=9DlCh34E1bN
Keywords: Reinforcement Learning, Reward Functions, Reward, Reward Hypothesis, Markov Decision Process
Abstract: Reward is the driving force for reinforcement-learning agents. This paper is dedicated to understanding the expressivity of reward as a way to capture tasks that we would want an agent to perform. We frame this study around three new abstract notions of “task” that might be desirable: (1) a set of acceptable behaviors, (2) a partial ordering over behaviors, or (3) a partial ordering over trajectories. Our main results prove that while reward can express many of these tasks, there exist instances of each task type that no Markov reward function can capture. We then provide a set of polynomial-time algorithms that construct a Markov reward function that allows an agent to optimize tasks of each of these three types, and correctly determine when no such reward function exists. We conclude with an empirical study that corroborates and illustrates our theoretical findings.

Research goal: Studying what can and can't be expressed with Markov reward
Empirical: yes
Algorithms: -
Seeds: -
Code: no
Env: yes
Hyperparameters: no

### The best of both worlds: stochastic and adversarial episodic MDPs with unknown transition
Link: https://openreview.net/forum?id=-zALR_-372y
Keywords: reinforcement learning, online learning, best of both worlds
Abstract: We consider the best-of-both-worlds problem for learning an episodic Markov Decision Process through episodes, with the goal of achieving regret when the losses are adversarial and simultaneously regret when the losses are (almost) stochastic. Recent work by [Jin and Luo, 2020]  achieves this goal when the fixed transition is known, and leaves the case of unknown transition as a major open question. In this work, we resolve this open problem by using the same Follow-the-Regularized-Leader (FTRL) framework together with a set of new techniques. Specifically, we first propose a loss-shifting trick in the FTRL analysis, which greatly simplifies the approach of [Jin and Luo, 2020] and already improves their results for the known transition case. Then, we extend this idea to the unknown transition case and develop a novel analysis which upper bounds the transition estimation error by the regret itself in the stochastic setting, a key property to ensure regret.

Research goal: Solving MDPs with unknown transitions
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Online Variational Filtering and Parameter Learning
Link: https://openreview.net/forum?id=et2st4Jqhc
Keywords: Approximate Inference, Variational inference, State-space models, Filtering, Time series, Online learning
Abstract: We present a variational method for online state estimation and parameter learning in state-space models (SSMs), a ubiquitous class of latent variable models for sequential data. As per standard batch variational techniques, we use stochastic gradients to simultaneously optimize a lower bound on the log evidence with respect to both model parameters and a variational approximation of the states' posterior distribution. However, unlike existing approaches, our method is able to operate in an entirely online manner, such that historic observations do not require revisitation after being incorporated and the cost of updates at each time step remains constant, despite the growing dimensionality of the joint posterior distribution of the states. This is achieved by utilizing backward decompositions of this joint posterior distribution and of its variational approximation, combined with Bellman-type recursions for the evidence lower bound and its gradients. We demonstrate the performance of this methodology across several examples, including high-dimensional SSMs and sequential Variational Auto-Encoders.

Research goal: Online variational inference for state estimation
Empirical: SSM only
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Risk Monotonicity in Statistical Learning
Link: https://openreview.net/forum?id=z5-chidgZU3
Keywords: Statistical Learning, Risk Monotonicity, Concentration Inequalities, PAC-Bayesian Bounds
Abstract: Acquisition of data is a difficult task in many applications of machine learning, and it is only natural that one hopes and expects the population risk to decrease (better performance) monotonically with increasing data points. It turns out, somewhat surprisingly, that this is not the case even for the most standard algorithms that minimize the empirical risk. Non-monotonic behavior of the risk and instability in training have manifested and appeared in the popular deep learning paradigm under the description of double descent. These problems highlight the current lack of understanding of learning algorithms and generalization. It is, therefore, crucial to pursue this concern and provide a characterization of such behavior. In this paper, we derive the first consistent and risk-monotonic (in high probability) algorithms for a general statistical learning setting under weak assumptions, consequently answering some questions posed by Viering et. al. 2019 on how to avoid non-monotonic behavior of risk curves. We further show that risk monotonicity need not necessarily come at the price of worse excess risk rates. To achieve this, we derive new empirical Bernstein-like concentration inequalities of independent interest that hold for certain non-i.i.d.~processes such as Martingale Difference Sequences.

Research goal: Relationshipo between data collection and risk
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Efficient First-Order Contextual Bandits: Prediction, Allocation, and Triangular Discrimination 
Link: https://openreview.net/forum?id=3qYgdGj9Svt
Keywords: contextual bandits, reinforcement learning, statistical learning, learning theory, fast rates, adaptivity, sequential probability assignment, conditional density estimation, logarithmic loss
Abstract: A recurring theme in statistical learning, online learning, and beyond is that faster convergence rates are possible for problems with low noise, often quantified by the performance of the best hypothesis; such results are known as first-order or small-loss guarantees. While first-order guarantees are relatively well understood in statistical and online learning, adapting to low noise in contextual bandits (and more broadly, decision making) presents major algorithmic challenges. In a COLT 2017 open problem, Agarwal, Krishnamurthy, Langford, Luo, and Schapire asked whether first-order guarantees are even possible for contextual bandits and---if so---whether they can be attained by efficient algorithms. We give a resolution to this question by providing an optimal and efficient reduction from contextual bandits to online regression with the logarithmic (or, cross-entropy) loss. Our algorithm is simple and practical, readily accommodates rich function classes, and requires no distributional assumptions beyond realizability. In a large-scale empirical evaluation, we find that our approach typically outperforms  comparable non-first-order methods.

Research goal: First-order guarantees for contextual bandits
Empirical: toy
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Learning Treatment Effects in Panels with General Intervention Patterns
Link: https://openreview.net/forum?id=3qYgdGj9Svt
Keywords: Causal Inference, Treatment Effect, Low-Rank Matrix Estimation, Panel Data, Synthetic Control
Abstract: The problem of causal inference with panel data is a central econometric question. The following is a fundamental version of this problem: Let be a low rank matrix and be a zero-mean noise matrix. For a `treatment' matrix with entries in we observe the matrix with entries where are unknown, heterogenous treatment effects. The problem requires we estimate the average treatment effect. The synthetic control paradigm provides an approach to estimating when places support on a single row. This paper extends that framework to allow rate-optimal recovery of for general, thus broadly expanding its applicability. Our guarantees are the first of their type in this general setting. Computational experiments on synthetic and real-world data show a substantial advantage over competing estimators. 

Research goal: Learning treatment effects
Empirical: toy
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -