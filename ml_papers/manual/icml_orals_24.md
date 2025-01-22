# ICML 2024 Orals


### Genie: Generative Interactive Environments
Link: https://openreview.net/forum?id=bJbSbJskOS
Keywords: generative environments
Abstract: We introduce Genie, the first generative interactive environment trained in an unsupervised manner from unlabelled Internet videos. The model can be prompted to generate an endless variety of action-controllable virtual worlds described through text, synthetic images, photographs, and even sketches. At 11B parameters, Genie can be considered a foundation world model. It is comprised of a spatiotemporal video tokenizer, an autoregressive dynamics model, and a simple and scalable latent action model. Genie enables users to act in the generated environments on a frame-by-frame basis despite training without any ground-truth action labels or other domain specific requirements typically found in the world model literature. Further the resulting learned latent action space facilitates training agents to imitate behaviors from unseen videos, opening the path for training generalist agents of the future.

Research goal: generative environments
Empirical: yes
Algorithms: BC
Seeds: 5
Code: no
Env: not full version
Hyperparameters: in appendix

### Environment Design for Inverse Reinforcement Learning
Link: https://openreview.net/forum?id=Ar0dsOMStE
Keywords: environment design, inverse rl
Abstract: Learning a reward function from demonstrations suffers from low sample-efficiency. Even with abundant data, current inverse reinforcement learning methods that focus on learning from a single environment can fail to handle slight changes in the environment dynamics. We tackle these challenges through adaptive environment design. In our framework, the learner repeatedly interacts with the expert, with the former selecting environments to identify the reward function as quickly as possible from the expert’s demonstrations in said environments. This results in improvements in both sample-efficiency and robustness, as we show experimentally, for both exact and approximate inference.

Research goal: adapting the environment for more stable inverse RL
Empirical: yes
Algorithms: BC, PPO
Seeds: 5
Code: yes
Env: no
Hyperparameters: no

### Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error
Link: https://openreview.net/forum?id=pgI9inG2Ny
Keywords: robust rl, inverse rl
Abstract: Establishing robust policies is essential to counter attacks or disturbances affecting deep reinforcement learning (DRL) agents. Recent studies explore state-adversarial robustness and suggest the potential lack of an optimal robust policy (ORP), posing challenges in setting strict robustness constraints. This work further investigates ORP: At first, we introduce a consistency assumption of policy (CAP) stating that optimal actions in the Markov decision process remain consistent with minor perturbations, supported by empirical and theoretical evidence. Building upon CAP, we crucially prove the existence of a deterministic and stationary ORP that aligns with the Bellman optimal policy. Furthermore, we illustrate the necessity of 
-norm when minimizing Bellman error to attain ORP. This finding clarifies the vulnerability of prior DRL algorithms that target the Bellman optimal policy with 
-norm and motivates us to train a Consistent Adversarial Robust Deep Q-Network (CAR-DQN) by minimizing a surrogate of Bellman Infinity-error. The top-tier performance of CAR-DQN across various benchmarks validates its practical effectiveness and reinforces the soundness of our theoretical analysis.

Research goal: adapting the environment for more stable inverse RL
Empirical: yes
Algorithms: DQN
Seeds: 5
Code: yes
Env: no
Hyperparameters: yes

### Self-Composing Policies for Scalable Continual Reinforcement Learning
Link: https://openreview.net/forum?id=f5gtX2VWSB
Keywords: continual rl, architectures
Abstract: This work introduces a growable and modular neural network architecture that naturally avoids catastrophic forgetting and interference in continual reinforcement learning. The structure of each module allows the selective combination of previous policies along with its internal policy accelerating the learning process on the current task. Unlike previous growing neural network approaches, we show that the number of parameters of the proposed approach grows linearly with respect to the number of tasks, and does not sacrifice plasticity to scale. Experiments conducted in benchmark continuous control and visual problems reveal that the proposed approach achieves greater knowledge transfer and performance than alternative methods.

Research goal: modular policies for continual RL
Empirical: yes
Algorithms: SAC, PPO
Seeds: 10
Code: yes
Env: yes
Hyperparameters: in appendix

### Rate-Optimal Policy Optimization for Linear Markov Decision Processes
Link: https://openreview.net/forum?id=VJwsDwuiuH
Keywords: linear mdps, policy optimization
Abstract: We study regret minimization in online episodic linear Markov Decision Processes, and propose a policy optimization algorithm that is computationally efficient, and obtains rate optimal regret where denotes the number of episodes. Our work is the first to establish the optimal rate (in terms of) of convergence in the stochastic setting with bandit feedback using a policy optimization based approach, and the first to establish the optimal rate in the adversarial setup with full information feedback, for which no algorithm with an optimal rate guarantee was previously known.

Research goal: solving linear MDPs
Empirical: no
Algorithms: -
Seeds: -
Code: -
Env: -
Hyperparameters: -

### Stop Regressing: Training Value Functions via Classification for Scalable Deep RL
Link: https://openreview.net/forum?id=dVpFKfqF3R
Keywords: reinforcement learning, q-learning, deep rl
Abstract: Value functions are an essential component in deep reinforcement learning (RL), that are typically trained via mean squared error regression to match bootstrapped target values. However, scaling value-based RL methods to large networks has proven challenging. This difficulty is in stark contrast to supervised learning: by leveraging a cross-entropy classification loss, supervised methods have scaled reliably to massive networks. Observing this discrepancy, in this paper, we investigate whether the scalability of deep RL can also be improved simply by using classification in place of regression for training value functions. We show that training value functions with categorical cross-entropy significantly enhances performance and scalability across various domains, including single-task RL on Atari 2600 games, multi-task RL on Atari with large-scale ResNets, robotic manipulation with Q-transformers, playing Chess without search, and a language-agent Wordle task with high-capacity Transformers, achieving state-of-the-art results on these domains. Through careful analysis, we show that categorical cross-entropy mitigates issues inherent to value-based RL, such as noisy targets and non-stationarity. We argue that shifting to categorical cross-entropy for training value functions can substantially improve the scalability of deep RL at little-to-no cost.

Research goal: improving q-learning
Empirical: yes
Algorithms: C51, DQN
Seeds: 5
Code: no
Env: no
Hyperparameters: in appendix

### Offline Actor-Critic Reinforcement Learning Scales to Large Models
Link: https://openreview.net/forum?id=tl2qmO5kpD
Keywords: offline rl, architectures
Abstract: We show that offline actor-critic reinforcement learning can scale to large models - such as transformers - and follows similar scaling laws as supervised learning. We find that offline actor-critic algorithms can outperform strong, supervised, behavioral cloning baselines for multi-task training on a large dataset; containing both sub-optimal and expert behavior on 132 continuous control tasks. We introduce a Perceiver-based actor-critic model and elucidate the key features needed to make offline RL work with self- and cross-attention modules. Overall, we find that: i) simple offline actor critic algorithms are a natural choice for gradually moving away from the currently predominant paradigm of behavioral cloning, and ii) via offline RL it is possible to learn multi-task policies that master many domains simultaneously, including real robotics tasks, from sub-optimal demonstrations or self-generated data.

Research goal: using larger models for offline RL
Empirical: no
Algorithms: BC
Seeds: -
Code: no
Env: no
Hyperparameters: in appendix

### ACE: Off-Policy Actor-Critic with Causality-Aware Entropy Regularization
Link: https://openreview.net/forum?id=tl2qmO5kpD
Keywords: reinforcement learning, exploration
Abstract: The varying significance of distinct primitive behaviors during the policy learning process has been overlooked by prior model-free RL algorithms. Leveraging this insight, we explore the causal relationship between different action dimensions and rewards to evaluate the significance of various primitive behaviors during training. We introduce a causality-aware entropy term that effectively identifies and prioritizes actions with high potential impacts for efficient exploration. Furthermore, to prevent excessive focus on specific primitive behaviors, we analyze the gradient dormancy phenomenon and introduce a dormancy-guided reset mechanism to further enhance the efficacy of our method. Our proposed algorithm, ACE: Off-policy Actor-critic with Causality-aware Entropy regularization, demonstrates a substantial performance advantage across 29 diverse continuous control tasks spanning 7 domains compared to model-free RL baselines, which underscores the effectiveness, versatility, and efficient sample efficiency of our approach. Benchmark results and videos are available at https://ace-rl.github.io/.

Research goal: better regularization in SAC (less need for hyperparameter tuning)
Empirical: yes
Algorithms: SAC, TD3
Seeds: 6
Code: yes
Env: no
Hyperparameters: partial in appendix

### OMPO: A Unified Framework for RL under Policy and Dynamics Shifts
Link: https://openreview.net/forum?id=R83VIZtHXA
Keywords: reinforcement learning, dynamics shifts, policy shifts
Abstract: Training reinforcement learning policies using environment interaction data collected from varying policies or dynamics presents a fundamental challenge. Existing works often overlook the distribution discrepancies induced by policy or dynamics shifts, or rely on specialized algorithms with task priors, thus often resulting in suboptimal policy performances and high learning variances. In this paper, we identify a unified strategy for online RL policy learning under diverse settings of policy and dynamics shifts: transition occupancy matching. In light of this, we introduce a surrogate policy learning objective by considering the transition occupancy discrepancies and then cast it into a tractable min-max optimization problem through dual reformulation. Our method, dubbed Occupancy-Matching Policy Optimization (OMPO), features a specialized actor-critic structure equipped with a distribution discriminator and a small-size local buffer. We conduct extensive experiments based on the OpenAI Gym, Meta-World, and Panda Robots environments, encompassing policy shifts under stationary and non-stationary dynamics, as well as domain adaption. The results demonstrate that OMPO outperforms the specialized baselines from different categories in all settings. We also find that OMPO exhibits particularly strong performance when combined with domain randomization, highlighting its potential in RL-based robotics applications.

Research goal: solving RL under dynamic shifts
Empirical: yes
Algorithms: SAC, TD3
Seeds: 10
Code: yes
Env: yes
Hyperparameters: in appendix

### Pausing Policy Learning in Non-stationary Reinforcement Learning
Link: https://openreview.net/forum?id=qY622O6Ehg
Keywords: reinforcement learning
Abstract: Real-time inference is a challenge of real-world reinforcement learning due to temporal differences in time-varying environments: the system collects data from the past, updates the decision model in the present, and deploys it in the future. We tackle a common belief that continually updating the decision is optimal to minimize the temporal gap. We propose forecasting an online reinforcement learning framework and show that strategically pausing decision updates yields better overall performance by effectively managing aleatoric uncertainty. Theoretically, we compute an optimal ratio between policy update and hold duration, and show that a non-zero policy hold duration provides a sharper upper bound on the dynamic regret. Our experimental evaluations on three different environments also reveal that a non-zero policy hold duration yields higher rewards compared to continuous decision updates.

Research goal: strategic non-updating in RL for better results
Empirical: yes
Algorithms: SAC, Q-learning
Seeds: -
Code: no
Env: no
Hyperparameters: partial in appendix

### Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study
Link: https://openreview.net/forum?id=6XH8R7YrSk
Keywords: rlhf, rl for llms
Abstract: Reinforcement Learning from Human Feedback (RLHF) is currently the most widely used method to align large language models (LLMs) with human preferences. Existing RLHF methods can be roughly categorized as either reward-based or reward-free. Novel applications such as ChatGPT and Claude leverage reward-based methods that first learn a reward model and apply actor-critic algorithms, such as Proximal Policy Optimization (PPO). However, in academic benchmarks, state-of-the-art results are often achieved via reward-free methods, such as Direct Preference Optimization (DPO). Is DPO truly superior to PPO? Why does PPO perform poorly on these benchmarks? In this paper, we first conduct both theoretical and empirical studies on the algorithmic properties of DPO and show that DPO may have fundamental limitations. Moreover, we also comprehensively examine PPO and reveal the key factors for the best performances of PPO in fine-tuning LLMs. Finally, we benchmark DPO and PPO across a collection of RLHF testbeds, ranging from dialogue to code generation. Experiment results demonstrate that PPO is able to surpass other alignment methods in all cases and achieve state-of-the-art results in challenging code competitions.

Research goal: fine-tuning LLMs
Empirical: yes
Algorithms: PPO
Seeds: -
Code: no
Env: yes
Hyperparameters: in appendix