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