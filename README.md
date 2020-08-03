# Tensorflow 2 PPO Implementation for MuJoCo Gym Environments

This is a simple implementation of the PPO Algorithm based on its accompanying [paper](https://arxiv.org/pdf/1707.06347.pdf) for use in MuJoCo gym environments.

This repository was mainly made for learning purposes.

# Requirements
- [MuJoCo](http://www.mujoco.org)
- [Open AI Baselines](https://github.com/openai/baselines)
- [Open AI Gym](https://github.com/openai/gym)
- Tensorflow 2 (Tested on v2.2.0)
- Matplotlib (Tested on v3.2.2)

# Usage



# Results

Note: Results are *very* seed-dependent. It is also possible that better hyperparameters exist for certain environments.

![image]

# Credits

This code was primarily based off this [PyTorch Implementation by ikostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) in order to achieve similar results. It utilizes the same environment wrappers, has a similar model structure, and logs results in a similar fashion.

This code also uses the DiagGaussian distribution and a couple code fragments from the [PPO2 Implementation in OpenAI Baselines](https://github.com/openai/baselines).