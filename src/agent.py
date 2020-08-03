import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Ignore general tf warnings
from collections import deque
import pickle
import matplotlib.pyplot as plt
import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from baselines import bench
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import plot_util as pu

from src.env_wrappers import VecNormalize, TimeLimitMask
from src.model import Model_Actor_Critic

class PPO_Agent():
    def __init__(self, params, env_name, model_dir, log_dir, plot_dir, seed=0):
        self.params = params

        self.env_name = env_name

        self.seed = seed

        self.model_dir = model_dir
        self.plot_dir = plot_dir
        self.log_dir = log_dir
    
    def construct_agent(self):
        self.env = gym.make(self.env_name) # Note: Do not unwrap, only need general attributes of env

        # Set seed for deterministic results
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.env.seed(self.seed)

        # TimeLimit wrapper useful for Mujoco environments
        self.env = TimeLimitMask(self.env)

        # Models - training step function, networks, and optimizer must be redefined/reconstructed after switching environments
        self.actor_critic = Model_Actor_Critic(self.env.observation_space.shape[0], 
                                               self.env.action_space.shape[0], 
                                               self.params['ACTOR_HIDDEN_UNITS'],
                                               self.params['CRITIC_HIDDEN_UNITS'])
        self.train_model = self.get_train_model_function()
        self.optimizer = Adam(learning_rate=self.params['LEARNING_RATE'], epsilon=self.params['OPTIMIZER_EPSILON'])

    def set_env(self, env_name, **kwargs): # 'seed' and 'params' are optional arguments
        self.env_name = env_name
        self.params = kwargs.get('params', self.params)
        self.seed = kwargs.get('seed', self.seed)

    def calculate_returns(self, rewards, masks, bad_masks, values):
        # Use generalized advantage estimator (balance of bias & variance with lambda-return compared to TD and MC)
        # If lambda = 1, then becomes MC. If lambda is 0, then becomes TD
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.params['GAMMA'] * values[i + 1] * masks[i] - values[i]
            gae = delta + self.params['GAMMA'] * self.params['LAMBDA'] * masks[i] * gae
            gae = gae * bad_masks[i] # If not a true transition due to environment reset, set gae to 0
            returns.insert(0, gae) # Keep inserting at beginning because we're traversing in reverse order
        returns += values[:-1]                                                                         
        return returns

    # Returns a new function whenever environments are switched because tf.function creates a graph specific to the function
    def get_train_model_function(self):
        # @tf.function decoration runs function in graph mode - signficantly faster than eager execution in tf 2.0
        @tf.function
        def train_model(states, actions, returns, old_values, old_logps):
            advantages = returns - old_values
            mean, var = tf.nn.moments(advantages, [0], keepdims=True) # Standardize advantages in each MINIBATCH
            advantages = (advantages - mean) / (tf.sqrt(var) + 1e-8)

            with tf.GradientTape() as tape:
                values, logps, dist_entropy = self.actor_critic.evaluate_actions(states, actions)

                ratio = tf.exp(logps - old_logps)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.params['CLIP_PARAM'], 1 + self.params['CLIP_PARAM'])

                actor_loss = tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                critic_loss = tf.reduce_mean(0.5 * tf.square(returns - values))

                loss = -actor_loss + (self.params['VALUE_FUNCTION_COEF'] * critic_loss) - (self.params['ENTROPY_COEF'] * dist_entropy)

            gradients = tape.gradient(loss, self.actor_critic.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_weights))
        return train_model

    def train(self):
        # Construct agent
        self.construct_agent()

        # Create log directory
        os.makedirs(os.path.join(self.log_dir, self.env_name.split('-')[0]), exist_ok=True)

        # Construct testing environment - bench.Monitor used for logging episodes, VecNormalize for normalizing states & rewards
        self.env = bench.Monitor(self.env, os.path.join(self.log_dir, self.env_name.split('-')[0], '0')) # FILE PATH MUST HAVE A DIGIT AT END
        self.env = DummyVecEnv([lambda: self.env]) # Needed to use the VecNormalize wrapper
        self.env = VecNormalize(self.env, 
                                ob=True, ret=True, 
                                gamma=self.params['GAMMA']) # Normalize states AND rewards for training

        episode_reward_summary = deque(maxlen=10)

        state = self.env.reset() # Reset once at beginning, all subsequent resets handled by monitor

        NUM_TRAIN_UPDATES = int(self.params['NUM_ENV_TIMESTEPS']) // self.params['NUM_TIMESTEPS_PER_UPDATE']
        MINIBATCH_SIZE = int(self.params['NUM_TIMESTEPS_PER_UPDATE']) // self.params['NUM_MINIBATCHES']

        for update in range(1, NUM_TRAIN_UPDATES + 1):
            start_time = time.time()

            total_update_reward = 0

            states = []
            actions = []
            rewards = []
            masks = []
            bad_masks = []
            values = []
            old_logps = []
            
            """Perform Rollout"""
            for update_step in range(1, self.params['NUM_TIMESTEPS_PER_UPDATE'] + 1):
                value, action, action_logp = self.actor_critic.act(state)
                
                next_state, reward, done, info = self.env.step(action)

                if 'episode' in info.keys():
                    episode_reward_summary.append(info['episode']['r']) # Logged by bench.Monitor
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_logps.append(action_logp)
                values.append(value)
                masks.append((0.0 if done else 1.0))
                bad_masks.append((0.0 if 'bad_transition' in info.keys() else 1.0)) # Occurs when a done is the result
                                                                                    # of exceeding the environment step limit
                state = next_state
            
            values.append(self.actor_critic.get_value(next_state))

            """Update Networks"""
            states = np.asarray(states, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32)
            masks = np.asarray(masks, dtype=np.float32)
            bad_masks = np.asarray(bad_masks, dtype=np.float32)
            values = np.asarray(values, dtype=np.float32)
            old_logps = np.asarray(old_logps, dtype=np.float32)

            # Caclulate returns
            returns = self.calculate_returns(rewards, masks, bad_masks, values)

            # Create minibatches and train model
            inds = np.arange(self.params['NUM_TIMESTEPS_PER_UPDATE'])
            for _ in range(self.params['NUM_EPOCHS']):
                np.random.shuffle(inds) # IMPORTANT: shuffling data indices for training stability. Since this problem is a sequential task,
                                        # consecutive samples are very similar to each other in unshuffled data. Our minibatch sizes are already
                                        # fairly small, so not shuffling could lead us to have conflicting gradients for each minibatch.
                for start in range(0, self.params['NUM_TIMESTEPS_PER_UPDATE'], MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_inds = inds[start:end]
                    mb = (arr[mb_inds] for arr in (states, actions, returns, values, old_logps))

                    self.train_model(*mb)
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            """ Print Summary """
            print("Update {}/{}, Timesteps completed {}\nLast {} episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, elapsed update time {:.3f} seconds\n"
                    .format(update, NUM_TRAIN_UPDATES, self.params['NUM_TIMESTEPS_PER_UPDATE'] * update, len(episode_reward_summary),
                    np.mean(episode_reward_summary), np.median(episode_reward_summary), np.min(episode_reward_summary),
                    np.max(episode_reward_summary), elapsed_time))

            if (update % self.params['SAVE_INTERVAL'] == 0 or update == NUM_TRAIN_UPDATES):
                # Save model weights
                model_path = os.path.join(self.model_dir, self.env_name, 'model_weights')
                self.actor_critic.save_weights(model_path)

                # Save ob_rms from enviornment via pickle so that it can be restored when testing 
                vecnorm_path = os.path.join(self.model_dir, self.env_name, 'vecnorm_stats.pickle')
                with open(vecnorm_path, 'wb') as handle:
                    pickle.dump(getattr(self.env, 'ob_rms', None), handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Done with environment, can close
        self.env.close()
    
    def test(self):
        # Construct agent
        self.construct_agent()

        # Reconstruct environment, with NON-NORMALIZED REWARDS
        self.env = DummyVecEnv([lambda: self.env]) # Needed to use the VecNormalize wrapper
        self.env = VecNormalize(self.env, ob=True, ret=False)

        # Restore model weights
        model_path = os.path.join(self.model_dir, self.env_name, 'model_weights')
        self.actor_critic.load_weights(model_path)

        # Restored saved ob_rms to environment, disable updates to ob_rms
        saved_ob_rms = None
        vecnorm_path = os.path.join(self.model_dir, self.env_name, 'vecnorm_stats.pickle')
        with open(vecnorm_path, 'rb') as handle:
            saved_ob_rms = pickle.load(handle)

        if saved_ob_rms is not None:
            self.env.ob_rms = saved_ob_rms
        self.env.eval()

        state = self.env.reset() # Reset once at beginning, all subsequent resets handled by monitor

        """ Only performing rollout for testing """
        while True: # Indefinitely performs rollout in simulator until closed
            self.env.render() 

            _, action, _ = self.actor_critic.act(state)
            next_state, _, _, _ = self.env.step(action)
            
            state = next_state

        # Done with environment, can close
        self.env.close()

    def plot_results(self):
        # Create plot directory
        os.makedirs(self.plot_dir, exist_ok=True)

        results = pu.load_results(os.path.join(self.log_dir, self.env_name.split('-')[0], ''))
        pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False)
        plt.xlabel('Timestep')
        plt.ylabel('Reward')

        fig = plt.gcf()
        plot_path = os.path.join(self.plot_dir, 'plot_' + self.env_name)
        fig.savefig(plot_path, bbox_inches='tight')

        plt.show()