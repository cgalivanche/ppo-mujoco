import argparse

from src.agent import PPO_Agent

if __name__ == '__main__':
    formatter = lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=28)
    parser = argparse.ArgumentParser(description='MuJoCo PPO', formatter_class=formatter)

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    # General Arguments
    required.add_argument('--mode', 
                          type=str, 
                          choices={"train", "test", "plot"}, 
                          help='Train a model, test a model, or plot last results', 
                          required=True)
    required.add_argument('--env', 
                          type=str, 
                          help='Enter MuJoCo environment name',
                          required=True)
    optional.add_argument('--seed', 
                          type=int, 
                          default=0, 
                          help='Enter seed for environment and pseudo-random number generators (default: 0)',
                          metavar='')
    optional.add_argument('--save', 
                          type=int, 
                          default=25, 
                          help='Number of updates to save after (default: 25)',
                          metavar='')
    optional.add_argument('--model_dir', 
                          type=str, 
                          default='models', 
                          help='Directory to save model weights (default \'models\')',
                          metavar='')
    optional.add_argument('--log_dir', 
                          type=str, 
                          default='logs', 
                          help='Directory to save bench monitor logs (default \'logs\')',
                          metavar='')
    optional.add_argument('--plot_dir', 
                          type=str, 
                          default='plots', 
                          help='Directory to save generated plots (default \'plots\')',
                          metavar='')

    # PPO-specific Arguments
    optional.add_argument('--env_steps', 
                          type=int, 
                          default=1e6, 
                          help='Number of timesteps to train agent in environment (default: 1e6)',
                          metavar='')
    optional.add_argument('--update_steps', 
                          type=int, 
                          default=2048, 
                          help='Number of timesteps per training update (default: 2048)',
                          metavar='')
    optional.add_argument('--minibatches', 
                          type=int, 
                          default=32, 
                          help='Number of minibatches per training update (default: 32)',
                          metavar='')
    optional.add_argument('--epochs', 
                          type=int, 
                          default=4,
                          help='Number of epochs per training update (default: 4)',
                          metavar='')
    optional.add_argument('--gamma', 
                          type=float, 
                          default=0.99, 
                          help='Discount Factor - usually between 0.9 and 0.99 (default: 0.99)',
                          metavar='')
    optional.add_argument('--clip', 
                          type=float, 
                          default=0.2, 
                          help='Clip parameter for PPO (defaut: 0.2)',
                          metavar='')
    optional.add_argument('--gae_lambda', 
                          type=float, 
                          default=0.95, 
                          help='Smoothing factor to reduce variance when using GAE (default: 0.95)',
                          metavar='')
    optional.add_argument('--vf_coef', 
                          type=float, 
                          default=0.5, 
                          help='Weight for the value function\'s loss in combined loss function (default: 0.5)',
                          metavar='')
    optional.add_argument('--ent_coef', 
                          type=float, 
                          default=0.0, 
                          help='Weight for the entropy of the policy - high entropy encourages exploration over exploitation (default 0.0)',
                          metavar='')

    # Model arguments
    optional.add_argument('--a_hidden', 
                          type=int, 
                          default=[64, 64], 
                          help='Number of actor hidden units/layers - e.g. \'64 64\' for two layers of 64 hidden units',
                          nargs='+',
                          metavar='')
    optional.add_argument('--c_hidden', 
                          type=int,
                          default=[64, 64], 
                          nargs='+',
                          help='Number of critic hidden units/layers - e.g. \'64 64\' for two layers of 64 hidden units',
                          metavar='')
    optional.add_argument('--learn_rate', 
                          type=float, 
                          default=3e-4, 
                          help='Learning rate of the optimizer (default: 3e-4)',
                          metavar='')
    optional.add_argument('--epsilon', 
                          type=float, 
                          default=1e-7, 
                          help='Value of epsilon for the optimizer - episilon used to avoid divide by zero errors when gradient is near zero (default: 1e-7)',
                          metavar='')

    args = parser.parse_args()

    params = {
        'NUM_ENV_TIMESTEPS' : args.env_steps,
        'NUM_TIMESTEPS_PER_UPDATE' : args.update_steps,
        'NUM_MINIBATCHES' : args.minibatches,
        'NUM_EPOCHS' : args.epochs,

        'GAMMA' : args.gamma,
        'CLIP_PARAM' : args.clip,
        'LAMBDA' : args.gae_lambda,
        'VALUE_FUNCTION_COEF' : args.vf_coef,
        'ENTROPY_COEF' : args.ent_coef,
        
        'ACTOR_HIDDEN_UNITS' : args.a_hidden, 
        'CRITIC_HIDDEN_UNITS' : args.c_hidden,
        'LEARNING_RATE' : args.learn_rate,
        'OPTIMIZER_EPSILON' : args.epsilon, # Optimizer uses epsilon to avoid a divide by zero error when updating parameters
                                            # when the gradient is almost zero. Therefore it should be a small number, but not
                                            # too small since it can cause large weight updates.

        'SAVE_INTERVAL' : args.save
    }

    agent = PPO_Agent(params, args.env, args.model_dir, args.log_dir, args.plot_dir, seed=args.seed)
    if (args.mode == "train"):
        agent.train()
    elif (args.mode == "test"):
        agent.test()
    else:
        agent.plot_results()