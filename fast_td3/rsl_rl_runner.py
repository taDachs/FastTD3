import torch
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper 
from rsl_rl.runners import OnPolicyRunner


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

def load_teacher_policy(env, checkpoint_path: str):
    agent_cfg = BasePPORunnerCfg()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # TODO: is there a nicer way to do this?
    env.num_obs = env.unwrapped.observation_manager.group_obs_dim["critic"][0]

    def get_observations() -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(env.unwrapped, "observation_manager"):
            obs_dict = env.unwrapped.observation_manager.compute()
        else:
            obs_dict = env.unwrapped._get_observations()
        return obs_dict["critic"], {"observations": obs_dict}

    env.get_observations = get_observations
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(checkpoint_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    return policy
