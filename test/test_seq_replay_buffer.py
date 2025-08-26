import torch
from fast_td3.fast_td3_utils import SequenceReplayBuffer
from tensordict import TensorDict

def test_sequence():
    num_envs = 16
    buffer_size = 256
    n_obs = 12
    n_act = 6
    seq_len = 10
    batch_size = 32
    rb = SequenceReplayBuffer(num_envs, buffer_size, n_obs, n_act, n_obs, False)

    for i in range(buffer_size + 5):
        obs = torch.ones((num_envs, n_obs)) * i
        act = torch.ones((num_envs, n_act)) * i + buffer_size
        next_obs = torch.ones((num_envs, n_obs)) * i + buffer_size*2
        rewards = torch.ones(num_envs) * i + buffer_size*3
        truncations = torch.ones(num_envs) * i + buffer_size*4
        if i == 127 or i == buffer_size + 4:
            dones = torch.ones(num_envs)
        else:
            dones = torch.zeros(num_envs)

        transition = TensorDict(
            {
                "observations": obs,
                "actions": act,
                "next": {
                    "observations": next_obs,
                    "rewards": rewards,
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(num_envs,),
        )

        rb.extend(transition)

    for _ in range(10):
        data = rb.sample(batch_size, seq_len=seq_len)

        assert data["observations"].shape == (batch_size * num_envs, seq_len, n_obs)
        assert data["actions"].shape == (batch_size * num_envs, seq_len, n_act)
        assert data["next"]["rewards"].shape == (batch_size * num_envs, seq_len)
        assert data["next"]["dones"].shape == (batch_size * num_envs, seq_len)
        assert data["next"]["truncations"].shape == (batch_size * num_envs, seq_len)
        assert data["next"]["observations"].shape == (batch_size * num_envs, seq_len, n_obs)
        assert data["next"]["effective_n_steps"].shape == (batch_size * num_envs, seq_len)
        assert data["mask"].shape == (batch_size * num_envs, seq_len)

        # ensure data is in order
        for i in range(batch_size * num_envs):
            obs = data["observations"][i]
            actions = data["actions"][i]
            mask = data["mask"][i]
            dones = data["next"]["dones"][i]
            for j in range(seq_len - 1):
                assert torch.all(obs[j, :n_act] + buffer_size == actions[j, :n_act])
                for k in range(n_obs):
                    assert (
                        obs[j, k] == obs[j+1, k] - 1 
                        or mask[j+1] == 0
                    )
                for k in range(n_act):
                    assert (
                        actions[j, k] == actions[j+1, k] - 1 
                        or mask[j+1] == 0
                    )

                # ensure mask is correct
                if mask[j] == 0:
                    assert torch.all(mask[j:] == 0)

                if dones[j] == 1:
                    assert torch.all(mask[j+1:] == 0)
