import math

import torch
import torch.nn as nn
from .fast_td3_utils import EmpiricalNormalization
from .fast_td3 import Actor, RNNActor
from .fast_td3_simbav2 import Actor as ActorSimbaV2, RNNActor as RNNActorSimbaV2
from .vision_backbone import DepthOnlyFCBackbone58x87
from fast_td3 import vision_backbone as vision_backbone


class Policy(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        args: dict,
        use_memory: bool = False,
        agent: str = "fasttd3",
    ):
        super().__init__()

        self.args = args
        self.n_obs = n_obs
        self.n_act = n_act

        self.use_memory = use_memory
        self.use_vision = args["use_vision"]

        num_envs = args["num_envs"]
        init_scale = args["init_scale"]
        actor_hidden_dim = args["actor_hidden_dim"]

        actor_kwargs = dict(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            device="cpu",
            init_scale=init_scale,
            hidden_dim=actor_hidden_dim,
            squash=args["squash"],
            noise_scheduling=args["noise_scheduling"],
            use_layer_norm=args["use_layer_norm"],
        )

        if self.use_memory:
            actor_kwargs["memory_type"] = args["memory_type"]
            actor_kwargs["memory_hidden_dim"] = args["memory_hidden_dim"]

        if self.use_vision:
            actor_kwargs["use_vision_latent"] = args["use_vision"]
            actor_kwargs["vision_latent_dim"] = args["vision_latent_dim"]



        if agent == "fasttd3":
            if use_memory:
                actor_cls = RNNActor
            else:
                actor_cls = Actor
        elif agent == "fasttd3_simbav2":
            if use_memory:
                actor_cls = RNNActorSimbaV2
            else:
                actor_cls = ActorSimbaV2

            actor_num_blocks = args["actor_num_blocks"]
            actor_kwargs.pop("init_scale")
            actor_kwargs.pop("use_layer_norm")
            actor_kwargs.update(
                {
                    "scaler_init": math.sqrt(2.0 / actor_hidden_dim),
                    "scaler_scale": math.sqrt(2.0 / actor_hidden_dim),
                    "alpha_init": 1.0 / (actor_num_blocks + 1),
                    "alpha_scale": 1.0 / math.sqrt(actor_hidden_dim),
                    "expansion": 4,
                    "c_shift": 3.0,
                    "num_blocks": actor_num_blocks,
                }
            )
        else:
            raise ValueError(f"Agent {agent} not supported")

        self.actor = actor_cls(
            **actor_kwargs,
        )
        if self.use_vision:
            self.vision_nn = DepthOnlyFCBackbone58x87(args["vision_latent_dim"])
        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device="cpu")

        self.actor.eval()
        self.obs_normalizer.eval()

    @torch.no_grad
    def forward(
        self,
        obs: torch.Tensor,
        hidden_in: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        image: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        norm_obs = self.obs_normalizer(obs)
        if self.use_memory:
            if self.use_vision:
                vision_latent = self.vision_nn(image.permute(0, 3, 1, 2) / self.args["normalize_vision_scalar"])
                vision_latent = vision_latent.unsqueeze(1)
            else:
                vision_latent = None
            norm_obs = norm_obs.unsqueeze(1)
            actions, hidden_out = self.actor(norm_obs, hidden_in=hidden_in, vision_latent=vision_latent)
            actions = actions.squeeze(1)

            return actions, hidden_out
        else:
            actions = self.actor(norm_obs)
            return actions


def load_policy(checkpoint_path, use_memory):
    torch_checkpoint = torch.load(
        f"{checkpoint_path}", map_location="cpu", weights_only=False,
    )
    args = torch_checkpoint["args"]

    agent = args.get("agent", "fasttd3")
    if agent == "fasttd3":
        if use_memory:
            n_obs = torch_checkpoint["actor_state_dict"]["memory.weight_ih_l0"].shape[-1]
            if args["use_vision"]:
                n_obs -= args["vision_latent_dim"]
        else:
            n_obs = torch_checkpoint["actor_state_dict"]["net.0.weight"].shape[-1]
        n_act = torch_checkpoint["actor_state_dict"]["fc_mu.0.weight"].shape[0]
    elif agent == "fasttd3_simbav2":
        # TODO: Too hard-coded, maybe save n_obs and n_act in the checkpoint?
        if use_memory:
            n_obs = torch_checkpoint["actor_state_dict"]["memory.weight_ih_l0"].shape[-1]
            if args["use_vision"]:
                n_obs -= args["vision_latent_dim"]
        else:
            n_obs = (
                torch_checkpoint["actor_state_dict"]["embedder.w.w.weight"].shape[-1] - 1
            )
        n_act = torch_checkpoint["actor_state_dict"]["predictor.mean_bias"].shape[0]
    else:
        raise ValueError(f"Agent {agent} not supported")

    policy = Policy(
        n_obs=n_obs,
        n_act=n_act,
        args=args,
        use_memory=use_memory,
        agent=agent,
    )
    policy.actor.load_state_dict(torch_checkpoint["actor_state_dict"])
    if args["use_vision"]:
        cleaned_state_dict = {}
        bad_name = "_orig_mod."

        for k, v in torch_checkpoint["vision_model"].items():
            if bad_name in k:
                cleaned_state_dict[k[len(bad_name):]] = v
            else:
                cleaned_state_dict[k] = v
        policy.vision_nn.load_state_dict(cleaned_state_dict)

    if len(torch_checkpoint["obs_normalizer_state"]) == 0:
        policy.obs_normalizer = nn.Identity()
    else:
        policy.obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])

    return policy
