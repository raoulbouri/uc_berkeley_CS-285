import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    _debug_done = False  # DEBUG: class-level flag; prints once across all instances

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # DEBUG: print input shapes before and after conversion (first call only)
        if not MLPPolicy._debug_done:
            print("\n" + "=" * 60)
            print("DEBUG [get_action] — Observation conversion")
            print(f"  obs numpy shape:         {obs.shape}  dtype: {obs.dtype}")

        obs_tensor = ptu.from_numpy(obs).unsqueeze(0)

        if not MLPPolicy._debug_done:
            print(f"  obs tensor shape:        {tuple(obs_tensor.shape)}  [batch dim added]")
            print(f"  device:                  {obs_tensor.device}")

        dist = self.forward(obs_tensor)  # forward() prints its own debug block
        action = dist.sample().squeeze(0)

        if not MLPPolicy._debug_done:
            print(f"  sampled action shape:    {tuple(action.shape)}")
            print(f"  sampled action value:    {ptu.to_numpy(action)}")
            print("=" * 60)
            MLPPolicy._debug_done = True

        return ptu.to_numpy(action)

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            action_logits = self.logits_net(obs)
            # DEBUG: print logits shape and values (first call only)
            if not MLPPolicy._debug_done:
                print(f"\nDEBUG [forward — discrete]:")
                print(f"  input obs shape:         {tuple(obs.shape)}")
                print(f"  output logits shape:     {tuple(action_logits.shape)}  [one logit per action]")
                print(f"  logits (first sample):   {action_logits[0].detach().cpu().numpy()}")
                print(f"  distribution type:       Categorical")
            return distributions.Categorical(logits=action_logits)
        else:
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            # DEBUG: print mean/std shape and values (first call only)
            if not MLPPolicy._debug_done:
                print(f"\nDEBUG [forward — continuous]:")
                print(f"  input obs shape:         {tuple(obs.shape)}")
                print(f"  mean shape:              {tuple(mean.shape)}")
                print(f"  std values (exp logstd): {std.detach().cpu().numpy()}")
                print(f"  distribution type:       Independent(Normal, reinterpreted_dims=1)")
            return distributions.Independent(distributions.Normal(mean, std), 1)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
    
        # Step 1: Get the action distribution from the forward pass
        action_dist = self.forward(obs)
    
        # Step 2: Compute log π(a|s) for the actions actually taken
        log_probs = action_dist.log_prob(actions)
    
        # Step 3: Policy gradient loss
        loss = -(log_probs * advantages).mean()
    
        # Step 4: Optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return {
            "Actor Loss": loss.item(),
        }
