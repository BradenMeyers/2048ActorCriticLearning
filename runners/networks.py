import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from runners.utils import N_ACTIONS, STATE_DIM, N_TILE_LEVELS

# = network
class CNNActorCritic(nn.Module):
    """
    CNN shared-trunk actor-critic.

    Why CNN?
        The merge condition — two equal adjacent tiles — is a 2x1 spatial
        pattern. A Conv2d filter can represent this directly. A flat linear
        network has to learn the same pattern implicitly across all 256 inputs,
        which takes much longer and has to relearn it for every tile level.

    Architecture:
        Input (1, 4, 4)
          → Conv2d(1→64,   kernel=2) → ReLU   # local 2x2 patterns
          → Conv2d(64→128, kernel=2) → ReLU   # higher-level structure
          → Flatten → Linear(512→256) → ReLU
               ↓
        ┌──────┴──────┐
        ↓             ↓
     Actor head   Critic head
     Linear(4)    Linear(1)
     softmax      raw scalar
     → π(a|s)     → V(s)
    """

    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2),    # (1,4,4) → (64,3,3)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2),  # (64,3,3) → (128,2,2)
            nn.ReLU(),
            nn.Flatten(),                        # → (512,)
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
        )

        self.actor_head  = nn.Linear(256, N_ACTIONS)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, mask: torch.Tensor = None):
        """
        Parameters
        ----------
        state : (1, 4, 4) from board_to_tensor — (channels, H, W)
                Conv2d needs (batch, C, H, W) = (1, 1, 4, 4), added here.
        mask  : (4,) bool — True = legal move

        Returns
        -------
        policy : (4,)  probability over actions  (batch dim squeezed)
        value  : (1,)  scalar state value        (batch dim squeezed)
        """
        # (1, 4, 4) → (1, 1, 4, 4): add batch dimension for Conv2d
        if state.dim() == 3:
            state = state.unsqueeze(0)

        features = self.trunk(state)          # → (1, 256)
        logits   = self.actor_head(features)  # → (1, 4)

        # Mask illegal moves — unsqueeze mask to broadcast over batch dim
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))

        policy = F.softmax(logits, dim=-1).squeeze(0)  # → (4,)
        value  = self.critic_head(features).squeeze(0) # → (1,)

        return policy, value
    
    # = state encoder

    def board_to_tensor(self, board: np.ndarray) -> torch.Tensor:
        """
        Convert 4x4 numpy board to a (1, 4, 4) float tensor using log2 encoding.

        Each cell → log2(value) / 15.0, normalised to [0, 1].
        Empty cells → 0.0.

        Why log2 instead of one-hot?
            One-hot treats each tile level as an independent category — the network
            has no idea 256 and 512 are related. Log2 encoding gives tiles ordinal
            meaning: equal adjacent cells are always exactly 1/15 apart in value,
            so the CNN can learn the merge condition (two equal neighbours) as a
            single consistent filter regardless of which tile level it is.

        Shape: (1, 4, 4) = (channels, H, W).
            forward() adds the batch dim → (1, 1, 4, 4) before Conv2d.
        """
        state = np.zeros((1, 4, 4), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                val = board[r, c]
                state[0, r, c] = math.log2(val) / 15.0 if val > 0 else 0.0
        return torch.tensor(state, dtype=torch.float32)
    

class LinearActorCritic(nn.Module):
    """
    Shared-trunk network with two heads:
        - Actor head  → policy π(a|s), probability over 4 moves
        - Critic head → value V(s), expected future return

    Architecture:
        Input (256) → Linear(512) → ReLU → Linear(512) → ReLU
                                                ↓
                              ┌─────────────────┤
                              ↓                 ↓
                         Actor head        Critic head
                         Linear(4)         Linear(1)
                         (logits)          (scalar)
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 512):
        super().__init__()

        # Shared trunk — learns board representations useful to both heads
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor head — outputs raw logits (NOT probabilities yet)
        self.actor_head = nn.Linear(hidden, N_ACTIONS)

        # Critic head — outputs a single scalar value estimate
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, mask: torch.Tensor = None):
        """
        Parameters
        ----------
        state : (batch, 256) or (256,) float tensor
        mask  : (batch, 4) or (4,) bool tensor — True = legal move
                If provided, illegal moves get -inf logits before softmax.

        Returns
        -------
        policy : (batch, 4) action probability distribution
        value  : (batch, 1) state value estimate
        """
        features = self.trunk(state)

        logits = self.actor_head(features)

        # Mask illegal moves — set their logits to -inf so softmax → 0
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        policy = F.softmax(logits, dim=-1)
        value  = self.critic_head(features)

        return policy, value
    
    # = state encoder

    def board_to_tensor(self,board: np.ndarray) -> torch.Tensor:
        """
        Convert a 4x4 numpy board to a (256,) one-hot float tensor.

        Each cell becomes a one-hot vector of length 16:
            empty  → index 0
            tile 2 → index 1
            tile 4 → index 2
            ...
            tile 2^15 → index 15
        """
        indices = np.where(board > 0, np.log2(board.clip(1)).astype(np.int32), 0).clip(0, N_TILE_LEVELS - 1)
        state = np.zeros((16, N_TILE_LEVELS), dtype=np.float32)
        state[np.arange(16), indices.flatten()] = 1.0
        return torch.from_numpy(state.flatten())