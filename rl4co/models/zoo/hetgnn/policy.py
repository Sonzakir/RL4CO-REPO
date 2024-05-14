from typing import Optional

import torch.nn as nn

from rl4co.models.common.constructive.autoregressive import (
    AutoregressiveDecoder,
    AutoregressiveEncoder,
    AutoregressivePolicy,
)
from rl4co.utils.pylogger import get_pylogger

from .decoder import HetGNNDecoder
from .encoder import HetGNNEncoder

log = get_pylogger(__name__)


class HetGNNPolicy(AutoregressivePolicy):
    def __init__(
        self,
        encoder: Optional[AutoregressiveEncoder] = None,
        decoder: Optional[AutoregressiveDecoder] = None,
        embed_dim: int = 64,
        num_encoder_layers: int = 2,
        env_name: str = "fjsp",
        init_embedding: Optional[nn.Module] = None,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "multistart_sampling",
        **constructive_policy_kw,
    ):
        if len(constructive_policy_kw) > 0:
            log.warn(f"Unused kwargs: {constructive_policy_kw}")

        if encoder is None:
            encoder = HetGNNEncoder(
                env_name=env_name,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                init_embedding=init_embedding,
            )

        # The decoder generates logits given the current td and heatmap
        if decoder is None:
            decoder = HetGNNDecoder(
                embed_dim=embed_dim,
                feed_forward_hidden_dim=embed_dim,
                feed_forward_layers=2,
            )

        # Pass to constructive policy
        super(HetGNNPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )
