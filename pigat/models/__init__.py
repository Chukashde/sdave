from pigat.models.pigat_jk import PIGATJK
import torch.nn as nn

from pigat.models.pigcn import PIGCN
from pigat.models.pigcn_jk import PIGCNJK
from pigat.models.pigatv2 import PIGATv2
from pigat.models.pigatv2_jk import PIGATv2JK

__all__ = ["PIGCN", "PIGCNJK", "PIGATv2", "PIGATv2JK", "create_model"]


def create_model(model_name: str, config: dict) -> nn.Module:
    """
    Factory function to create a model instance.

    Args:
        model_name: Name of the model ('pigcn' or 'pigatv2')
        config: Configuration dictionary

    Returns:
        Model instance
    """
    if model_name == 'pigcn':
        return PIGCN(
            embedding_dim=config['dim_embedding'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['number_classes'],
            dropout=config['dropout'],
            device=config['device']
        )
    elif model_name == 'pigcn_jk':
        return PIGCNJK(
            embedding_dim=config['dim_embedding'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['number_classes'],
            dropout=config['dropout'],
            device=config['device']
        )
    elif model_name == 'pigatv2':
        return PIGATv2(
            embedding_dim=config['dim_embedding'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['number_classes'],
            dropout=config['dropout'],
            device=config['device'],
            num_heads=config['num_heads'],
            feat_drop=config['feat_drop'],
            attn_drop=config['attn_drop']
        )
    elif model_name == 'pigat_jk':
        return PIGATJK(
            embedding_dim=config['dim_embedding'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['number_classes'],
            dropout=config['dropout'],
            device=config['device'],
            num_heads=config['num_heads'],
            feat_drop=config['feat_drop'],
            attn_drop=config['attn_drop']
        )
    elif model_name == 'pigatv2_jk':
        return PIGATv2JK(
            embedding_dim=config['dim_embedding'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['number_classes'],
            dropout=config['dropout'],
            device=config['device'],
            num_heads=config['num_heads'],
            feat_drop=config['feat_drop'],
            attn_drop=config['attn_drop']
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
