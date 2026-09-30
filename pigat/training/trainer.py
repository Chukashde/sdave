import torch
import torch.nn as nn
import dgl
from time import time
from typing import Tuple, Dict, Any, Optional
from itertools import chain

import os

from pigat.models import create_model
from pigat.eval.maxcut import loss_func

embed_path = './embed.pt'

class Trainer:
    """Trainer class for GNN models."""
    
    def __init__(self, config: dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config['device']
        self.dtype = config['dtype']
        
    def create_model_and_optimizer(self, 
                                  model_name: str, 
                                  n_nodes: int) -> Tuple[nn.Module, nn.Embedding, torch.optim.Adam]:
        """
        Create model, embedding layer, and optimizer.
        
        Args:
            model_name: Name of the model
            n_nodes: Number of nodes in the graph
            
        Returns:
            Tuple of (model, embedding, optimizer)
        """
        # Create model
        net = create_model(model_name, self.config)
        net = net.type(self.dtype).to(self.device)
        
        # Create embedding layer
        # if os.path.exists(embed_path):
        #     print("🔹 Loading existing embedding...")
        #     # embed = nn.Embedding(n_nodes, self.config['dim_embedding'])
        #     # embed.load_state_dict(torch.load(embed_path))
        # else:
        print("🆕 Creating new embedding and saving to file...")
        embed = nn.Embedding(n_nodes, self.config['dim_embedding'])
        # torch.manual_seed(42)  # optional, for reproducibility
        # nn.init.uniform_(embed.weight, -1.0, 1.0)  # custom init
        # torch.save(embed.state_dict(), embed_path)
        embed = embed.type(self.dtype).to(self.device)
        
        # Create optimizer
        params = chain(net.parameters(), embed.parameters())
        optimizer = torch.optim.Adam(params, lr=self.config['learning_rate'])
        
        return net, embed, optimizer
    
    def train(self, 
             net: nn.Module, 
             embed: nn.Embedding, 
             optimizer: torch.optim.Adam,
             dgl_graph: dgl.DGLGraph,
             q_torch: torch.Tensor,
             verbose: bool = True) -> Dict[str, Any]:
        """
        Train the GNN model.
        
        Args:
            net: GNN model
            embed: Embedding layer
            optimizer: Optimizer
            dgl_graph: DGL graph
            q_torch: QUBO matrix as torch tensor
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training results
        """
        # Initialize tracking variables
        inputs = embed.weight
        prev_loss = 1.0
        patience_count = 0
        
        # Track best solution
        best_bitstring = torch.zeros((dgl_graph.number_of_nodes(),)).type(self.dtype).to(self.device)
        best_loss = loss_func(best_bitstring.float(), q_torch)
        best_epoch = 0
        best_last_hidden = None
        
        # Track training history
        loss_history = []
        
        t_start = time()
        
        # Training loop
        for epoch in range(self.config['number_epochs']):
            # Forward pass
            probs, last_hidden = net(dgl_graph, inputs)
            probs = probs[:, 0] if len(probs.shape) > 1 else probs
            
            # Calculate loss
            loss = loss_func(probs, q_torch)
            loss_value = loss.detach().item()
            loss_history.append(loss_value)
            
            # Get discrete solution
            bitstring = (probs.detach() >= self.config['prob_threshold']) * 1
            
            # Update best solution
            if loss < best_loss:
                best_loss = loss
                best_bitstring = bitstring
                best_epoch = epoch
                best_last_hidden = last_hidden.detach() if last_hidden is not None else None
            
            # Check convergence
            if (abs(loss_value - prev_loss) <= self.config['tolerance']) or (loss_value - prev_loss > 0):
                patience_count += 1
            else:
                patience_count = 0
            
            if patience_count >= self.config['patience']:
                if verbose:
                    print(f'Early stopping at epoch {epoch} (patience: {self.config["patience"]})')
                break
            
            prev_loss = loss_value
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{self.config["number_epochs"]}: Loss = {loss_value:.6f}')
        
        t_end = time()
        training_time = t_end - t_start
        
        # Get final solution
        with torch.no_grad():
            final_probs, final_last_hidden = net(dgl_graph, inputs)
            final_probs = final_probs[:, 0] if len(final_probs.shape) > 1 else final_probs
            final_bitstring = (final_probs.detach() >= self.config['prob_threshold']) * 1
        
        if verbose:
            print(f'Training completed in {training_time:.3f}s')
            print(f'Final loss: {loss_value:.6f}')
            print(f'Best loss: {best_loss.item():.6f} at epoch {best_epoch}')
        
        return {
            'final_epoch': epoch,
            'training_time': training_time,
            'final_loss': loss_value,
            'best_loss': best_loss.item(),
            'best_epoch': best_epoch,
            'final_bitstring': final_bitstring,
            'best_bitstring': best_bitstring,
            'final_last_hidden': final_last_hidden,
            'best_last_hidden': best_last_hidden,
            'loss_history': loss_history,
            'converged': patience_count >= self.config['patience']
        }


def quick_train(model_name: str,
               nx_graph,
               q_torch: torch.Tensor,
               config: dict,
               verbose: bool = True) -> Dict[str, Any]:
    """
    Quick training function for single experiment.
    
    Args:
        model_name: Name of the model
        nx_graph: NetworkX graph
        q_torch: QUBO matrix
        config: Configuration dictionary
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training results
    """
    # Create DGL graph
    dgl_graph = dgl.from_networkx(nx_graph=nx_graph)
    pass  # DGL graph stays on CPU
    
    # Create trainer
    trainer = Trainer(config)
    
    # Create model and optimizer
    net, embed, optimizer = trainer.create_model_and_optimizer(
        model_name, nx_graph.number_of_nodes()
    )
    
    # Train model
    results = trainer.train(net, embed, optimizer, dgl_graph, q_torch, verbose)
    
    return results
