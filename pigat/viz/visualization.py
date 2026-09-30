import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from typing import List, Tuple, Optional, Dict, Any

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


class ExperimentVisualizer:
    """Visualization class for GNN MaxCut experiment results."""
    
    def __init__(self, results_dir: str = './results'):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory containing CSV result files
        """
        self.results_dir = results_dir
        self.colors = {
            'PIGAT-v2': "#1f77b4",
            'PIGAT': "#1f77b4",
            'PIGAT-V2': '#1f77b4',
            'PIGATv2': '#1f77b4',
            'PIGATV2': '#1f77b4',
            'PIGCN': '#ff8a23',
            'pigcn': '#ff8a23',
            'pigatv2': '#1f77b4'
        }
        self.markers = {
            'PIGAT-v2': 'o',
            'PIGAT': 'o',
            'PIGAT-V2': 'o', 
            'PIGATv2': 'o',
            'PIGATV2': 'o',
            'PIGCN': 's',
            'pigcn': 's',
            'pigatv2': 'o'
        }
        
    def load_all_data(self, pattern: str = '*.csv') -> pd.DataFrame:
        """
        Load all CSV files matching the pattern.
        
        Args:
            pattern: Glob pattern for CSV files
            
        Returns:
            Combined DataFrame with all results
        """
        all_data = []
        
        # Search in results directory
        search_path = os.path.join(self.results_dir, pattern)
        csv_files = glob.glob(search_path)
        
        # Also search in current directory if no files found
        if not csv_files:
            csv_files = glob.glob(pattern)
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Standardize graph name column
                if 'Graph_name' in df.columns:
                    df['graph_name'] = df['Graph_name']
                elif 'graph_name' in df.columns:
                    df['graph_name'] = df['graph_name']
                else:
                    # Try to infer from filename if not found
                    df['graph_name'] = os.path.splitext(os.path.basename(csv_file))[0]
                
                # Standardize model names
                if 'model_name' in df.columns:
                    df['model'] = df['model_name'].str.upper().str.replace('PIGATV2', 'PIGAT')
                elif 'pigatv2' in csv_file.lower():
                    df['model'] = 'PIGAT'
                elif 'pigcn' in csv_file.lower():
                    df['model'] = 'PIGCN'
                else:
                    # Try to infer from data
                    df['model'] = 'Unknown'
                
                # Add source file info
                df['source_file'] = os.path.basename(csv_file)
                all_data.append(df)
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Loaded {len(combined_df)} total experiments")
            print(f"Found graphs: {combined_df['graph_name'].unique()}")
            return combined_df
        
        return pd.DataFrame()
    
    def aggregate_by_model_and_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by model, graph name, and number of layers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Aggregated DataFrame with means and standard deviations
        """
        # Identify seed column
        seed_col = None
        for col in ['experiment_seed', 'seed']:
            if col in df.columns:
                seed_col = col
                break
        
        # Identify metric columns based on what's available
        metric_mappings = {
            'cut_value': ['cut_value', 'gnn_cut_size'],
            'edge_cut_ratio': ['edge_cut_ratio', 'gnn_edge_cut_ratio'],
            'best_dirichlet_energy': ['best_dirichlet_energy'],
            'final_dirichlet_energy': ['final_dirichlet_energy'],
            'best_mad': ['best_mad', 'best_mean_avg_distance', 'best_mean_average_distance'],
            'final_mad': ['final_mad', 'final_mean_avg_distance', 'final_mean_average_distance'],
            'best_cosine_sim': ['best_cosine_sim', 'best_cosine_similarity'],
            'final_cosine_sim': ['final_cosine_sim', 'final_cosine_similarity'],
            'training_time': ['training_time', 'gnn_training_time']
        }
        
        # Find available columns
        available_metrics = {}
        for metric_name, possible_cols in metric_mappings.items():
            for col in possible_cols:
                if col in df.columns:
                    available_metrics[metric_name] = col
                    break
        
        # Prepare aggregation dict
        agg_dict = {}
        for metric_name, col_name in available_metrics.items():
            agg_dict[col_name] = ['mean', 'std', 'min', 'max', 'count']
        
        # Group by model, graph name, and layers
        if seed_col and df[seed_col].nunique() > 1:
            print(f"Multiple seeds detected ({df[seed_col].nunique()} unique seeds)")
            grouped = df.groupby(['model', 'graph_name', 'num_hidden_layers']).agg(agg_dict)
        else:
            print("Single seed or no seed information")
            grouped = df.groupby(['model', 'graph_name', 'num_hidden_layers']).agg(agg_dict)
        
        # Flatten multi-level columns
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        return grouped
    
    def plot_comparison(self,
                       data: pd.DataFrame,
                       metric: str,
                       title: str,
                       ylabel: str,
                       save_path: Optional[str] = None,
                       log_scale: bool = False,
                       show_std: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a comparison plot for a specific metric.
        
        Args:
            data: Aggregated DataFrame
            metric: Metric column name
            title: Plot title
            ylabel: Y-axis label
            save_path: Path to save figure
            log_scale: Whether to use log scale for y-axis
            show_std: Whether to show error bars
            
        Returns:
            Figure and axes objects
        """
        # Get graph names for title
        graph_names = data['graph_name'].unique()
        if len(graph_names) == 1:
            graph_title = f"{title} - {graph_names[0]}"
        else:
            graph_title = f"{title} - Multiple Graphs ({', '.join(graph_names[:3])}{'...' if len(graph_names) > 3 else ''})"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot for each model
        for model in data['model'].unique():
            model_data = data[data['model'] == model].sort_values('num_hidden_layers')
            
            x = model_data['num_hidden_layers']
            
            # Find the appropriate column
            y_col = None
            yerr_col = None
            
            # Check for aggregated columns
            for col in model_data.columns:
                if metric in col and '_mean' in col:
                    y_col = col
                    yerr_col = col.replace('_mean', '_std')
                    break
            
            # Fallback to direct column
            if not y_col and metric in model_data.columns:
                y_col = metric
            
            if y_col:
                y = model_data[y_col]
                
                # Get error bars if available
                yerr = None
                if show_std and yerr_col in model_data.columns:
                    yerr = model_data[yerr_col]
                    if yerr.isna().all() or (yerr == 0).all():
                        yerr = None
                
                # Choose color and marker
                color = self.colors.get(model, '#333333')
                marker = self.markers.get(model, 'o')
                
                # Plot with or without error bars
                # Main line
                ax.plot(
                    x,
                    y,
                    label=model,
                    color=color,
                    marker=marker,
                    markersize=6,
                    linewidth=2
                )

                # Shaded std region
                if yerr is not None:
                    ax.fill_between(
                        x,
                        y - yerr,
                        y + yerr,
                        color=color,
                        alpha=0.20
                    )
                else:
                    ax.plot(x, y,
                           label=model,
                           color=color,
                           marker=marker,
                           markersize=6,
                           linewidth=2,
                           alpha=0.8)
        
        ax.set_xlabel('Depth K', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(graph_title, fontsize=14, fontweight='bold')
        
        # Set log scale if requested
        if log_scale:
            ax.set_yscale('log')
        
        # Add legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=False)
        
        # Set x-axis ticks
        max_layers = int(data['num_hidden_layers'].max())
        min_layers = int(data['num_hidden_layers'].min())
        if max_layers - min_layers > 10:
            ax.set_xticks(range(min_layers, max_layers + 1, 2))
        else:
            ax.set_xticks(range(min_layers, max_layers + 1))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig, ax
    
    def create_comprehensive_plots(self,
                                 data: pd.DataFrame,
                                 output_dir: str = './plots') -> None:
        """
        Create all comparison plots for the experiment.
        
        Args:
            data: Aggregated DataFrame
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get graph names for filename prefix
        graph_names = data['graph_name'].unique()
        graph_prefix = "_".join(graph_names) if len(graph_names) <= 3 else "multiple_graphs"
        
        # Define metrics to plot based on available columns
        metric_configs = []
        
        # Check which metrics are available
        columns = data.columns.tolist()
        
        # Cut size
        for col in columns:
            if 'cut' in col.lower() and 'mean' in col:
                metric_configs.append((col.replace('_mean', ''), 'Cut Value vs Layer Depth', 'Cut Value'))
                break
        
        # Dirichlet Energy
        for col in columns:
            if 'final_dirichlet' in col.lower() and 'mean' in col:
                metric_configs.append((col.replace('_mean', ''), 'Dirichlet Energy vs Layer Depth', 
                                      'Dirichlet Energy', True))  # Log scale
                break
        
        # MAD
        for col in columns:
            if 'final' in col and ('mad' in col.lower() or 'mean_avg' in col.lower()) and 'mean' in col:
                metric_configs.append((col.replace('_mean', ''), 'Feature Dispersion (MAD) vs Layer Depth', 'Feature Dispersion (MAD)'))
                break
        
        # Cosine Similarity
        for col in columns:
            if 'final_cosine' in col.lower() and 'mean' in col:
                metric_configs.append((col.replace('_mean', ''), 'Global Cosine Similarity vs Layer Depth', 
                                      'Global Cosine Similarity'))
                break
        
        # Training Time
        for col in columns:
            if 'training_time' in col.lower() and 'mean' in col:
                metric_configs.append((col.replace('_mean', ''), 'Training Time vs Layer Depth', 
                                      'Training Time (s)'))
                break
        
        # Create plots
        for config in metric_configs:
            metric, title, ylabel = config[:3]
            log_scale = config[3] if len(config) > 3 else False
            
            save_path = os.path.join(output_dir, f'{graph_prefix}_{metric.replace("_mean", "")}_comparison.png')
            fig, ax = self.plot_comparison(data, metric, title, ylabel, save_path, log_scale)
            plt.show()
    
    def create_multi_panel_figure(self,
                                data: pd.DataFrame,
                                output_dir: str = './plots') -> None:
        """
        Create a multi-panel figure with key metrics.
        
        Args:
            data: Aggregated DataFrame
            output_dir: Directory to save plots
        """
        # Get graph names for title
        graph_names = data['graph_name'].unique()
        if len(graph_names) == 1:
            main_title = f'Model Comparison: PIGAT vs PIGCN - {graph_names[0]}'
            graph_prefix = graph_names[0]
        else:
            main_title = f'Model Comparison: PIGAT vs PIGCN - Multiple Graphs'
            graph_prefix = "multiple_graphs"
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
        # Find available metrics
        columns = data.columns.tolist()
        plot_positions = []
        
        # Define desired metrics and their positions
        desired_metrics = [
            ('cut', 'Cut Size', axes[0, 0], False),
            ('final_cosine', 'Final Cosine Similarity', axes[0, 1], False),
            ('final_dirichlet', 'Final Dirichlet Energy', axes[0, 2], True),
            ('final_mad|final_mean_avg', 'Final MAD', axes[1, 0], False),
            ('training_time', 'Training Time (s)', axes[1, 1], False),
            ('final_cosine', 'Final Cosine Similarity', axes[1, 2], False)
        ]
        
        for pattern, label, ax, log_scale in desired_metrics:
            # Find matching column
            col_found = None
            for col in columns:
                patterns = pattern.split('|')
                for p in patterns:
                    if p in col.lower() and 'mean' in col:
                        col_found = col.replace('_mean', '')
                        break
                if col_found:
                    break
            
            if col_found:
                plot_positions.append((col_found, label, ax, log_scale))
            else:
                ax.text(0.5, 0.5, f'{label}\n(Data not available)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Create plots
        for metric, ylabel, ax, log_scale in plot_positions:
            for model in data['model'].unique():
                model_data = data[data['model'] == model].sort_values('num_hidden_layers')
                
                x = model_data['num_hidden_layers']
                
                # Find the column
                y_col = None
                yerr_col = None
                for col in model_data.columns:
                    if metric in col and '_mean' in col:
                        y_col = col
                        yerr_col = col.replace('_mean', '_std')
                        break
                
                if not y_col and metric in model_data.columns:
                    y_col = metric
                
                if y_col:
                    y = model_data[y_col]
                    yerr = model_data[yerr_col] if yerr_col in model_data.columns else None
                    
                    if yerr is not None and (yerr.isna().all() or (yerr == 0).all()):
                        yerr = None
                    
                    color = self.colors.get(model, '#333333')
                    marker = self.markers.get(model, 'o')
                    
                    ax.plot(
                        x,
                        y,
                        label=model,
                        color=color,
                        marker=marker,
                        markersize=6,
                        linewidth=2
                    )

                    # Shaded std region
                    if yerr is not None:
                        ax.fill_between(
                            x,
                            y - yerr,
                            y + yerr,
                            color=color,
                            alpha=0.20
                        )
                    else:
                        ax.plot(x, y,
                               label=model,
                               color=color,
                               marker=marker,
                               markersize=5,
                               linewidth=1.5,
                               alpha=0.8)
            
            ax.set_xlabel('Layers', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(ylabel, fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if log_scale:
                ax.set_yscale('log')
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'{graph_prefix}_multi_panel_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the experiments.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Summary statistics DataFrame
        """
        summary = []
        
        for graph_name in df['graph_name'].unique():
            graph_data = df[df['graph_name'] == graph_name]
            
            for model in graph_data['model'].unique():
                model_data = graph_data[graph_data['model'] == model]
                
                # Find best configuration
                cut_col = None
                for col in ['cut_value', 'gnn_cut_size']:
                    if col in model_data.columns:
                        cut_col = col
                        break
                
                if cut_col:
                    best_idx = model_data[cut_col].idxmax()
                    best_row = model_data.loc[best_idx]
                    
                    summary.append({
                        'Graph': graph_name,
                        'Model': model,
                        'Best Cut': best_row[cut_col],
                        'Best Layers': best_row['num_hidden_layers'],
                        'Avg Cut': model_data[cut_col].mean(),
                        'Std Cut': model_data[cut_col].std(),
                        'Total Experiments': len(model_data)
                    })
        
        return pd.DataFrame(summary)


def main():
    """Main function to create all visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize GNN MaxCut experiment results')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing result CSV files')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Directory to save plots')
    parser.add_argument('--pattern', type=str, default='*.csv',
                       help='Pattern for CSV files to load')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ExperimentVisualizer(args.results_dir)
    
    # Load data
    print("Loading experiment data...")
    df = visualizer.load_all_data(args.pattern)
    
    if df.empty:
        print("No data files found. Please check the results directory.")
        return
    
    print(f"\nFound data for models: {df['model'].unique()}")
    print(f"Layer range: {df['num_hidden_layers'].min()} - {df['num_hidden_layers'].max()}")
    
    # Generate summary statistics
    print("\nSummary Statistics:")
    summary = visualizer.generate_summary_statistics(df)
    print(summary.to_string(index=False))
    
    # Aggregate data
    print("\nAggregating data by model and layer...")
    aggregated_data = visualizer.aggregate_by_model_and_layer(df)
    
    # Create plots
    print("\nCreating individual comparison plots...")
    visualizer.create_comprehensive_plots(aggregated_data, args.output_dir)
    
    print("\nCreating multi-panel comparison figure...")
    visualizer.create_multi_panel_figure(aggregated_data, args.output_dir)
    
    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()