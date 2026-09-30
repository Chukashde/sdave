#!/usr/bin/env python
import sys
import os

from pigat.viz import ExperimentVisualizer
import argparse
import pandas as pd


def analyze_results(csv_file: str) -> None:
    """
    Analyze a single results file and print key statistics.
    
    Args:
        csv_file: Path to CSV file
    """
    df = pd.read_csv(csv_file)
    
    print(f"\nAnalyzing: {csv_file}")
    print("="*60)
    
    # Basic info
    print(f"Total experiments: {len(df)}")
    
    if 'model_name' in df.columns:
        print(f"Models: {df['model_name'].unique()}")
    
    if 'num_hidden_layers' in df.columns:
        print(f"Layer range: {df['num_hidden_layers'].min()}-{df['num_hidden_layers'].max()}")
    
    # Find cut column
    cut_col = None
    for col in ['cut_value', 'gnn_cut_size']:
        if col in df.columns:
            cut_col = col
            break
    
    if cut_col:
        print(f"\nPerformance Statistics:")
        print(f"  Best cut: {df[cut_col].max()}")
        print(f"  Average cut: {df[cut_col].mean():.2f}")
        print(f"  Std dev: {df[cut_col].std():.2f}")
        
        # Best configuration
        best_idx = df[cut_col].idxmax()
        best_row = df.loc[best_idx]
        print(f"\nBest Configuration:")
        if 'model_name' in df.columns:
            print(f"  Model: {best_row['model_name']}")
        print(f"  Layers: {best_row['num_hidden_layers']}")
        if 'seed' in df.columns or 'experiment_seed' in df.columns:
            seed_col = 'seed' if 'seed' in df.columns else 'experiment_seed'
            print(f"  Seed: {best_row[seed_col]}")
    
    # Oversmoothing metrics
    print(f"\nOversmoothing Metrics (Best Solutions):")
    if 'best_dirichlet_energy' in df.columns:
        print(f"  Dirichlet Energy range: {df['best_dirichlet_energy'].min():.4f} - {df['best_dirichlet_energy'].max():.4f}")
    if 'best_mad' in df.columns or 'best_mean_avg_distance' in df.columns:
        mad_col = 'best_mad' if 'best_mad' in df.columns else 'best_mean_avg_distance'
        print(f"  MAD range: {df[mad_col].min():.4f} - {df[mad_col].max():.4f}")
    if 'best_cosine_sim' in df.columns or 'best_cosine_similarity' in df.columns:
        cos_col = 'best_cosine_sim' if 'best_cosine_sim' in df.columns else 'best_cosine_similarity'
        print(f"  Cosine Similarity range: {df[cos_col].min():.4f} - {df[cos_col].max():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GNN MaxCut experiment results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Directory containing result CSV files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./plots',
        help='Directory to save plots'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.csv',
        help='Pattern for CSV files to load'
    )
    
    parser.add_argument(
        '--analyze_only',
        action='store_true',
        help='Only analyze data without creating plots'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Specific CSV file to analyze'
    )
    
    args = parser.parse_args()
    
    # If specific file provided, analyze it
    if args.file:
        if os.path.exists(args.file):
            analyze_results(args.file)
        else:
            print(f"Error: File {args.file} not found")
            sys.exit(1)
        
        if args.analyze_only:
            return
    
    # Create visualizer
    visualizer = ExperimentVisualizer(args.results_dir)
    
    # Load data
    print("Loading experiment data...")
    df = visualizer.load_all_data(args.pattern)
    
    if df.empty:
        print("\nNo data files found!")
        print(f"Searched in: {args.results_dir}")
        print(f"Pattern: {args.pattern}")
        print("\nPlease run experiments first:")
        print("  python scripts/run_sweep.py --graph dataset/Gset/G14")
        return
    
    print(f"\nFound data for models: {df['model'].unique()}")
    
    # Find layer range
    if 'num_hidden_layers' in df.columns:
        print(f"Layer range: {df['num_hidden_layers'].min()} - {df['num_hidden_layers'].max()}")
    
    # Check for seeds
    seed_info = "unknown"
    if 'experiment_seed' in df.columns:
        seed_info = df['experiment_seed'].nunique()
    elif 'seed' in df.columns:
        seed_info = df['seed'].nunique()
    print(f"Number of seeds: {seed_info}")
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    summary = visualizer.generate_summary_statistics(df)
    print(summary.to_string(index=False))
    
    if args.analyze_only:
        print("\nAnalysis complete. Use --analyze_only=False to generate plots.")
        return
    
        # Process graph by graph
    graph_names = df['graph_name'].unique()

    for graph_name in graph_names:
        print(f"\nProcessing graph: {graph_name}")

        # Filter single graph
        graph_df = df[df['graph_name'] == graph_name]

        # Aggregate only this graph
        aggregated_data = visualizer.aggregate_by_model_and_layer(graph_df)

        # Create plots
        print(f"Creating plots for {graph_name}...")
        visualizer.create_comprehensive_plots(
            aggregated_data,
            args.output_dir
        )

        visualizer.create_multi_panel_figure(
            aggregated_data,
            args.output_dir
        )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"All plots saved to: {args.output_dir}/")
    print("\nGenerated files:")
    
    # List generated files
    import glob
    plot_files = glob.glob(os.path.join(args.output_dir, '*.png'))
    for plot_file in sorted(plot_files):
        print(f"  - {os.path.basename(plot_file)}")


if __name__ == "__main__":
    main()