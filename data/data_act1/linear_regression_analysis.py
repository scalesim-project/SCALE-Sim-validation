#!/usr/bin/env python3
"""
Linear Regression Analysis for Activation Functions
Analyzes the relationship between kernel shape product and latency for different activation functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import os

def load_data(file_path):
    """Load the merged functions data."""
    df = pd.read_csv(file_path)
    return df

def perform_linear_regression(x, y):
    """Perform linear regression and return results."""
    # Reshape for sklearn
    X = x.values.reshape(-1, 1)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate correlation coefficient
    correlation, p_value = stats.pearsonr(x, y)
    
    return {
        'model': model,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r2_score': r2,
        'mse': mse,
        'rmse': rmse,
        'correlation': correlation,
        'p_value': p_value,
        'predictions': y_pred
    }

def analyze_all_activation_functions(df):
    """Perform linear regression for all activation functions."""
    # Get activation function columns (exclude kernel_shape and tuple_product)
    activation_columns = [col for col in df.columns if col.endswith('_avg_duration_us')]
    
    results = {}
    
    print("Linear Regression Analysis Results")
    print("="*60)
    print(f"{'Activation Function':<20} {'Slope':<12} {'Intercept':<12} {'R²':<8} {'RMSE':<10} {'P-value':<10}")
    print("-"*80)
    
    for activation in activation_columns:
        # Extract activation function name
        act_name = activation.replace('_avg_duration_us', '')
        
        # Perform linear regression
        result = perform_linear_regression(df['tuple_product'], df[activation])
        results[act_name] = result
        
        # Print results
        print(f"{act_name:<20} {result['slope']:<12.6f} {result['intercept']:<12.6f} "
              f"{result['r2_score']:<8.4f} {result['rmse']:<10.6f} {result['p_value']:<10.3e}")
    
    return results

def create_visualization(df, results):
    """Create visualizations for the linear regression analysis."""
    activation_columns = [col for col in df.columns if col.endswith('_avg_duration_us')]
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Linear Regression: Kernel Shape Product vs Latency for Activation Functions', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, activation in enumerate(activation_columns):
        ax = axes[i]
        act_name = activation.replace('_avg_duration_us', '')
        
        # Plot scatter points
        ax.scatter(df['tuple_product'], df[activation], alpha=0.6, s=30, color='steelblue')
        
        # Plot regression line
        result = results[act_name]
        ax.plot(df['tuple_product'], result['predictions'], color='red', linewidth=2)
        
        # Formatting
        ax.set_xlabel('Kernel Shape Product')
        ax.set_ylabel('Latency (μs)')
        ax.set_title(f'{act_name.upper()}\nR² = {result["r2_score"]:.4f}')
        ax.grid(True, alpha=0.3)
        
        # Add equation text
        equation = f'y = {result["slope"]:.2e}x + {result["intercept"]:.4f}'
        ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_comparison_plot(results):
    """Create comparison plots for different metrics."""
    act_names = list(results.keys())
    
    # Extract metrics
    slopes = [results[act]['slope'] for act in act_names]
    intercepts = [results[act]['intercept'] for act in act_names]
    r2_scores = [results[act]['r2_score'] for act in act_names]
    correlations = [results[act]['correlation'] for act in act_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Linear Regression Metrics Across Activation Functions', 
                 fontsize=16, fontweight='bold')
    
    # Slopes
    axes[0,0].bar(act_names, slopes, color='skyblue', edgecolor='navy', alpha=0.7)
    axes[0,0].set_title('Slopes (Rate of Change)')
    axes[0,0].set_ylabel('Slope')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # Intercepts
    axes[0,1].bar(act_names, intercepts, color='lightcoral', edgecolor='darkred', alpha=0.7)
    axes[0,1].set_title('Intercepts (Base Latency)')
    axes[0,1].set_ylabel('Intercept (μs)')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # R² scores
    axes[1,0].bar(act_names, r2_scores, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    axes[1,0].set_title('R² Scores (Goodness of Fit)')
    axes[1,0].set_ylabel('R² Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim(0, 1)
    
    # Correlations
    axes[1,1].bar(act_names, correlations, color='gold', edgecolor='orange', alpha=0.7)
    axes[1,1].set_title('Correlation Coefficients')
    axes[1,1].set_ylabel('Correlation')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(-1, 1)
    
    plt.tight_layout()
    return fig

def save_results_to_csv(results, output_path):
    """Save regression results to a CSV file."""
    results_data = []
    
    for act_name, result in results.items():
        results_data.append({
            'activation_function': act_name,
            'slope': result['slope'],
            'intercept': result['intercept'],
            'r2_score': result['r2_score'],
            'mse': result['mse'],
            'rmse': result['rmse'],
            'correlation': result['correlation'],
            'p_value': result['p_value']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    return results_df

def main():
    """Main analysis function."""
    # File paths
    data_file = 'merged_functions_by_shape.csv'
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found in current directory")
        return
    
    # Load data
    print("Loading data...")
    df = load_data(data_file)
    print(f"Loaded {len(df)} data points")
    
    # Perform analysis
    print("\nPerforming linear regression analysis...")
    results = analyze_all_activation_functions(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Main regression plots
    fig1 = create_visualization(df, results)
    fig1.savefig('linear_regression_plots.png', dpi=300, bbox_inches='tight')
    print("Saved: linear_regression_plots.png")
    
    # Comparison plots
    fig2 = create_comparison_plot(results)
    fig2.savefig('regression_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: regression_metrics_comparison.png")
    
    # Save results to CSV
    results_df = save_results_to_csv(results, 'linear_regression_results.csv')
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Best R² Score: {results_df['activation_function'].iloc[results_df['r2_score'].idxmax()]} "
          f"(R² = {results_df['r2_score'].max():.4f})")
    print(f"Worst R² Score: {results_df['activation_function'].iloc[results_df['r2_score'].idxmin()]} "
          f"(R² = {results_df['r2_score'].min():.4f})")
    print(f"Highest Correlation: {results_df['activation_function'].iloc[results_df['correlation'].idxmax()]} "
          f"(r = {results_df['correlation'].max():.4f})")
    print(f"Steepest Slope: {results_df['activation_function'].iloc[results_df['slope'].idxmax()]} "
          f"(slope = {results_df['slope'].max():.2e})")
    print(f"Average R² Score: {results_df['r2_score'].mean():.4f}")
    
    plt.show()

if __name__ == "__main__":
    main()






