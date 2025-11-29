#!/usr/bin/env python3
"""
Linear Regression Analysis by Operation Type and Dimension Count
Analyzes the relationship between tuple_product and avg_duration_us
Separates analysis by:
1. Operation type (div, mul, sub)
2. Dimension count (1D, 2D)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import os
import re

def extract_operation_type(kernel_name):
    """Extract operation type from kernel name"""
    # Remove quotes and extract operation before first underscore
    clean_name = kernel_name.strip('"')
    
    # Extract operation (div, mul, sub, add, etc.)
    match = re.match(r'([a-zA-Z]+)_', clean_name)
    if match:
        return match.group(1)
    return 'unknown'

def load_and_categorize_data(file_path):
    """Load data and categorize by operation and dimension count"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total kernels: {len(df)}")
    
    # Extract operation types
    df['operation'] = df['kernel_name'].apply(extract_operation_type)
    
    # Determine dimension count
    df['dimension_count'] = df['dim_1'].apply(lambda x: '2D' if pd.notna(x) else '1D')
    
    # Create combined category
    df['op_dim_category'] = df['operation'] + '_' + df['dimension_count']
    
    print(f"\nOperation types found: {sorted(df['operation'].unique())}")
    print(f"Dimension types: {sorted(df['dimension_count'].unique())}")
    print(f"Combined categories: {sorted(df['op_dim_category'].unique())}")
    
    # Show distribution
    print(f"\nData distribution:")
    category_counts = df['op_dim_category'].value_counts().sort_index()
    for category, count in category_counts.items():
        print(f"  {category}: {count} kernels")
    
    return df

def perform_regression_by_category(df, category, x_col='tuple_product', y_col='avg_duration_us'):
    """Perform linear regression for a specific category"""
    
    category_data = df[df['op_dim_category'] == category].copy()
    
    if len(category_data) == 0:
        print(f"No data available for category: {category}")
        return None
    
    print(f"\n=== Linear Regression Analysis for {category} ===")
    
    # Extract data
    X = category_data[x_col].values.reshape(-1, 1)
    y = category_data[y_col].values
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Pearson correlation
    correlation, p_value = stats.pearsonr(X.flatten(), y)
    
    # Print results
    print(f"Number of data points: {len(category_data)}")
    print(f"Slope (coefficient): {model.coef_[0]:.8f}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Pearson Correlation: {correlation:.6f}")
    print(f"P-value: {p_value:.2e}")
    
    # Statistical significance
    if p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif p_value < 0.01:
        significance = "** (p < 0.01)"
    elif p_value < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "(not significant)"
    
    print(f"Statistical significance: {significance}")
    print(f"Linear equation: y = {model.coef_[0]:.8f}x + {model.intercept_:.6f}")
    
    # Data range information
    print(f"{x_col} range: {X.min():.0f} to {X.max():.0f}")
    print(f"{y_col} range: {y.min():.6f} to {y.max():.6f}")
    
    return {
        'category': category,
        'model': model,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'correlation': correlation,
        'p_value': p_value,
        'significance': significance,
        'data': category_data,
        'n_points': len(category_data)
    }

def create_comprehensive_plots(results_dict, output_dir='.'):
    """Create comprehensive plots for all categories"""
    
    # Get all categories and organize them
    categories = list(results_dict.keys())
    operations = sorted(set([cat.split('_')[0] for cat in categories]))
    dimensions = ['1D', '2D']
    
    # Set up color palette
    colors = {'div': 'blue', 'mul': 'green', 'sub': 'red'}
    
    # Create main regression plots
    fig, axes = plt.subplots(len(operations), 2, figsize=(15, 5 * len(operations)))
    if len(operations) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Linear Regression Analysis by Operation and Dimension', fontsize=16, fontweight='bold')
    
    for i, operation in enumerate(operations):
        for j, dimension in enumerate(dimensions):
            category = f"{operation}_{dimension}"
            ax = axes[i, j]
            
            if category in results_dict and results_dict[category] is not None:
                result = results_dict[category]
                X = result['X'].flatten()
                y = result['y']
                y_pred = result['y_pred']
                
                # Scatter plot with regression line
                ax.scatter(X, y, alpha=0.6, color=colors.get(operation, 'gray'), s=50)
                ax.plot(X, y_pred, color='red', linewidth=2, 
                       label=f'R² = {result["r2"]:.4f}')
                
                ax.set_xlabel('Tuple Product')
                ax.set_ylabel('Average Duration (μs)')
                ax.set_title(f'{operation.upper()} - {dimension}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add equation as text
                eq_text = f'y = {result["model"].coef_[0]:.2e}x + {result["model"].intercept_:.3f}'
                ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       verticalalignment='top')
            else:
                ax.text(0.5, 0.5, f'No data for\n{category}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, style='italic')
                ax.set_title(f'{operation.upper()} - {dimension}')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'operation_dimension_regression.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nMain regression plots saved to: {plot_path}")
    plt.show()

def create_comparison_plots(results_dict, output_dir='.'):
    """Create comparison plots across operations and dimensions"""
    
    # Filter out None results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison Analysis Across Operations and Dimensions', fontsize=16, fontweight='bold')
    
    # Extract data for comparison
    categories = list(valid_results.keys())
    r2_scores = [valid_results[cat]['r2'] for cat in categories]
    slopes = [valid_results[cat]['model'].coef_[0] for cat in categories]
    rmse_values = [valid_results[cat]['rmse'] for cat in categories]
    n_points = [valid_results[cat]['n_points'] for cat in categories]
    
    # Colors for different operations
    colors = []
    for cat in categories:
        op = cat.split('_')[0]
        if op == 'div':
            colors.append('blue')
        elif op == 'mul':
            colors.append('green')
        elif op == 'sub':
            colors.append('red')
        else:
            colors.append('gray')
    
    # R² comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(categories)), r2_scores, color=colors, alpha=0.7)
    ax1.set_xlabel('Operation-Dimension Category')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Fit Quality (R² Score)')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Slope comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(categories)), slopes, color=colors, alpha=0.7)
    ax2.set_xlabel('Operation-Dimension Category')
    ax2.set_ylabel('Slope (μs per unit product)')
    ax2.set_title('Performance Scaling (Slope)')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # RMSE comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(categories)), rmse_values, color=colors, alpha=0.7)
    ax3.set_xlabel('Operation-Dimension Category')
    ax3.set_ylabel('RMSE (μs)')
    ax3.set_title('Prediction Error (RMSE)')
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Sample size comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(categories)), n_points, color=colors, alpha=0.7)
    ax4.set_xlabel('Operation-Dimension Category')
    ax4.set_ylabel('Number of Data Points')
    ax4.set_title('Sample Size')
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, n_points):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'operation_dimension_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {plot_path}")
    plt.show()

def create_summary_table(results_dict, output_dir='.'):
    """Create a summary table of all results"""
    
    # Filter out None results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    
    if len(valid_results) == 0:
        print("No valid results for summary table")
        return
    
    # Create summary dataframe
    summary_data = []
    for category, result in valid_results.items():
        operation, dimension = category.split('_')
        summary_data.append({
            'Operation': operation,
            'Dimension': dimension,
            'Category': category,
            'N_Points': result['n_points'],
            'Slope': result['model'].coef_[0],
            'Intercept': result['model'].intercept_,
            'R²': result['r2'],
            'RMSE': result['rmse'],
            'Correlation': result['correlation'],
            'P_Value': result['p_value'],
            'Significance': result['significance']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['Operation', 'Dimension'])
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'operation_dimension_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to: {csv_path}")
    
    # Display table
    print(f"\nSUMMARY TABLE:")
    print("=" * 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(summary_df.round(6))
    
    return summary_df

def generate_detailed_report(results_dict, summary_df, output_dir='.'):
    """Generate detailed analysis report"""
    
    report_path = os.path.join(output_dir, 'operation_dimension_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Linear Regression Analysis by Operation and Dimension\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Analysis: Relationship between Tuple Product and Average Duration\n")
        f.write("Separated by Operation Type and Dimension Count\n")
        f.write("Dataset: kernel_statistics_filtered.csv\n\n")
        
        # Individual results
        valid_results = {k: v for k, v in results_dict.items() if v is not None}
        
        for category, result in sorted(valid_results.items()):
            operation, dimension = category.split('_')
            f.write(f"{category.upper()} ANALYSIS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Operation: {operation}\n")
            f.write(f"Dimension: {dimension}\n")
            f.write(f"Number of data points: {result['n_points']}\n")
            f.write(f"Slope: {result['model'].coef_[0]:.8f} μs per unit product\n")
            f.write(f"Intercept: {result['model'].intercept_:.6f} μs\n")
            f.write(f"R² Score: {result['r2']:.6f}\n")
            f.write(f"RMSE: {result['rmse']:.6f} μs\n")
            f.write(f"Pearson Correlation: {result['correlation']:.6f}\n")
            f.write(f"P-value: {result['p_value']:.2e}\n")
            f.write(f"Statistical significance: {result['significance']}\n")
            f.write(f"Linear equation: y = {result['model'].coef_[0]:.8f}x + {result['model'].intercept_:.6f}\n")
            f.write(f"Tuple product range: {result['X'].min():.0f} to {result['X'].max():.0f}\n")
            f.write(f"Duration range: {result['y'].min():.6f} to {result['y'].max():.6f} μs\n\n")
        
        # Comparative analysis
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("-" * 50 + "\n")
        
        if not summary_df.empty:
            # Best R² scores
            best_r2 = summary_df.loc[summary_df['R²'].idxmax()]
            f.write(f"Best model fit (highest R²): {best_r2['Category']} (R² = {best_r2['R²']:.6f})\n")
            
            # Steepest slopes
            steepest_slope = summary_df.loc[summary_df['Slope'].idxmax()]
            f.write(f"Steepest scaling (highest slope): {steepest_slope['Category']} (slope = {steepest_slope['Slope']:.8f})\n")
            
            # Compare by operation
            f.write("\nBy Operation:\n")
            for operation in summary_df['Operation'].unique():
                op_data = summary_df[summary_df['Operation'] == operation]
                f.write(f"  {operation.upper()}:\n")
                for _, row in op_data.iterrows():
                    f.write(f"    {row['Dimension']}: R² = {row['R²']:.4f}, Slope = {row['Slope']:.8f}\n")
            
            # Compare by dimension
            f.write("\nBy Dimension:\n")
            for dimension in summary_df['Dimension'].unique():
                dim_data = summary_df[summary_df['Dimension'] == dimension]
                f.write(f"  {dimension}:\n")
                avg_r2 = dim_data['R²'].mean()
                avg_slope = dim_data['Slope'].mean()
                f.write(f"    Average R²: {avg_r2:.4f}\n")
                f.write(f"    Average Slope: {avg_slope:.8f}\n")
        
        f.write(f"\nFiles generated:\n")
        f.write("- operation_dimension_regression.png\n")
        f.write("- operation_dimension_comparison.png\n")
        f.write("- operation_dimension_summary.csv\n")
        f.write("- operation_dimension_analysis_report.txt\n")
    
    print(f"Detailed report saved to: {report_path}")

def main():
    # File path
    file_path = "kernel_statistics_filtered.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        print(f"Current directory: {os.getcwd()}")
        return
    
    try:
        # Load and categorize data
        df = load_and_categorize_data(file_path)
        
        # Get all unique categories
        categories = sorted(df['op_dim_category'].unique())
        
        # Perform regression for each category
        results_dict = {}
        for category in categories:
            results_dict[category] = perform_regression_by_category(df, category)
        
        # Create visualizations
        create_comprehensive_plots(results_dict)
        create_comparison_plots(results_dict)
        
        # Create summary table
        summary_df = create_summary_table(results_dict)
        
        # Generate detailed report
        generate_detailed_report(results_dict, summary_df)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        
        # Print key findings
        valid_results = {k: v for k, v in results_dict.items() if v is not None}
        if valid_results:
            print(f"\nKEY FINDINGS:")
            print(f"Total categories analyzed: {len(valid_results)}")
            
            # Best and worst R² scores
            r2_scores = [(k, v['r2']) for k, v in valid_results.items()]
            best_r2 = max(r2_scores, key=lambda x: x[1])
            worst_r2 = min(r2_scores, key=lambda x: x[1])
            
            print(f"Best model fit: {best_r2[0]} (R² = {best_r2[1]:.4f})")
            print(f"Worst model fit: {worst_r2[0]} (R² = {worst_r2[1]:.4f})")
            
            # Compare slopes
            slopes = [(k, v['model'].coef_[0]) for k, v in valid_results.items()]
            steepest = max(slopes, key=lambda x: x[1])
            flattest = min(slopes, key=lambda x: x[1])
            
            print(f"Steepest scaling: {steepest[0]} (slope = {steepest[1]:.8f})")
            print(f"Flattest scaling: {flattest[0]} (slope = {flattest[1]:.8f})")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
