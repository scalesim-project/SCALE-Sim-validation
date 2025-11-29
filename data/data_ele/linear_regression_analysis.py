#!/usr/bin/env python3
"""
Linear Regression Analysis for Kernel Statistics
Analyzes the relationship between tuple_product (product of dimensions) and avg_duration_us
Separates analysis for 1D and 2D shapes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import os

def load_and_analyze_data(file_path):
    """Load data and analyze shape distribution"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total kernels: {len(df)}")
    
    # Analyze shape distribution
    # 1D shapes: have dim_0 but dim_1 is NaN
    # 2D shapes: have both dim_0 and dim_1
    
    df_1d = df[df['dim_1'].isna()].copy()
    df_2d = df[df['dim_1'].notna()].copy()
    
    print(f"\n1D shapes: {len(df_1d)} kernels")
    print(f"2D shapes: {len(df_2d)} kernels")
    
    return df, df_1d, df_2d

def perform_linear_regression(df, shape_type, x_col='tuple_product', y_col='avg_duration_us'):
    """Perform linear regression analysis"""
    if len(df) == 0:
        print(f"No data available for {shape_type} shapes")
        return None
    
    print(f"\n=== Linear Regression Analysis for {shape_type} Shapes ===")
    
    # Extract data
    X = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values
    
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
    print(f"Number of data points: {len(df)}")
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
    
    # Linear equation
    print(f"Linear equation: y = {model.coef_[0]:.8f}x + {model.intercept_:.6f}")
    
    # Data range information
    print(f"{x_col} range: {X.min():.0f} to {X.max():.0f}")
    print(f"{y_col} range: {y.min():.6f} to {y.max():.6f}")
    
    return {
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
        'data': df
    }

def create_plots(results_1d, results_2d, output_dir='.'):
    """Create regression plots for both 1D and 2D shapes"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Regression Analysis: Tuple Product vs Average Duration', fontsize=16, fontweight='bold')
    
    # Plot 1D regression
    if results_1d is not None:
        ax1 = axes[0, 0]
        X_1d = results_1d['X'].flatten()
        y_1d = results_1d['y']
        y_pred_1d = results_1d['y_pred']
        
        # Scatter plot
        ax1.scatter(X_1d, y_1d, alpha=0.6, color='blue', s=50)
        ax1.plot(X_1d, y_pred_1d, color='red', linewidth=2, label=f'R² = {results_1d["r2"]:.4f}')
        
        ax1.set_xlabel('Tuple Product (dim_0)')
        ax1.set_ylabel('Average Duration (μs)')
        ax1.set_title('1D Shapes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot for 1D
        ax2 = axes[1, 0]
        residuals_1d = y_1d - y_pred_1d
        ax2.scatter(y_pred_1d, residuals_1d, alpha=0.6, color='blue', s=50)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_xlabel('Predicted Duration (μs)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('1D Shapes - Residuals')
        ax2.grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No 1D data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[1, 0].text(0.5, 0.5, 'No 1D data', ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Plot 2D regression
    if results_2d is not None:
        ax3 = axes[0, 1]
        X_2d = results_2d['X'].flatten()
        y_2d = results_2d['y']
        y_pred_2d = results_2d['y_pred']
        
        # Scatter plot
        ax3.scatter(X_2d, y_2d, alpha=0.6, color='green', s=50)
        ax3.plot(X_2d, y_pred_2d, color='red', linewidth=2, label=f'R² = {results_2d["r2"]:.4f}')
        
        ax3.set_xlabel('Tuple Product (dim_0 × dim_1)')
        ax3.set_ylabel('Average Duration (μs)')
        ax3.set_title('2D Shapes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Residuals plot for 2D
        ax4 = axes[1, 1]
        residuals_2d = y_2d - y_pred_2d
        ax4.scatter(y_pred_2d, residuals_2d, alpha=0.6, color='green', s=50)
        ax4.axhline(y=0, color='red', linestyle='--')
        ax4.set_xlabel('Predicted Duration (μs)')
        ax4.set_ylabel('Residuals')
        ax4.set_title('2D Shapes - Residuals')
        ax4.grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No 2D data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[1, 1].text(0.5, 0.5, 'No 2D data', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'linear_regression_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.show()

def create_detailed_analysis_plots(results_1d, results_2d, output_dir='.'):
    """Create additional detailed analysis plots"""
    
    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Analysis: Data Distributions and Correlations', fontsize=16, fontweight='bold')
    
    # 1D analysis
    if results_1d is not None:
        data_1d = results_1d['data']
        
        # Tuple product distribution
        axes[0, 0].hist(data_1d['tuple_product'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Tuple Product')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('1D: Tuple Product Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Duration distribution
        axes[0, 1].hist(data_1d['avg_duration_us'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_xlabel('Average Duration (μs)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('1D: Duration Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Log-log plot
        axes[0, 2].loglog(data_1d['tuple_product'], data_1d['avg_duration_us'], 'o', alpha=0.6, color='blue')
        axes[0, 2].set_xlabel('Tuple Product (log scale)')
        axes[0, 2].set_ylabel('Average Duration (log scale)')
        axes[0, 2].set_title('1D: Log-Log Plot')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 2D analysis
    if results_2d is not None:
        data_2d = results_2d['data']
        
        # Tuple product distribution
        axes[1, 0].hist(data_2d['tuple_product'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Tuple Product')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('2D: Tuple Product Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Duration distribution
        axes[1, 1].hist(data_2d['avg_duration_us'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Average Duration (μs)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('2D: Duration Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Log-log plot
        axes[1, 2].loglog(data_2d['tuple_product'], data_2d['avg_duration_us'], 'o', alpha=0.6, color='green')
        axes[1, 2].set_xlabel('Tuple Product (log scale)')
        axes[1, 2].set_ylabel('Average Duration (log scale)')
        axes[1, 2].set_title('2D: Log-Log Plot')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'detailed_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis plot saved to: {plot_path}")
    plt.show()

def generate_summary_report(results_1d, results_2d, output_dir='.'):
    """Generate a summary report of the analysis"""
    
    report_path = os.path.join(output_dir, 'regression_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Linear Regression Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Analysis: Relationship between Tuple Product and Average Duration\n")
        f.write("Dataset: kernel_statistics_filtered.csv\n\n")
        
        # 1D Results
        f.write("1D SHAPES ANALYSIS\n")
        f.write("-" * 30 + "\n")
        if results_1d is not None:
            f.write(f"Number of data points: {len(results_1d['data'])}\n")
            f.write(f"Slope: {results_1d['model'].coef_[0]:.8f}\n")
            f.write(f"Intercept: {results_1d['model'].intercept_:.6f}\n")
            f.write(f"R² Score: {results_1d['r2']:.6f}\n")
            f.write(f"RMSE: {results_1d['rmse']:.6f}\n")
            f.write(f"Pearson Correlation: {results_1d['correlation']:.6f}\n")
            f.write(f"P-value: {results_1d['p_value']:.2e}\n")
            f.write(f"Statistical significance: {results_1d['significance']}\n")
            f.write(f"Linear equation: y = {results_1d['model'].coef_[0]:.8f}x + {results_1d['model'].intercept_:.6f}\n")
        else:
            f.write("No 1D data available\n")
        
        f.write("\n2D SHAPES ANALYSIS\n")
        f.write("-" * 30 + "\n")
        if results_2d is not None:
            f.write(f"Number of data points: {len(results_2d['data'])}\n")
            f.write(f"Slope: {results_2d['model'].coef_[0]:.8f}\n")
            f.write(f"Intercept: {results_2d['model'].intercept_:.6f}\n")
            f.write(f"R² Score: {results_2d['r2']:.6f}\n")
            f.write(f"RMSE: {results_2d['rmse']:.6f}\n")
            f.write(f"Pearson Correlation: {results_2d['correlation']:.6f}\n")
            f.write(f"P-value: {results_2d['p_value']:.2e}\n")
            f.write(f"Statistical significance: {results_2d['significance']}\n")
            f.write(f"Linear equation: y = {results_2d['model'].coef_[0]:.8f}x + {results_2d['model'].intercept_:.6f}\n")
        else:
            f.write("No 2D data available\n")
        
        f.write("\nINTERPRETATION\n")
        f.write("-" * 30 + "\n")
        
        if results_1d is not None and results_2d is not None:
            if results_1d['r2'] > results_2d['r2']:
                f.write("1D shapes show stronger linear relationship than 2D shapes.\n")
            else:
                f.write("2D shapes show stronger linear relationship than 1D shapes.\n")
            
            f.write(f"1D slope: {results_1d['model'].coef_[0]:.8f} μs per unit product\n")
            f.write(f"2D slope: {results_2d['model'].coef_[0]:.8f} μs per unit product\n")
        
        f.write("\nFiles generated:\n")
        f.write("- linear_regression_analysis.png\n")
        f.write("- detailed_analysis.png\n")
        f.write("- regression_analysis_report.txt\n")
    
    print(f"\nSummary report saved to: {report_path}")

def main():
    # File path
    file_path = "kernel_statistics_filtered.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        print(f"Current directory: {os.getcwd()}")
        return
    
    try:
        # Load and analyze data
        df, df_1d, df_2d = load_and_analyze_data(file_path)
        
        # Perform linear regression for each shape type
        results_1d = perform_linear_regression(df_1d, "1D") if len(df_1d) > 0 else None
        results_2d = perform_linear_regression(df_2d, "2D") if len(df_2d) > 0 else None
        
        # Create visualizations
        create_plots(results_1d, results_2d)
        create_detailed_analysis_plots(results_1d, results_2d)
        
        # Generate summary report
        generate_summary_report(results_1d, results_2d)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        
        # Summary comparison
        if results_1d is not None and results_2d is not None:
            print(f"\nCOMPARISON SUMMARY:")
            print(f"1D Shapes: R² = {results_1d['r2']:.4f}, Slope = {results_1d['model'].coef_[0]:.8f}")
            print(f"2D Shapes: R² = {results_2d['r2']:.4f}, Slope = {results_2d['model'].coef_[0]:.8f}")
            
            if results_1d['r2'] > results_2d['r2']:
                print("1D shapes show stronger linear relationship.")
            else:
                print("2D shapes show stronger linear relationship.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
