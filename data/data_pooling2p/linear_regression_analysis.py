#!/usr/bin/env python3
"""
Linear regression analysis for total_elements vs avg_duration_us
from reduce-window kernel analysis data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import sys
from pathlib import Path

def perform_linear_regression_analysis(input_file='kernel_analysis.csv'):
    """
    Perform linear regression analysis on total_elements vs avg_duration_us
    """
    try:
        # Read the kernel analysis data
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        
        # Extract the variables for regression
        X = df['total_elements'].values.reshape(-1, 1)  # Independent variable
        y = df['avg_duration_us'].values  # Dependent variable
        
        print(f"Analyzing {len(df)} data points")
        print(f"Total elements range: {X.min():,} to {X.max():,}")
        print(f"Average duration range: {y.min():.3f} to {y.max():.3f} μs")
        
        # Perform linear regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate statistics
        r2 = r2_score(y, y_pred)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Calculate Pearson correlation coefficient
        correlation, p_value = stats.pearsonr(X.flatten(), y)
        
        # Calculate residuals
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        
        print(f"\nLinear Regression Results:")
        print(f"Equation: y = {slope:.6e}x + {intercept:.3f}")
        print(f"R-squared: {r2:.4f}")
        print(f"Correlation coefficient: {correlation:.4f}")
        print(f"P-value: {p_value:.2e}")
        print(f"RMSE: {rmse:.3f} μs")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Main scatter plot with regression line
        ax1.scatter(X, y, alpha=0.6, color='blue', s=50)
        ax1.plot(X, y_pred, color='red', linewidth=2, label=f'y = {slope:.2e}x + {intercept:.2f}')
        ax1.set_xlabel('Total Elements')
        ax1.set_ylabel('Average Duration (μs)')
        ax1.set_title(f'Linear Regression: Total Elements vs Average Duration\nR² = {r2:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Log-scale plot
        ax2.scatter(X, y, alpha=0.6, color='green', s=50)
        ax2.plot(X, y_pred, color='red', linewidth=2)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Total Elements (log scale)')
        ax2.set_ylabel('Average Duration (μs, log scale)')
        ax2.set_title('Log-Log Scale View')
        ax2.grid(True, alpha=0.3)
        
        # Residuals plot
        ax3.scatter(y_pred, residuals, alpha=0.6, color='purple', s=50)
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_xlabel('Predicted Duration (μs)')
        ax3.set_ylabel('Residuals (μs)')
        ax3.set_title(f'Residuals Plot\nRMSE = {rmse:.3f} μs')
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot for residuals normality check
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot of Residuals\n(Normality Check)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_plot = 'linear_regression_analysis.png'
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as {output_plot}")
        
        # Save detailed results to CSV
        results_df = pd.DataFrame({
            'total_elements': X.flatten(),
            'avg_duration_us': y,
            'predicted_duration_us': y_pred,
            'residuals': residuals
        })
        
        # Add kernel information for reference
        results_df = pd.concat([
            df[['kernel_name', 'batch_size', 'channels', 'height', 'width']],
            results_df
        ], axis=1)
        
        results_output = 'regression_results.csv'
        results_df.to_csv(results_output, index=False)
        print(f"Detailed results saved as {results_output}")
        
        # Create summary statistics
        summary_stats = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r2,
            'correlation': correlation,
            'p_value': p_value,
            'rmse': rmse,
            'n_points': len(df),
            'min_elements': int(X.min()),
            'max_elements': int(X.max()),
            'min_duration': float(y.min()),
            'max_duration': float(y.max())
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_output = 'regression_summary.csv'
        summary_df.to_csv(summary_output, index=False)
        print(f"Summary statistics saved as {summary_output}")
        
        plt.show()
        
        return model, summary_stats
        
    except Exception as e:
        print(f"Error performing analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "kernel_analysis.csv"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        sys.exit(1)
    
    # Run analysis
    model, stats = perform_linear_regression_analysis(str(input_file))




