#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
import sys
import os

# Add the parent directory to sys.path to import linear_models
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
try:
    from linear_models import matmul_scale_sim_model
except ImportError:
    # If import fails, define the function directly
    def matmul_scale_sim_model(m: int, n: int, k: int, systolic_array_size: int = 128) -> int:
        v1 = (2*systolic_array_size + systolic_array_size + m - 2) * math.ceil(n / systolic_array_size) * math.ceil(k / systolic_array_size)
        m, n = n, m
        v2 = (2*systolic_array_size + systolic_array_size + m - 2) * math.ceil(n / systolic_array_size) * math.ceil(k / systolic_array_size)
        return min(v1, v2)

def add_scale_sim_column(df):
    """Add scale sim model column to the dataframe"""
    scale_sim_values = []
    
    for _, row in df.iterrows():
        m, n, k = int(row['dim_m']), int(row['dim_n']), int(row['dim_k'])
        scale_sim_value = matmul_scale_sim_model(m, n, k)
        scale_sim_values.append(scale_sim_value)
    
    df['scale_sim_cycles'] = scale_sim_values
    return df

def perform_regression_analysis(df):
    """Perform linear regression analysis"""
    # Features and target
    X = df[['scale_sim_cycles']].values
    y = df['fusion_avg'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Print results
    print("=== Linear Regression Results ===")
    print(f"Coefficient (slope): {model.coef_[0]:.8f}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"\nLinear Model Equation:")
    print(f"fusion_avg = {model.coef_[0]:.8f} * scale_sim_cycles + {model.intercept_:.6f}")
    
    return model, y_pred, r2, mse, mae, rmse

def create_visualizations(df, model, y_pred):
    """Create visualization plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot with regression line
    ax1 = axes[0, 0]
    ax1.scatter(df['scale_sim_cycles'], df['fusion_avg'], alpha=0.6, s=50)
    ax1.plot(df['scale_sim_cycles'], y_pred, 'r-', linewidth=2, label='Regression Line')
    ax1.set_xlabel('Scale Sim Cycles')
    ax1.set_ylabel('Fusion Average (ms)')
    ax1.set_title('Scale Sim Cycles vs Fusion Average')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual plot
    ax2 = axes[0, 1]
    residuals = df['fusion_avg'] - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Actual vs Predicted
    ax3 = axes[1, 0]
    min_val = min(df['fusion_avg'].min(), y_pred.min())
    max_val = max(df['fusion_avg'].max(), y_pred.max())
    ax3.scatter(df['fusion_avg'], y_pred, alpha=0.6, s=50)
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title('Actual vs Predicted Values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution of residuals
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scale_sim_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_by_dimension_groups(df):
    """Analyze regression by different dimension groups"""
    print("\n=== Analysis by Dimension Groups ===")
    
    # Group by M dimension
    m_groups = df.groupby('dim_m')
    print("\nAnalysis by M dimension:")
    for m_val, group in m_groups:
        if len(group) >= 5:  # Only analyze groups with sufficient data
            X = group[['scale_sim_cycles']].values
            y = group['fusion_avg'].values
            model = LinearRegression()
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))
            print(f"M={m_val}: R²={r2:.4f}, Slope={model.coef_[0]:.8f}, Intercept={model.intercept_:.4f}")
    
    # Group by N dimension
    n_groups = df.groupby('dim_n')
    print("\nAnalysis by N dimension:")
    for n_val, group in n_groups:
        if len(group) >= 5:
            X = group[['scale_sim_cycles']].values
            y = group['fusion_avg'].values
            model = LinearRegression()
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))
            print(f"N={n_val}: R²={r2:.4f}, Slope={model.coef_[0]:.8f}, Intercept={model.intercept_:.4f}")
    
    # Group by K dimension
    k_groups = df.groupby('dim_k')
    print("\nAnalysis by K dimension:")
    for k_val, group in k_groups:
        if len(group) >= 5:
            X = group[['scale_sim_cycles']].values
            y = group['fusion_avg'].values
            model = LinearRegression()
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))
            print(f"K={k_val}: R²={r2:.4f}, Slope={model.coef_[0]:.8f}, Intercept={model.intercept_:.4f}")

def main():
    # Load the CSV file
    csv_path = 'fusion_statistics_report.csv'
    print(f"Loading data from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        print("\nOriginal columns:", list(df.columns))
        
        # Add scale sim column
        print("\nAdding scale sim cycles column...")
        df = add_scale_sim_column(df)
        print("Scale sim column added successfully!")
        
        # Display basic statistics
        print("\n=== Basic Statistics ===")
        print(f"Scale Sim Cycles - Min: {df['scale_sim_cycles'].min()}, Max: {df['scale_sim_cycles'].max()}")
        print(f"Fusion Average - Min: {df['fusion_avg'].min():.6f}, Max: {df['fusion_avg'].max():.6f}")
        
        # Save updated CSV
        output_path = 'fusion_statistics_report_with_scale_sim.csv'
        df.to_csv(output_path, index=False)
        print(f"\nUpdated CSV saved as: {output_path}")
        
        # Perform regression analysis
        print("\nPerforming regression analysis...")
        model, y_pred, r2, mse, mae, rmse = perform_regression_analysis(df)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(df, model, y_pred)
        
        # Analyze by dimension groups
        analyze_by_dimension_groups(df)
        
        # Show correlation matrix
        print("\n=== Correlation Analysis ===")
        correlation_cols = ['dim_m', 'dim_n', 'dim_k', 'scale_sim_cycles', 'fusion_avg']
        corr_matrix = df[correlation_cols].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(4))
        
        # Display sample of data with new column
        print("\n=== Sample Data with Scale Sim Column ===")
        sample_cols = ['kernel_name', 'dim_m', 'dim_n', 'dim_k', 'scale_sim_cycles', 'fusion_avg']
        print(df[sample_cols].head(10))
        
        return df, model
        
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found!")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    df, model = main()
