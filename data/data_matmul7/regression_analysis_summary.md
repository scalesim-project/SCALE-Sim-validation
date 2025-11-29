# Scale Sim Model Regression Analysis Summary

## Overview
This analysis adds a `scale_sim_cycles` column to the fusion statistics report based on the `matmul_scale_sim_model` function and performs comprehensive linear regression analysis.

## Data Summary
- **Total samples**: 343 matrix multiplication configurations
- **Scale Sim Cycles range**: 89,984 to 4,585,472 cycles
- **Fusion Average range**: 43.39ms to 966.63ms

## Main Regression Results

### Overall Model Performance
- **Linear Model Equation**: `fusion_avg = 0.00020227 * scale_sim_cycles + 29.721747`
- **R² Score**: 0.9786 (97.86% variance explained)
- **Root Mean Squared Error**: 24.98ms
- **Mean Absolute Error**: 18.17ms

### Key Findings
1. **Excellent correlation**: The scale sim model shows very strong predictive power with R² = 0.9786
2. **Linear relationship**: The relationship between scale sim cycles and fusion time is highly linear
3. **Consistent performance**: The model works well across different matrix dimensions

## Dimension-Specific Analysis

### By M Dimension (Rows)
| M Value | R² Score | Slope | Intercept |
|---------|----------|-------|-----------|
| 1024    | 0.9576   | 0.000214 | 17.09 |
| 1536    | 0.9388   | 0.000198 | 28.59 |
| 2048    | 0.9580   | 0.000216 | 22.19 |
| 2560    | 0.9762   | 0.000205 | 26.21 |
| 3072    | 0.9625   | 0.000199 | 36.26 |
| 3584    | 0.9678   | 0.000196 | 41.42 |
| 4096    | 0.9881   | 0.000197 | 40.76 |

### By N Dimension (Columns)
| N Value | R² Score | Slope | Intercept |
|---------|----------|-------|-----------|
| 1024    | 0.9204   | 0.000186 | 27.43 |
| 1536    | 0.9441   | 0.000179 | 36.71 |
| 2048    | 0.9842   | 0.000197 | 27.66 |
| 2560    | 0.9815   | 0.000192 | 36.79 |
| 3072    | 0.9833   | 0.000190 | 46.84 |
| 3584    | 0.9729   | 0.000188 | 59.26 |
| 4096    | 0.9824   | 0.000202 | 54.92 |

### By K Dimension (Inner Dimension)
| K Value | R² Score | Slope | Intercept |
|---------|----------|-------|-----------|
| 1024    | 0.9838   | 0.000230 | 21.59 |
| 1536    | 0.9605   | 0.000253 | 11.96 |
| 2048    | 0.9673   | 0.000221 | 28.96 |
| 2560    | 0.9755   | 0.000214 | 18.30 |
| 3072    | 0.9936   | 0.000207 | 10.81 |
| 3584    | 0.9904   | 0.000209 | 5.95 |
| 4096    | 0.9855   | 0.000207 | 6.70 |

## Correlation Analysis

The correlation matrix reveals:
- **Scale Sim Cycles ↔ Fusion Average**: 0.9893 (very strong positive correlation)
- **M dimension ↔ Fusion Average**: 0.5357 (moderate positive correlation)
- **N dimension ↔ Fusion Average**: 0.5810 (moderate positive correlation)
- **K dimension ↔ Fusion Average**: 0.4800 (moderate positive correlation)

## Key Insights

1. **Model Validation**: The scale sim model is highly effective at predicting actual execution times with 97.86% accuracy.

2. **Consistent Slope**: The slope values across different dimensions are remarkably consistent (~0.0002), suggesting the model captures the fundamental timing relationship well.

3. **Dimension Impact**: 
   - N dimension shows slightly higher correlation with execution time than M or K
   - K dimension variations show the most consistent R² scores (all > 0.96)

4. **Intercept Variation**: The intercept values vary by dimension, suggesting some fixed overhead that depends on matrix dimensions beyond the scale sim cycles.

## Practical Applications

This analysis validates that the `matmul_scale_sim_model` function can be used to:
- Predict matrix multiplication execution times with high accuracy
- Compare performance across different matrix configurations
- Optimize matrix multiplication workloads by understanding cycle requirements

## Files Generated

1. `fusion_statistics_report_with_scale_sim.csv` - Original data with added scale sim cycles column
2. `scale_sim_regression_analysis.png` - Comprehensive visualization plots
3. `add_scale_sim_analysis.py` - Analysis script for reproducibility
4. `regression_analysis_summary.md` - This summary document



