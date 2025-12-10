# SCALE-Sim Validation Framework

A comprehensive framework for validating and benchmarking neural network operations using JAX profiling and latency prediction models.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This validation framework provides tools to:
- Profile neural network operations (matrix multiplication, pooling, normalization, activation, elementwise)
- Compare predicted latencies against actual TPU execution times
- Generate detailed performance reports and visualizations
- Support multiple operation types with flexible configurations
- Benchmark on TPU by default (CPU also supported for testing)

## Main Entry Point

**`unified_model_verification.py`** is the primary script for running verification tests. It orchestrates validation across multiple operation types and generates comprehensive performance reports.

### Quick Start

```bash
# Run unified verification for all operation types
python unified_model_verification.py

# Run specific operation type verification
python unified_model_verification.py --matmul
python unified_model_verification.py --pooling
python unified_model_verification.py --elementwise
python unified_model_verification.py --normalization
python unified_model_verification.py --activation
```

### Available Options

| Command | Description |
|---------|-------------|
| *(no args)* | Run unified verification for all operation types |
| `--matmul` | Matrix multiplication verification (small-medium sizes) |
| `--large-matmul` | Large matrix multiplication verification (1K-4K dimensions) |
| `--pooling` | Max and average pooling operations |
| `--elementwise` | Element-wise operations (add, subtract, multiply, divide) |
| `--elementwise-3d` | 3D element-wise operations |
| `--normalization` | Layer norm, RMS norm, and batch normalization |
| `--activation` | Activation functions (ReLU, Sigmoid, Tanh, etc.) |
| `--help` | Show help message |

## Architecture

### Core Components

#### 1. **unified_model_verification.py**
The main entry point that orchestrates verification tests.

**Key Functions:**
- `run_unified_verification()`: Runs comprehensive tests across all operation types
- `matmul_unified_verification()`: Specialized matrix multiplication tests
- `pooling_unified_verification()`: Pooling operation tests
- `elementwise_unified_verification()`: Element-wise operation tests
- `normalization_unified_verification()`: Normalization operation tests
- `activation_unified_verification()`: Activation function tests

**Workflow:**
1. Creates verification directory for results
2. Initializes `ModelVerification` instance
3. Adds test configurations for various operations and shapes
4. Executes verification and collects results
5. Generates detailed analysis reports grouped by operation type

#### 2. **flexible_validation.py**
Low-level validation framework for profiling JAX kernels.

**Key Classes:**
- `ValidationConfig`: Defines kernel configuration (type, shapes, parameters)
- `ValidationPackage`: Handles single kernel profiling and trace parsing
- `ValidationManager`: Manages multiple validation packages

**Features:**
- JAX kernel compilation and profiling on TPU
- Trace event filtering and analysis
- StableHLO intermediate representation generation
- SCALE-Sim topology file generation

#### 3. **latency_prediction.py**
Implements latency prediction models for different operation types.

**Key Classes:**
- `PredictionManager`: Manages prediction configurations and generates predictions

**Supported Operations:**
- **Elementwise**: ADD, SUBTRACT, MULTIPLY, DIVIDE
- **Activation**: RELU, SIGMOID, TANH, LEAKY_RELU, ELU, SELU, LINEAR, BINARY
- **Normalization**: LAYER_NORM, RMS_NORM, BATCH_NORM
- **Pooling**: MAX_POOLING, AVG_POOLING
- **Matmul**: Matrix multiplication operations

**Prediction Strategy:**
- Uses linear models from `linear_models.py`
- Handles shape transformations (1D, 2D, 3D+)
- Applies dimension-specific corrections for edge cases

#### 4. **operation_classification.py**
Defines operation taxonomy and types.

**Enumerations:**
- `OperationType`: High-level operation categories
- `OperationElementwise`: Element-wise operations
- `OperationActivation`: Activation functions
- `OperationNormalization`: Normalization operations
- `OperationPooling`: Pooling operations
- `OperationMatmul`: Matrix multiplication types

#### 5. **kernel_functions.py**
Contains JAX implementations of validation kernels.

**Key Enumerations:**
- `KernelType`: Maps to specific JAX kernel implementations
- `ScaleSimTopologyType`: GEMM vs CONV topology types

**Example Kernels:**
- Matrix operations: `validation_matrix_multiply`, `validation_dot_product`
- Activations: `validation_relu`, `validation_sigmoid`, `validation_tanh`
- Normalizations: `validation_layer_norm`, `validation_batch_norm`, `validation_rms_norm`
- Pooling: `validation_max_pooling`, `validation_avg_pooling`

#### 6. **linear_models.py**
Pre-trained linear regression models for latency prediction.

Contains operation-specific models like:
- `linear_model_elementwise_add_1d(size)`
- `linear_model_activation_relu_2d(size)`
- `linear_model_matmul(m, n, k)`
- `linear_model_normalization_layer_norm_2d(size)`

#### 7. **kernel_configs.py**
Helper functions to generate `ValidationConfig` objects.

**Example Functions:**
```python
generate_matrix_multiply_config(name, M, N, K)
generate_layer_norm_config(name, shape, axis)
generate_max_pooling_config(name, shape, window_shape, strides, padding)
```

#### 8. **utils.py**
Utility classes and functions.

**Key Classes:**
- `DataFrameGenerator`: Flexible DataFrame construction with column alignment

#### 9. **trace_parser.py**
Parses JAX profiling traces.

**Key Class:**
- `TraceParser`: Extracts and processes profiling data from trace directories

## Output Files

After running verification, results are saved to the specified directory (e.g., `./unified_verification_results/`):

### Main Results
- **`merged_verification_results.csv`**: Complete verification results with columns:
  - `Operation_Type`: Category (elementwise, activation, matmul, etc.)
  - `Operation`: Specific operation (ADD, RELU, LINEAR, etc.)
  - `Input_Shapes`: Input tensor shapes
  - `Predicted_Latency_us`: Model prediction in microseconds
  - `Actual_Duration_us`: Measured GPU execution time
  - `Error_Percentage`: Prediction error percentage

### Profiling Data
- **`filtered_events_avg_fusion.csv`**: Average fusion kernel durations
  - `kernel_name`: Configuration name
  - `dur(us)`: Duration in microseconds

### Trace Data
Individual trace directories for each configuration containing:
- `trace_events.json`: Raw JAX profiling events
- `filtered_events.json`: Filtered relevant events

## Example Usage

### Basic Verification

```python
from unified_model_verification import run_unified_verification

# Run comprehensive verification
results = run_unified_verification()

# Results DataFrame includes:
# - Predicted vs Actual latencies
# - Error percentages
# - Operation metadata
```

### Custom Verification

```python
import operation_classification as oc
from model_verification import ModelVerification

# Create custom verification
verifier = ModelVerification(profile_dir="./my_results")

# Add custom configurations
verifier.add_verification_config(
    operation_type=oc.OperationType.MATMUL,
    operation=oc.OperationMatmul.LINEAR,
    shapes=[(512, 256), (256, 512)],
    operation_params={'M': 512, 'N': 512, 'K': 256}
)

# Run verification
results = verifier.verify()
```

### Using Validation Manager Directly

```python
from flexible_validation import ValidationManager, ValidationConfig
from kernel_functions import KernelType
import jax.numpy as jnp

# Create validation manager
manager = ValidationManager(profile_dir="./my_traces")

# Add configuration
config = ValidationConfig(
    name="my_matmul",
    kernel_type=KernelType.MATRIX_MULTIPLY,
    inputs=[((256, 128), jnp.float16), ((128, 256), jnp.float16)]
)
manager.add_config(config)

# Profile operations
manager.profile_all_packages(repeat=5)
manager.parse_all_packages()

# Get results
df = manager.get_filtered_events_dataframe_for_avg_fusion_duration()
```

## Performance Metrics

The framework calculates several performance metrics:

### Error Metrics
- **MAPE (Mean Absolute Percentage Error)**: Average absolute prediction error
- **RMSE (Root Mean Square Error)**: RMS of prediction errors in microseconds
- **Min/Max Error**: Range of prediction errors

### Analysis Reports

Example output from `run_unified_verification()`:

```
============================================================
DETAILED ANALYSIS BY OPERATION TYPE
============================================================
ADD          | Tests: 11 | MAPE:   8.23% | RMSE:   45.67 μs | Range: 2.1%-15.8%
RELU         | Tests:  9 | MAPE:   5.45% | RMSE:   32.18 μs | Range: 1.2%-12.3%
MATMUL       | Tests:  9 | MAPE:  12.67% | RMSE:  156.89 μs | Range: 3.4%-28.9%
LAYER_NORM   | Tests:  7 | MAPE:   7.89% | RMSE:   78.45 μs | Range: 2.8%-18.6%

============================================================
OVERALL STATISTICS
============================================================
Total test cases: 36
Overall MAPE: 8.56%
Overall RMSE: 78.30 μs

🏆 Best prediction:
   activation - relu
   Shape: [(512,)]
   Error: 1.23%

⚠️  Worst prediction:
   matmul - linear
   Shape: [(1024, 768), (768, 512)]
   Error: 28.92%
```

## Installation

### Dependencies

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Hardware Support

The framework is configured for **TPU** by default. The `requirements.txt` includes `jax[tpu]` for TPU support.

#### For CPU-Only

For CPU-only execution (testing/development):

```bash
pip install jax>=0.4.0
```

### Core Dependencies

- **JAX** (>=0.4.0): For TPU kernel execution and profiling
- **pandas** (>=1.5.0): Data manipulation and CSV output
- **numpy** (>=1.23.0): Numerical computations

### Python Version

- **Python 3.8+** required

## Advanced Features

### StableHLO Generation
```python
package = ValidationPackage(config)
package.setup_validation()
stablehlo_ir = package.get_stableHLO_string()
```

### SCALE-Sim Integration
```python
manager = ValidationManager()
# ... add configs ...
manager.write_scale_sim_topology_csv()
# Generates: scale_sim_gemm_topology.csv, scale_sim_conv_topology.csv
```

### Custom Trace Filtering
```python
package.parse_profile_trace()
filtered_events = package.filter_profile_trace_events(trace_events)
```

## Configuration Tips

### Shape Selection
- **1D shapes**: For vector operations (elementwise, activations)
- **2D shapes**: For matrices and 2D operations
- **3D shapes**: For normalization (batch, sequence, hidden)
- **4D shapes**: For pooling/convolution (batch, channels, height, width)

### Operation Parameters
Different operations support various parameters:
```python
# Layer Norm: axis parameter
operation_params={'axis': (2,)}

# Pooling: window and stride
operation_params={'window_shape': (2, 2), 'strides': (2, 2), 'padding': 'VALID'}
```

## Troubleshooting

### Common Issues

1. **JAX Not Found**: Ensure JAX is installed with TPU support
   ```bash
   pip install jax[tpu]>=0.4.0
   ```

2. **TPU Not Detected**: Verify TPU access and JAX installation
   ```python
   import jax
   print(jax.devices())  # Should show TPU devices
   ```

3. **Profile Directory Errors**: Directory is created automatically, but ensure write permissions

4. **Shape Mismatches**: Verify input shapes match operation requirements
   - Matmul: `(M, K)` and `(K, N)`
   - Pooling: `(N, C, H, W)` format

5. **Missing Results**: Check that `model_verification.py` is properly imported and available

## Contributing

When adding new operations:
1. Add kernel function to `kernel_functions.py`
2. Add enum to `operation_classification.py`
3. Implement prediction model in `latency_prediction.py`
4. Add linear model coefficients to `linear_models.py`
5. Create config generator in `kernel_configs.py`
6. Add test case to `unified_model_verification.py`

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Part of the [SCALE-Sim Project](https://github.com/scalesim-project/SCALE-Sim-validation).

