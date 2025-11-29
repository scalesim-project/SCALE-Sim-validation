#!/usr/bin/env python3
"""
Verification script using existing kernel validation data.

This script uses the ModelVerification class to verify predictions against
kernels that already have validation data in the data/ directory.
"""

import os
import sys
sys.path.append('.')
import operation_classification as oc
from model_verification import ModelVerification

def verify_add_kernels():
    """Verify ADD operation kernels based on existing validation data."""
    
    print("=== Verifying ADD Kernels ===\n")
    
    # Create verification directory
    verification_dir = "./verification_add_kernels"
    os.makedirs(verification_dir, exist_ok=True)
    
    # Initialize model verification
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    # Add 1D ADD operations (based on data_add_1d_1)
    print("Adding 1D ADD operations...")
    add_1d_shapes = [
        (1024,), (2048,), (4096,), (8192,), (16384,), (32768,)
    ]
    
    for shape in add_1d_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.ELEMENTWISE,
            operation=oc.OperationElementwise.ADD,
            shapes=[shape],
            operation_params={}
        )
    
    # Add 2D ADD operations (based on data_add_2d_1)
    print("Adding 2D ADD operations...")
    add_2d_shapes = [
        (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)
    ]
    
    for shape in add_2d_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.ELEMENTWISE,
            operation=oc.OperationElementwise.ADD,
            shapes=[shape],
            operation_params={}
        )
    
    print(f"Total ADD configurations: {len(model_verifier.prediction_manager.config_list)}")
    
    # Run verification
    print("\nRunning ADD kernel verification...")
    try:
        results = model_verifier.verify()
        print(f"\nADD verification completed!")
        print(f"Results saved to: {verification_dir}/merged_verification_results.csv")
        return results
    except Exception as e:
        print(f"Error during ADD verification: {e}")
        return None

def verify_matmul_kernels():
    """Verify Matrix Multiplication kernels based on existing validation data."""
    
    print("\n=== Verifying Matrix Multiplication Kernels ===\n")
    
    # Create verification directory
    verification_dir = "./verification_matmul_kernels"
    os.makedirs(verification_dir, exist_ok=True)
    
    # Initialize model verification
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    # Matrix multiplication configurations (based on data_matmul_linear)
    print("Adding Matrix Multiplication operations...")
    matmul_configs = [
        # (M, N, K) configurations
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
        (1024, 512, 256),
        (512, 1024, 512),
    ]
    
    for M, N, K in matmul_configs:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.MATMUL,
            operation=oc.OperationMatmul.LINEAR,
            shapes=[(M, K), (K, N)],
            operation_params={'M': M, 'N': N, 'K': K}
        )
    
    print(f"Total MATMUL configurations: {len(model_verifier.prediction_manager.config_list)}")
    
    # Run verification
    print("\nRunning Matrix Multiplication kernel verification...")
    try:
        results = model_verifier.verify()
        print(f"\nMATMUL verification completed!")
        print(f"Results saved to: {verification_dir}/merged_verification_results.csv")
        return results
    except Exception as e:
        print(f"Error during MATMUL verification: {e}")
        return None

def verify_activation_kernels():
    """Verify Activation function kernels."""
    
    print("\n=== Verifying Activation Kernels ===\n")
    
    # Create verification directory
    verification_dir = "./verification_activation_kernels"
    os.makedirs(verification_dir, exist_ok=True)
    
    # Initialize model verification
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    # RELU activation configurations (based on data_relu1, data_act1)
    print("Adding RELU activation operations...")
    relu_shapes = [
        (512,), (1024,), (2048,), (4096,), (8192,),
        (64, 64), (128, 128), (256, 256), (512, 512)
    ]
    
    for shape in relu_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.ACTIVATION,
            operation=oc.OperationActivation.RELU,
            shapes=[shape],
            operation_params={}
        )
    
    print(f"Total RELU configurations: {len(model_verifier.prediction_manager.config_list)}")
    
    # Run verification
    print("\nRunning RELU kernel verification...")
    try:
        results = model_verifier.verify()
        print(f"\nRELU verification completed!")
        print(f"Results saved to: {verification_dir}/merged_verification_results.csv")
        return results
    except Exception as e:
        print(f"Error during RELU verification: {e}")
        return None

def verify_normalization_kernels():
    """Verify Layer Normalization kernels."""
    
    print("\n=== Verifying Layer Normalization Kernels ===\n")
    
    # Create verification directory
    verification_dir = "./verification_layernorm_kernels"
    os.makedirs(verification_dir, exist_ok=True)
    
    # Initialize model verification
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    # Layer normalization configurations (based on data_layer_norm1, data_layer_norm2)
    print("Adding Layer Normalization operations...")
    layernorm_shapes = [
        (128, 256), (256, 512), (512, 1024), (1024, 2048),
        (64, 128), (256, 256), (512, 512)
    ]
    
    for shape in layernorm_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.NORMALIZATION,
            operation=oc.OperationNormalization.LAYER_NORM,
            shapes=[shape],
            operation_params={'axis': (-1,)}
        )
    
    print(f"Total Layer Norm configurations: {len(model_verifier.prediction_manager.config_list)}")
    
    # Run verification
    print("\nRunning Layer Normalization kernel verification...")
    try:
        results = model_verifier.verify()
        print(f"\nLayer Norm verification completed!")
        print(f"Results saved to: {verification_dir}/merged_verification_results.csv")
        return results
    except Exception as e:
        print(f"Error during Layer Norm verification: {e}")
        return None

def run_comprehensive_verification():
    """Run verification for all kernel types."""
    
    print("=" * 60)
    print("COMPREHENSIVE MODEL VERIFICATION")
    print("=" * 60)
    
    all_results = {}
    
    # Verify each kernel type
    verification_functions = [
        ("ADD", verify_add_kernels),
        ("MATMUL", verify_matmul_kernels),
        ("RELU", verify_activation_kernels),
        ("LAYER_NORM", verify_normalization_kernels),
    ]
    
    for name, verify_func in verification_functions:
        try:
            result = verify_func()
            if result is not None:
                all_results[name] = result
                print(f"✅ {name} verification successful")
            else:
                print(f"❌ {name} verification failed")
        except Exception as e:
            print(f"❌ {name} verification failed with error: {e}")
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL VERIFICATION SUMMARY")
    print("=" * 60)
    
    for name, result in all_results.items():
        if result is not None:
            mape = result['Error_Percentage'].abs().mean()
            rmse = ((result['Predicted_Latency_us'] - result['Actual_Duration_us']) ** 2).mean() ** 0.5
            print(f"{name:12} | MAPE: {mape:6.2f}% | RMSE: {rmse:8.2f} μs | Tests: {len(result):3d}")
    
    return all_results

def run_quick_test():
    """Run a quick test with just a few kernels."""
    
    print("=== Quick Model Verification Test ===\n")
    
    verification_dir = "./quick_verification_test"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    # Just a few test cases
    quick_configs = [
        # ADD operations
        (oc.OperationType.ELEMENTWISE, oc.OperationElementwise.ADD, [(1024,)], {}),
        (oc.OperationType.ELEMENTWISE, oc.OperationElementwise.ADD, [(128, 128)], {}),
        
        # RELU activation
        (oc.OperationType.ACTIVATION, oc.OperationActivation.RELU, [(2048,)], {}),
        
        # Small matrix multiplication
        (oc.OperationType.MATMUL, oc.OperationMatmul.LINEAR, [(128, 256), (256, 512)], 
         {'M': 128, 'N': 512, 'K': 256}),
    ]
    
    print("Adding quick test configurations...")
    for op_type, operation, shapes, params in quick_configs:
        model_verifier.add_verification_config(op_type, operation, shapes, params)
    
    print(f"Running quick verification with {len(quick_configs)} test cases...")
    
    try:
        results = model_verifier.verify()
        print(f"\nQuick verification completed!")
        print(f"Results saved to: {verification_dir}/merged_verification_results.csv")
        return results
    except Exception as e:
        import traceback
        print(f"Error during quick verification: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            run_quick_test()
        elif sys.argv[1] == "--add":
            verify_add_kernels()
        elif sys.argv[1] == "--matmul":
            verify_matmul_kernels()
        elif sys.argv[1] == "--relu":
            verify_activation_kernels()
        elif sys.argv[1] == "--layernorm":
            verify_normalization_kernels()
        else:
            print("Usage: python verify_existing_kernels.py [--quick|--add|--matmul|--relu|--layernorm]")
    else:
        # Run comprehensive verification by default
        run_comprehensive_verification()
