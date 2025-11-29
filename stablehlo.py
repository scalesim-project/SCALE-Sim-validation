import jax
import jax.numpy as jnp
import flexible_validation as fv
import kernel_configs as kc
import kernel_functions as kf



config_list = []

config_list.append(kc.generate_matrix_multiply_config("matmul_64x64", 64, 64, 64))
config_list.append(kc.generate_dot_product_config("dot_product_64", 64))
config_list.append(kc.generate_convolve2d_config("convolve2d_64", 64))
config_list.append(kc.generate_vector_op_config("vector_op_64", kf.KernelType.VECTOR_ADD, (64,)))
config_list.append(kc.generate_vector_op_config("vector_op_64", kf.KernelType.VECTOR_MUL, (64,)))
config_list.append(kc.generate_activation_config("activation_64", kf.KernelType.RELU, (64,)))
config_list.append(kc.generate_activation_config("activation_64", kf.KernelType.SIGMOID, (64,)))
config_list.append(kc.generate_activation_config("activation_64", kf.KernelType.TANH, (64,)))
config_list.append(kc.generate_activation_config("activation_64", kf.KernelType.LEAKY_RELU, (64,)))
config_list.append(kc.generate_activation_config("activation_64", kf.KernelType.ELU, (64,)))
config_list.append(kc.generate_activation_config("activation_64", kf.KernelType.SELU, (64,)))
config_list.append(kc.generate_activation_config("activation_64", kf.KernelType.PARAMETRIC_RELU, (64,)))
config_list.append(kc.generate_activation_config("activation_64", kf.KernelType.BINARY_STEP, (64,)))
config_list.append(kc.generate_activation_config("activation_64", kf.KernelType.LINEAR, (64,)))
config_list.append(kc.generate_layer_norm_config("layer_norm_64", (2,128,512), (2,)))
config_list.append(kc.generate_rms_norm_config("rms_norm_64", (2,128,512), (2,)))
config_list.append(kc.generate_batch_norm_inference_config("batch_norm_inference_64", (64,64,32,32), 1))
config_list.append(kc.generate_max_pooling_config("max_pooling_64", (1, 64, 128, 128), (2, 2), (2, 2), "VALID"))
config_list.append(kc.generate_avg_pooling_config("avg_pooling_64", (1, 64, 128, 128), (2, 2), (2, 2), "VALID"))

manager = fv.ValidationManager(profile_dir="./traces/trace_stablehlo")

for config in config_list:
    manager.add_config(config)

manager.profile_all_packages(repeat = 1)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)
# stablehlo_strings = manager.get_all_stableHLO_strings()
# hlo_file = open("./traces/trace_stablehlo/stablehlo.txt", "w")
# for package in manager.packages:
#     hlo_file.write(f"=== {package.config.name} ===\n")
#     hlo_file.write(package.get_stableHLO_string())
#     hlo_file.write("\n")
# hlo_file.close()
