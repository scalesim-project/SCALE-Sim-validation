import jax
import jax.numpy as jnp
import flexible_validation as fv
import kernel_configs as kc
import kernel_functions as kf



shape_list = []
# for M in range(128, 8193, 128):
#             shape_list.append((M,))
# print(len(shape_list))

# for M in range(32, 1025,32):
#     for N in range(32, 1025, 32):
#         if M*N > 8192:
#             continue
#         shape_list.append((M, N))
# print(len(shape_list))

# for M in range(8, 256, 8):
#     for N in range(8, 256, 8):
#         for K in range(8, 256, 8):
#             if M*N*K > 8192:
#                 continue
#             shape_list.append((M, N, K))
# print(len(shape_list))


# 1D shapes
for m in range(128, 8193, 128):
    shape_list.append((m,))
print(len(shape_list))


# 2D shapes
for m in range(128, 1025, 128):
    for n in range(128, 1025, 128):
        shape_list.append((m, n))
print(len(shape_list))


config_list = []
for shape in shape_list:
    shape_str = str(shape).replace(" ", "")
    config_name = f"sub_{shape}"
    config_list.append(kc.generate_vector_op_config(config_name,kf.KernelType.VECTOR_SUB, shape))

for shape in shape_list:
    shape_str = str(shape).replace(" ", "")
    config_name = f"mul_{shape}"
    config_list.append(kc.generate_vector_op_config(config_name,kf.KernelType.VECTOR_MUL, shape))
    
for shape in shape_list:
    shape_str = str(shape).replace(" ", "")
    config_name = f"div_{shape}"
    config_list.append(kc.generate_vector_op_config(config_name,kf.KernelType.VECTOR_DIV, shape))

manager = fv.ValidationManager(profile_dir="./traces/trace_element_wise_repeat10")

for config in config_list:
    manager.add_config(config)

manager.profile_all_packages(repeat = 10)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)