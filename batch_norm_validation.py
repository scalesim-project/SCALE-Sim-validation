import jax
import jax.numpy as jnp
import flexible_validation as fv
import kernel_configs as kc
import kernel_functions as kf



shape_list = []



# for N in [1, 2, 4, 8, 16, 32]:
#     for L in [128,256,512,1024,2048,4096]:
#         for H in [512,768,1024,2048,4096,8192]:
#             shape_list.append((N, L, H))

for N in range(64, 257, 64):
    for C in range(64, 257, 64):
        for H in range(32, 129, 32):
            for L in range(32, 129, 32):
                shape_list.append((N, C, H, L))

print(len(shape_list))



config_list = []

for shape in shape_list:
    shape_str = str(shape).replace(" ", "")
    config_name = f"batch_norm_inference_{shape_str}"
    config_list.append(kc.generate_batch_norm_inference_config(config_name, shape, axis = 1))


manager = fv.ValidationManager(profile_dir="./traces/trace_layer_batch_norm_repeat5")

for config in config_list:
    manager.add_config(config)

# manager.profile_all_packages(repeat = 5)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)