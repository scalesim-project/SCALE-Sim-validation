import jax
import jax.numpy as jnp
import flexible_validation as fv
import kernel_configs as kc
import kernel_functions as kf



shape_list = []
N = 1
for C in range(64, 267, 64):
    for H in range(128, 1025, 128):
        for W in range(128, 1025, 128):
            shape_list.append((N, C, H, W))







config_list = []

for shape in shape_list:
    shape_str = str(shape).replace(" ", "")
    config_name = f"poolaving_{shape_str}"
    config_list.append(kc.generate_avg_pooling_config(config_name, shape, window_shape = (2, 2), strides = (2, 2), padding = "VALID"))

manager = fv.ValidationManager(profile_dir="./traces/trace_layer_avg_pooling_repeat10")

for config in config_list:
    manager.add_config(config)

# manager.profile_all_packages(repeat = 3)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)