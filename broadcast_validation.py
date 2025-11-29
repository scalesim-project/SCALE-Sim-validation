import jax
import jax.numpy as jnp
import flexible_validation as fv
import kernel_configs as kc
import kernel_functions as kf


config_list = []
for base_size in [1, 2, 4, 8]:
    for cast_size in [128, 256, 512, 1024]:
        config_list.append(kc.generate_broadcast_to_dim_config(f"broadcast_base_{base_size}_cast_{cast_size}",(base_size,), (cast_size, base_size), (1,)))


manager = fv.ValidationManager(profile_dir="./traces/trace_broadcast_validation")

for config in config_list:
    manager.add_config(config)

manager.profile_all_packages(repeat = 1)
manager.parse_all_packages()
df = manager.get_filtered_events_dataframe(save_to_file=True)