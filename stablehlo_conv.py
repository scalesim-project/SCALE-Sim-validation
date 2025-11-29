from calendar import c
import os
import jax
import jax.numpy as jnp
import jax.lax as lax


def original_convolve2d(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    return jax.scipy.signal.convolve2d(input_A, input_B)

def custom_convolve2d(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    """
    StableHLO-ish steps (generalized):
      1) reverse kernel on spatial dims (like @_flip)
      2) broadcast_in_dim to NCHW / OIHW
      3) convolution with padding = (Kh-1, Kw-1)  (i.e., 'full' conv)
      4) (optional) slice no-op
      5) reshape to 2D
    """
    if input_A.ndim != 2 or input_B.ndim != 2:
        raise ValueError("input_A and input_B must be rank-2 arrays")

    H, W   = input_A.shape
    Kh, Kw = input_B.shape

    # 1) reverse over spatial dims
    k_rev = lax.rev(input_B, dimensions=(0, 1))  # (Kh, Kw)

    # 2) broadcast_in_dim to NCHW / OIHW with C=1
    a_broadcast = lax.broadcast_in_dim(
        input_A, shape=(1, 1, H, W), broadcast_dimensions=(2, 3)
    )  # (1,1,H,W)

    k_broadcast = lax.broadcast_in_dim(
        k_rev, shape=(1, 1, Kh, Kw), broadcast_dimensions=(2, 3)
    )  # (1,1,Kh,Kw) as OIHW with O=1, I=1

    # 3) convolution: 'full' => padding = (Kh-1, Kw-1)
    pad_h = Kh - 1
    pad_w = Kw - 1
    y = lax.conv_general_dilated(
        lhs=a_broadcast,
        rhs=k_broadcast,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=1,
        batch_group_count=1,
        precision=None,
        preferred_element_type=None,
    )  # -> (1,1,H+Kh-1, W+Kw-1)

    # 4) explicit slice kept to mirror StableHLO structure (no-op here)
    out_h = H + Kh - 1
    out_w = W + Kw - 1
    y_sliced = lax.slice(y, (0, 0, 0, 0), (1, 1, out_h, out_w))

    # 5) reshape back to (H+Kh-1, W+Kw-1)
    return lax.reshape(y_sliced, (out_h, out_w))

def custom_convolve2d_step0(input_B: jnp.ndarray) -> jnp.ndarray:
    return lax.rev(input_B, dimensions=(0, 1))

def custom_convolve2d_step1(input_A: jnp.ndarray) -> jnp.ndarray:
    H, W = input_A.shape
    return lax.broadcast_in_dim(input_A, shape=(1, 1, H, W), broadcast_dimensions=(2, 3))

def custom_convolve2d_step2(input_B: jnp.ndarray) -> jnp.ndarray:
    Kh, Kw = input_B.shape
    return lax.broadcast_in_dim(input_B, shape=(1, 1, Kh, Kw), broadcast_dimensions=(2, 3))

def custom_convolve2d_step3(input_A: jnp.ndarray, input_B: jnp.ndarray) -> jnp.ndarray:
    _, _,Kh, Kw = input_B.shape
    pad_h = Kh - 1
    pad_w = Kw - 1
    return lax.conv_general_dilated(input_A, input_B, (1, 1), ((pad_h, pad_h), (pad_w, pad_w)), dimension_numbers=("NCHW", "OIHW", "NCHW"))

def custom_convolve2d_step4(input_A: jnp.ndarray) -> jnp.ndarray:
    _, _, out_h, out_w = input_A.shape
    return lax.slice(input_A, (0, 0, 0, 0), (1, 1, out_h, out_w))

def custom_convolve2d_step5(input_A: jnp.ndarray) -> jnp.ndarray:
    _, _, out_h, out_w = input_A.shape
    return lax.reshape(input_A, (out_h, out_w))


input_A: jnp.ndarray = jax.random.normal(jax.random.key(0), (64, 64), jnp.float16)
input_B: jnp.ndarray = jax.random.normal(jax.random.key(0), (3, 3), jnp.float16)

# original_output: jnp.ndarray = original_convolve2d(input_A, input_B)
# custom_output: jnp.ndarray = custom_convolve2d(input_A, input_B)

jit_original_conv_lowered = jax.jit(jax.named_call(original_convolve2d, name="my_original_convolve2d")).lower(jax.ShapeDtypeStruct(input_A.shape, input_A.dtype), jax.ShapeDtypeStruct(input_B.shape, input_B.dtype))
jit_original_conv_compiled = jit_original_conv_lowered.compile()
jit_custom_conv_lowered = jax.jit(jax.named_call(custom_convolve2d, name="my_custom_convolve2d")).lower(jax.ShapeDtypeStruct(input_A.shape, input_A.dtype), jax.ShapeDtypeStruct(input_B.shape, input_B.dtype))
jit_custom_conv_compiled = jit_custom_conv_lowered.compile()


# stablehlo_original_conv = jit_original_conv_lowered.as_text(dialect='stablehlo')
# stablehlo_custom_conv = jit_custom_conv_lowered.as_text(dialect='stablehlo')


dir_path = "./traces/trace_stablehlo_conv_compare"
# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)
# with open(os.path.join(dir_path, "stablehlo_original_conv.txt"), "w") as f:
#     f.write(stablehlo_original_conv)
# with open(os.path.join(dir_path, "stablehlo_custom_conv.txt"), "w") as f:
#     f.write(stablehlo_custom_conv)


step0_output: jnp.ndarray =  custom_convolve2d_step0(input_B).block_until_ready()
step1_output: jnp.ndarray = custom_convolve2d_step1(input_A).block_until_ready()
step2_output: jnp.ndarray = custom_convolve2d_step2(input_B).block_until_ready()
step3_output: jnp.ndarray = custom_convolve2d_step3(step1_output, step2_output).block_until_ready()
step4_output: jnp.ndarray = custom_convolve2d_step4(step3_output).block_until_ready()
step5_output: jnp.ndarray = custom_convolve2d_step5(step4_output).block_until_ready()

jit_step0_compiled = jax.jit(jax.named_call(custom_convolve2d_step0, name="my_convolve2d_step0")).lower(jax.ShapeDtypeStruct(input_B.shape, input_B.dtype)).compile()
jit_step1_compiled = jax.jit(jax.named_call(custom_convolve2d_step1, name="my_convolve2d_step1")).lower(jax.ShapeDtypeStruct(input_A.shape, input_A.dtype)).compile()
jit_step2_compiled = jax.jit(jax.named_call(custom_convolve2d_step2, name="my_convolve2d_step2")).lower(jax.ShapeDtypeStruct(step0_output.shape, step0_output.dtype)).compile()
jit_step3_compiled = jax.jit(jax.named_call(custom_convolve2d_step3, name="my_convolve2d_step3")).lower(jax.ShapeDtypeStruct(step1_output.shape, step1_output.dtype), jax.ShapeDtypeStruct(step2_output.shape, step2_output.dtype)).compile()
jit_step4_compiled = jax.jit(jax.named_call(custom_convolve2d_step4, name="my_convolve2d_step4")).lower(jax.ShapeDtypeStruct(step3_output.shape, step3_output.dtype)).compile()
jit_step5_compiled = jax.jit(jax.named_call(custom_convolve2d_step5, name="my_convolve2d_step5")).lower(jax.ShapeDtypeStruct(step4_output.shape, step4_output.dtype)).compile()



with jax.profiler.trace(dir_path):
    # jit_custom_conv_compiled(input_A, input_B).block_until_ready()
    # jit_original_conv_compiled(input_A, input_B).block_until_ready()

    step0_output = jit_step0_compiled(input_B).block_until_ready()
    step1_output = jit_step1_compiled(input_A).block_until_ready()
    step2_output = jit_step2_compiled(step0_output).block_until_ready()
    step3_output = jit_step3_compiled(step1_output, step2_output).block_until_ready()
    step4_output = jit_step4_compiled(step3_output).block_until_ready()
    step5_output = jit_step5_compiled(step4_output).block_until_ready()

