"""Benchmark case registry for NineToothed T1-2-1 evaluation.

Defines a collection of benchmark cases covering specialization-hit and
fallback scenarios (tile divisibility, contiguity, dtype, and
dimensionality).
"""

from benchmark.schema import BenchmarkCase

# fmt: off

# =========================================================================
# Specialization-hit cases  -- these SHOULD trigger a specialized variant
# =========================================================================

# 1D, size divisible by tile, contiguous, float32
DIVISIBLE_TILE_ADD_1024 = BenchmarkCase(
    name="divisible_tile_add_1024",
    description="1D element-wise add, size=1024 tile=256 (divisible, contiguous)",
    specialization_expected=True,
    input_shape=(1024,),
    dtype="float32",
    tile_size=256,
    func_name="add_divisible_1d",
)

# 1D, size divisible by tile, contiguous, float16
DIVISIBLE_TILE_ADD_4096_F16 = BenchmarkCase(
    name="divisible_tile_add_4096_f16",
    description="1D element-wise add, size=4096 tile=256 (divisible, contiguous, float16)",
    specialization_expected=True,
    input_shape=(4096,),
    dtype="float16",
    tile_size=256,
    func_name="add_divisible_f16",
)

# 2D, both dims divisible, contiguous, float32
DIVISIBLE_TILE_ADD_2D = BenchmarkCase(
    name="divisible_tile_add_2d",
    description="2D element-wise add, shape=(512,256) tile=(128,128) (divisible, contiguous)",
    specialization_expected=True,
    input_shape=(512, 256),
    dtype="float32",
    tile_size=128,
    func_name="add_divisible_2d",
)

# 1D, size divisible by tile, multiplication kernel, float32
DIVISIBLE_TILE_MUL_2048 = BenchmarkCase(
    name="divisible_tile_mul_2048",
    description="1D element-wise mul, size=2048 tile=256 (divisible, contiguous)",
    specialization_expected=True,
    input_shape=(2048,),
    dtype="float32",
    tile_size=256,
    func_name="mul_divisible_1d",
)

# =========================================================================
# Fallback cases  -- these should NOT trigger a specialized variant
# =========================================================================

# 1D, size NOT divisible by tile
NON_DIVISIBLE_TILE_ADD_1000 = BenchmarkCase(
    name="non_divisible_tile_add_1000",
    description="1D element-wise add, size=1000 tile=256 (non-divisible)",
    specialization_expected=False,
    input_shape=(1000,),
    dtype="float32",
    tile_size=256,
    func_name="add_non_divisible_1d",
)

# 1D, size not divisible by tile, float16
NON_DIVISIBLE_TILE_ADD_5000_F16 = BenchmarkCase(
    name="non_divisible_tile_add_5000_f16",
    description="1D element-wise add, size=5000 tile=256 (non-divisible, float16)",
    specialization_expected=False,
    input_shape=(5000,),
    dtype="float16",
    tile_size=256,
    func_name="add_non_divisible_f16",
)

# 2D, first dim not divisible, float32
NON_DIVISIBLE_TILE_ADD_2D = BenchmarkCase(
    name="non_divisible_tile_add_2d",
    description="2D element-wise add, shape=(513,128) tile=(128,128) (non-divisible first dim)",
    specialization_expected=False,
    input_shape=(513, 128),
    dtype="float32",
    tile_size=128,
    func_name="add_non_divisible_2d",
)

# 1D, size not divisible, multiplication kernel
NON_DIVISIBLE_TILE_MUL_1500 = BenchmarkCase(
    name="non_divisible_tile_mul_1500",
    description="1D element-wise mul, size=1500 tile=256 (non-divisible)",
    specialization_expected=False,
    input_shape=(1500,),
    dtype="float32",
    tile_size=256,
    func_name="mul_non_divisible_1d",
)

# -------------------------------------------------------------------------
# Grouped lists
# -------------------------------------------------------------------------

SPEC_HIT_CASES = [
    DIVISIBLE_TILE_ADD_1024,
    DIVISIBLE_TILE_ADD_4096_F16,
    DIVISIBLE_TILE_ADD_2D,
    DIVISIBLE_TILE_MUL_2048,
]

FALLBACK_CASES = [
    NON_DIVISIBLE_TILE_ADD_1000,
    NON_DIVISIBLE_TILE_ADD_5000_F16,
    NON_DIVISIBLE_TILE_ADD_2D,
    NON_DIVISIBLE_TILE_MUL_1500,
]

ALL_CASES = SPEC_HIT_CASES + FALLBACK_CASES

# So that ``from benchmark.cases import *`` is well-behaved.
__all__ = [
    "DIVISIBLE_TILE_ADD_1024",
    "DIVISIBLE_TILE_ADD_4096_F16",
    "DIVISIBLE_TILE_ADD_2D",
    "DIVISIBLE_TILE_MUL_2048",
    "NON_DIVISIBLE_TILE_ADD_1000",
    "NON_DIVISIBLE_TILE_ADD_5000_F16",
    "NON_DIVISIBLE_TILE_ADD_2D",
    "NON_DIVISIBLE_TILE_MUL_1500",
    "SPEC_HIT_CASES",
    "FALLBACK_CASES",
    "ALL_CASES",
]
