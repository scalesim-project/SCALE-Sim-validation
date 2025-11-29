from enum import Enum


class OperationType(Enum):
    ELEMENTWISE = "elementwise"
    MATMUL = "matmul"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    POOLING = "pooling"



class OperationBase(Enum):
    pass

class OperationElementwise(OperationBase):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

class OperationActivation(OperationBase):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky"  # Updated to match report naming
    ELU = "elu"
    SELU = "selu"
    PARAMETRIC_RELU = "parametric"  # Updated to match report naming
    LINEAR = "linear"
    BINARY = "binary"  # Added based on report

class OperationNormalization(OperationBase):
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer"  # Updated to match report naming
    RMS_NORM = "rms"  # Updated to match report naming
    INSTANCE_NORM = "instance_norm"
    GROUP_NORM = "group_norm"

class OperationPooling(OperationBase):
    MAX_POOLING = "max_pooling"
    AVG_POOLING = "avg_pooling"
    GLOBAL_MAX_POOLING = "global_max_pooling"
    GLOBAL_AVG_POOLING = "global_avg_pooling"
    POOLING_2D = "pooling_2d"
    POOLING_3D = "pooling_3d"

class OperationMatmul(OperationBase):
    LINEAR = "linear"
    BATCH_MATMUL = "batch_matmul"
    MATRIX_VECTOR = "matrix_vector"







