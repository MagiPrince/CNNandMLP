from model import customModelWithLocalization
from resnet_and_mlp import resnetModelWithLocalization
from qresnet_and_mlp import qresnetModelWithLocalization
import tensorflow as tf
from keras.optimizers import Adam

model = resnetModelWithLocalization(64)

def custom_loss(y_true, y_pred):
    # Coordinates loss
    coords_loss = tf.keras.losses.mean_squared_error(y_true[:, :, :2], y_pred[:, :, :2])
    
    # Confidence loss
    confidence_loss = tf.keras.losses.binary_crossentropy(y_true[:, :, 2:], y_pred[:, :, 2:])
    
    # Combine both losses
    return coords_loss + confidence_loss

model.compile(optimizer=Adam(learning_rate=0.0005), loss=custom_loss, metrics=['accuracy'])

model.summary()

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
######################################Calculate FLOPS##########################################
def get_flops(model):
    '''
    Calculate FLOPS
    Parameters
    ----------
    model : tf.keras.Model
        Model for calculating FLOPS.

    Returns
    -------
    flops.total_float_ops : int
        Calculated FLOPS for the model
    '''
    
    batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops

flops = get_flops(model)
print(f"FLOPS: {flops}")