from model import customModelWithLocalization
from resnet_and_mlp import resnetModelWithLocalization
from qresnet_and_mlp import qresnetModelWithLocalization
import tensorflow as tf
from keras.optimizers import Adam

model = qresnetModelWithLocalization(64)

def custom_loss(y_true, y_pred):
    # Coordinates loss
    coords_loss = tf.keras.losses.mean_squared_error(y_true[:, :, :2], y_pred[:, :, :2])
    
    # Confidence loss
    confidence_loss = tf.keras.losses.binary_crossentropy(y_true[:, :, 2:], y_pred[:, :, 2:])
    
    # Combine both losses
    return coords_loss + confidence_loss

model.compile(optimizer=Adam(learning_rate=0.0005), loss=custom_loss, metrics=['accuracy'])




model.summary()

