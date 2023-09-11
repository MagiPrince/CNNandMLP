import hls4ml
import tensorflow as tf
import numpy as np
from resnet_and_mlp import resnetModelWithLocalization
import keras

model = resnetModelWithLocalization(30)

model.summary()

config = hls4ml.utils.config_from_keras_model(model)

print(config)

config["Model"]["ReuseFactor"] = 1000000

# config["compiler"] = "vitis_hls"


print("-----------------------------------")
print("Configuration")
print(config)
print("-----------------------------------")
# part tested : xc7z030sbv485-3
hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=config, output_dir='model_1/hls4ml_prj', part='xcku035-sfva784-3-e', backend="Vitis"
)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

hls_model.compile()

# You can print the configuration to see some default parameters
print(config)

# Convert it to a hls project
# hls_model = hls4ml.converters.keras_to_hls(config, compiler="vitis_hls")

# Use Vivado HLS to synthesize the model
# This might take several minutes
hls_model.build()

# Print out the report if you want
hls4ml.report.read_vivado_report('my-hls-test')