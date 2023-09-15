import hls4ml
import tensorflow as tf
import numpy as np
from qresnet_and_mlp import qresnetModelWithLocalization
import keras
import os

# os.environ['PATH'] += os.pathsep + '/tools/Xilinx/Vitis_HLS/2023.1/bin'
os.environ['PATH'] += os.pathsep + "/tools/Xilinx/Vitis_HLS/2022.2/bin"

model = qresnetModelWithLocalization(2)

model.summary()

config = hls4ml.utils.config_from_keras_model(model, granularity="name")

print(config)

# Set the precision and reuse factor for the full model
config['Model']['Precision'] = 'ap_fixed<16,6>'
config['Model']['ReuseFactor'] = 1

for Layer in config['LayerName'].keys():
    config['LayerName'][Layer]['Strategy'] = 'Latency'
    config['LayerName'][Layer]['ReuseFactor'] = 1

# config['LayerName']['output_softmax']['Strategy'] = 'Stable'

# config['IOType'] = 'io_stream'  # Must set this if using CNNs!

# config["compiler"] = "vitis_hls"


print("-----------------------------------")
print("Configuration")
print(config)
print("-----------------------------------")
# part tested : xc7z030sbv485-3
# hls_model = hls4ml.converters.convert_from_keras_model(
#     model, hls_config=config, output_dir='model_1/hls4ml_prj', part='xcku035-sfva784-3-e', backend="Vitis"
# )

cfg = hls4ml.converters.create_config(backend='Vitis')
cfg['IOType'] = 'io_stream'  # Must set this if using CNNs!
cfg['HLSConfig'] = config
cfg['KerasModel'] = model
cfg['OutputDir'] = 'model_1/'
cfg['Part'] = 'xc7z030sbv485-3'

hls_model = hls4ml.converters.keras_to_hls(cfg)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

hls_model.compile()

# You can print the configuration to see some default parameters
print(config)

# Convert it to a hls project
# hls_model = hls4ml.converters.keras_to_hls(config, compiler="vitis_hls")

# Use Vivado HLS to synthesize the model
# This might take several minutes
hls_model.build(csim=False, synth=True)

# Print out the report if you want
hls4ml.report.read_vivado_report('model_1/hls4ml_prj/')