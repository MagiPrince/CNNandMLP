import hls4ml
import tensorflow as tf
import numpy as np
from resnet_and_mlp import resnetModelWithLocalization
from qresnet_and_mlp import qresnetModelWithLocalization
import keras
import os

# os.environ['PATH'] += os.pathsep + '/tools/Xilinx/Vitis_HLS/2023.1/bin'
# os.environ['PATH'] += os.pathsep + "/tools/Xilinx/Vitis_HLS/2022.2/bin"
os.environ['PATH'] += os.pathsep + "/tools/Xilinx/Vitis_HLS/2022.1/bin"

model = qresnetModelWithLocalization(1)

model.summary()

config = hls4ml.utils.config_from_keras_model(model, granularity="name", default_reuse_factor=1)

# print(config)

# Set the precision and reuse factor for the full model
config['Model']['Precision'] = 'ap_fixed<16,6>'
# config['Model']['ReuseFactor'] = 1

for layer in config['LayerName'].keys():
    config['LayerName'][layer]['Strategy'] = 'latency'
    # config['LayerName'][layer]['ReuseFactor'] = 1

for layer in model.layers:
    if layer.__class__.__name__ in ['QConv2D', 'QDense']:
        w = layer.get_weights()[0]
        layersize = np.prod(w.shape)
        # print("{}: {}".format(layer.name, layersize))  # 0 = weights, 1 = biases
        if layersize > 4096:  # assuming that shape[0] is batch, i.e., 'None'
            print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name, layersize))
            # config['LayerName'][layer.name]['Strategy'] = 'resource'
    # elif layer.__class__.__name__ in ["Flatten", "Concatenate"]:
    #         print(layer.name)
    #         config['LayerName'][layer.name]['Strategy'] = 'resource'

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
cfg['Part'] = 'xcu250-figd2104-2L-e'
# cfg['Part'] = 'xczu7cg-fbvb900-1-e'
# cfg['Part'] = 'xc7z030sbv485-3' #Zynq-7000

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