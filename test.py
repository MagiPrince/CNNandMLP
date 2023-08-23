import hls4ml
from model import customModelWithLocalization

model = customModelWithLocalization(10)

print(model.summary())

config = hls4ml.utils.config_from_keras_model(model)

config['Model']['ReuseFactor'] = 1000000

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=config, output_dir='model_1/hls4ml_prj', part='xcu250-figd2104-2L-e'#, io_type="io_serial"
)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

hls_model.compile()

hls_model.build(csim=False)

hls4ml.report.read_vivado_report('model_1/hls4ml_prj/')