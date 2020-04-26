# run in TF 2

import tensorflow.compat.v1

graph_def_file = 'unknown_dim_reshape' # without '.pb'

input_arrays = ['Placeholder', 'Placeholder_1']
output_arrays = ['Relu']

converter = tensorflow.compat.v1.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file + ".pb", input_arrays, output_arrays)

converter.experimental_new_converter = True
converter.experimental_new_quantizer = True
converter.allow_custom_ops = True

tflite_model = converter.convert()

open(graph_def_file + ".tflite", "wb").write(tflite_model)

