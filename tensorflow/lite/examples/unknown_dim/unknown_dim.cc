/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
   Copyright 2020 Hyun Sik Yoon. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// Usage: minimal <tflite model>

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "unknown_dim <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  int base_index = 0;

  // This is [-1, -1] input
  // If this is not called, internal shape is set to scalar and error happens.
  {
    TfLiteQuantizationParams quant;
    interpreter->SetTensorParametersReadWrite(
        0, kTfLiteFloat32, "Placeholder", {2, 3}, quant);
  }
  /*
  These lines are not needed to be called
  interpreter->SetTensorParametersReadWrite(
      1, kTfLiteInt32, "Placeholder_1", {2}, quant);

  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "Relu", {}, quant);
  */

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  printf("=== Input value => ");
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < 6; i++) {
    input[i] = i-3;
    printf("%f, ", input[i]);
  }
  printf("\n");

  // fill new_sizes
  printf("=== new shape of RESHAPE op => ");
  auto new_shape_input = interpreter->typed_tensor<int>(1);
  new_shape_input[0] = 1;
  new_shape_input[1] = 6;
  printf("%d, %d\n", new_shape_input[0], new_shape_input[1]);

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // output shape
  int tensor_ind = 0;
  printf("=== tensor list:\n");
  for(auto &tensor: interpreter->primary_subgraph().tensors())
  {
    printf("\t - tensor #%d: [", tensor_ind++);
    auto dims = tensor.dims;
    for (int i = 0; i < dims->size; i++)
      printf("%d,", dims->data[i]);
    printf("]\n");
  }

  // Read output buffers
  printf("=== Output value => ");
  auto output = interpreter->typed_tensor<float>(2);
  for (int i = 0; i < 6; i++) {
    printf("%f, ", output[i]);
  }
  printf("\n\nDone.");

  return 0;
}

void PrintTfLiteIntVector(const TfLiteIntArray* v) {
  if (!v) {
    printf(" (null)\n");
    return;
  }
  for (int k = 0; k < v->size; k++) {
    printf(" %d", v->data[k]);
  }
  printf("\n");
}

void print_op_tensor(Interpreter *interpreter)
{
  // TODO(@hyunsik.yoon) need to modify
  for (size_t node_index = 0; node_index < interpreter->nodes_size();
       node_index++) {
    const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
        interpreter->node_and_registration(static_cast<int>(node_index));
    const TfLiteNode& node = node_and_reg->first;
    const TfLiteRegistration& reg = node_and_reg->second;
    if (reg.custom_name != nullptr) {
      printf("Node %3zu Operator Custom Name %s\n", node_index,
             reg.custom_name);
    } else {
      printf("Node %3zu Operator Builtin Code %3d %s\n", node_index,
             reg.builtin_code, EnumNamesBuiltinOperator()[reg.builtin_code]);
    }
    printf("  Inputs:");
    PrintTfLiteIntVector(node.inputs);
    printf("  Outputs:");
    PrintTfLiteIntVector(node.outputs);
    if (node.intermediates && node.intermediates->size) {
      printf("  Intermediates:");
      PrintTfLiteIntVector(node.intermediates);
    }
    if (node.temporaries && node.temporaries->size) {
      printf("  Temporaries:");
      PrintTfLiteIntVector(node.temporaries);
    }
  }
}
