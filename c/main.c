// main_infer.c
#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/lite/c/c_api.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// — Use the INFERENCE‐only model header instead of the trainable one —
#include "emnist_infer.h"

// Helper to map 0→‘A’, 25→‘Z’
static char idx_to_char(int idx) {
  return (idx >= 0 && idx < 26) ? ('A' + idx) : '?';
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <28x28_grayscale.png>\n", argv[0]);
    return 1;
  }

  // 1) Load the INFERENCE TFLite model from memory
  TfLiteModel *model = TfLiteModelCreate(
      models_emnist_infer_tflite,
      models_emnist_infer_tflite_len
  );
  if (!model) {
    fprintf(stderr, "Failed to load inference model\n");
    return 1;
  }

  // 2) Interpreter setup (no TF‐ops required for a pure inference model)
  TfLiteInterpreterOptions *opts = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(opts, 4);
  TfLiteInterpreter *interp = TfLiteInterpreterCreate(model, opts);
  TfLiteInterpreterOptionsDelete(opts);
  TfLiteModelDelete(model);

  // 3) Allocate tensors
  if (TfLiteInterpreterAllocateTensors(interp) != kTfLiteOk) {
    fprintf(stderr, "Tensor allocation failed\n");
    TfLiteInterpreterDelete(interp);
    return 1;
  }

  // 4) Load & preprocess the 28×28 grayscale PNG
  int w, h, channels;
  unsigned char *img = stbi_load(argv[1], &w, &h, &channels, 1);
  if (!img || w != 28 || h != 28) {
    fprintf(stderr, "Error: need a 28×28 grayscale image\n");
    return 1;
  }

  TfLiteTensor *in_t = TfLiteInterpreterGetInputTensor(interp, 0);
  float *in_buf = (float *)TfLiteTensorData(in_t);
  int in_dims = TfLiteTensorNumDims(in_t);
  int num_pixels = 1;
  for (int i = 0; i < in_dims; i++) {
    num_pixels *= TfLiteTensorDim(in_t, i);
  }

  for (int i = 0; i < num_pixels; i++) {
    in_buf[i] = img[i] / 255.0f;
  }
  stbi_image_free(img);

  // 5) Run inference
  if (TfLiteInterpreterInvoke(interp) != kTfLiteOk) {
    fprintf(stderr, "Inference failed\n");
    TfLiteInterpreterDelete(interp);
    return 1;
  }

  // 6) Read & print output
  const TfLiteTensor *out_t = TfLiteInterpreterGetOutputTensor(interp, 0);
  const float *out_buf = (const float *)TfLiteTensorData(out_t);
  int out_dims = TfLiteTensorNumDims(out_t);
  int out_classes = TfLiteTensorDim(out_t, out_dims - 1);

  int best_idx = 0;
  float best_prob = out_buf[0];
  for (int i = 1; i < out_classes; i++) {
    if (out_buf[i] > best_prob) {
      best_prob = out_buf[i];
      best_idx = i;
    }
  }

  printf("Predicted: %c  (index=%d, prob=%.3f)\n",
    idx_to_char(best_idx), best_idx, best_prob);

  TfLiteInterpreterDelete(interp);
  return 0;
}
