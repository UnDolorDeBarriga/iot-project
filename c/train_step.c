// ──────────────────────────────────────────────────────────────────────────────
// train_step.c
//
//   A minimal “one‐step” training example for the EMNIST alphabet model.
//   • Includes the “experimental” C‐API header (for signature runners & ResetVars).
//   • Assumes you have already run:
//       xxd -i emnist_litert_train.tflite > emnist_train.h
//     and that `emnist_train.h` sits alongside this file.
// ──────────────────────────────────────────────────────────────────────────────

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_experimental.h>   // <-- for TfLiteSignatureRunner* & ResetVariableTensors

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "emnist_train.h"   // generated via:  xxd -i emnist_litert_train.tflite > emnist_train.h

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <28x28_grayscale.png> <label_int>\n", argv[0]);
    fprintf(stderr, "  label_int should be 0..25, corresponding to A..Z.\n");
    return 1;
  }

  // 1) Load the TRAINABLE model from memory:
  TfLiteModel *model = TfLiteModelCreate(
      models_emnist_train_tflite,
      models_emnist_train_tflite_len
  );
  if (!model) {
    fprintf(stderr, "Failed to load trainable model\n");
    return 1;
  }

  // 2) Interpreter setup:
  TfLiteInterpreterOptions *opts = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(opts, 4);
  TfLiteInterpreter *interp = TfLiteInterpreterCreate(model, opts);
  TfLiteInterpreterOptionsDelete(opts);
  TfLiteModelDelete(model);

  // 3) Grab the “train” signature runner:
  TfLiteSignatureRunner *train_runner =
      TfLiteInterpreterGetSignatureRunner(interp, "train");
  if (!train_runner) {
    fprintf(stderr, "ERROR: could not find signature named \"train\"\n");
    TfLiteInterpreterDelete(interp);
    return 1;
  }

  // 4) Reset all variable tensors (so they start from saved weights):
  if (TfLiteInterpreterResetVariableTensors(interp) != kTfLiteOk) {
    fprintf(stderr, "ERROR: ResetVariableTensors failed\n");
    TfLiteInterpreterDelete(interp);
    return 1;
  }

  // 5) Allocate tensors for the signature runner:
  if (TfLiteSignatureRunnerAllocateTensors(train_runner) != kTfLiteOk) {
    fprintf(stderr, "ERROR: AllocateTensors (signature) failed\n");
    TfLiteInterpreterDelete(interp);
    return 1;
  }

  // 6) Load & preprocess the 28×28 image:
  int w, h, channels;
  unsigned char *img = stbi_load(argv[1], &w, &h, &channels, 1);
  if (!img || w != 28 || h != 28) {
    fprintf(stderr, "Error: need a 28×28 grayscale PNG\n");
    if (img) stbi_image_free(img);
    TfLiteInterpreterDelete(interp);
    return 1;
  }
  TfLiteTensor *img_tensor =
      TfLiteSignatureRunnerGetInputTensor(train_runner, "images");
  if (!img_tensor) {
    fprintf(stderr, "ERROR: could not get input tensor \"images\"\n");
    stbi_image_free(img);
    TfLiteInterpreterDelete(interp);
    return 1;
  }
  float *img_buf = (float *)TfLiteTensorData(img_tensor);

  int img_dims = TfLiteTensorNumDims(img_tensor);
  int num_pixels = 1;
  for (int i = 0; i < img_dims; i++) {
    num_pixels *= TfLiteTensorDim(img_tensor, i);
  }
  for (int i = 0; i < num_pixels; i++) {
    img_buf[i] = img[i] / 255.0f;
  }
  stbi_image_free(img);

  // 7) Set the “labels” input tensor (expects int64_t):
  TfLiteTensor *label_tensor =
      TfLiteSignatureRunnerGetInputTensor(train_runner, "labels");
  if (!label_tensor) {
    fprintf(stderr, "ERROR: could not get input tensor \"labels\"\n");
    TfLiteInterpreterDelete(interp);
    return 1;
  }
  int64_t *lbl_buf = (int64_t *)TfLiteTensorData(label_tensor);
  int lbl = atoi(argv[2]);
  lbl_buf[0] = (int64_t)lbl;

  // 8) Invoke one training step:
  if (TfLiteSignatureRunnerInvoke(train_runner) != kTfLiteOk) {
    fprintf(stderr, "Inference/Training step failed\n");
    TfLiteInterpreterDelete(interp);
    return 1;
  }

  // 9) Read back the “loss” output by name:
  const TfLiteTensor *loss_tensor =
      TfLiteSignatureRunnerGetOutputTensor(train_runner, "loss");
  if (!loss_tensor) {
    fprintf(stderr, "ERROR: could not get output tensor \"loss\"\n");
    TfLiteInterpreterDelete(interp);
    return 1;
  }
  const float *loss_buf = (const float *)TfLiteTensorData(loss_tensor);
  printf("Step loss = %.6f\n", loss_buf[0]);

  // 10) (Optional) You could also grab “infer” signature here to see
  //     how the model classifies *after* this update, but we’ll stop.

  TfLiteInterpreterDelete(interp);
  return 0;
}
