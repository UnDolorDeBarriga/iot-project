// train_step.c
#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/lite/c/c_api.h>
#include "emnist_train_model.h"

// A dummy label for our one-image batch
static int64_t train_label[1] = { 2 };  // e.g. class “C”

int main() {
  // 1) Load the trainable model
  TfLiteModel* model = TfLiteModelCreate(
      emnist_litert_train_tflite, emnist_litert_train_tflite_len);
  if (!model) {
    fprintf(stderr, "Failed to load train model\n");
    return 1;
  }

  // 2) Interpreter with TF ops
  TfLiteInterpreterOptions* opts = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(opts, 2);
  TfLiteInterpreter* interp =
    TfLiteInterpreterCreate(model, opts);
  TfLiteInterpreterOptionsDelete(opts);
  TfLiteModelDelete(model);

  // IMPORTANT: initialize all variable tensors
  TfLiteInterpreterResetVariableTensors(interp);

  if (TfLiteInterpreterAllocateTensors(interp) != kTfLiteOk) {
    fprintf(stderr,"Tensor allocation failed\n");
    return 1;
  }

  // 3) Get the “train” signature runner
  TfLiteSignatureRunner* train_runner =
    TfLiteInterpreterGetSignatureRunner(interp, "train");
  if (!train_runner) {
    fprintf(stderr,"No 'train' signature\n");
    return 1;
  }

  // 4) Prep input tensors
  // images: [1,28,28,1] float32
  TfLiteTensor* img_t =
    TfLiteSignatureRunnerGetInputTensor(train_runner, "images");
  float* img_buf = (float*)TfLiteTensorData(img_t);
  // Fill with a dummy all-zeros image (replace with real data)
  for (int i = 0; i < 28*28; i++) img_buf[i] = 0.0f;

  // labels: [1] int64
  TfLiteTensor* lbl_t =
    TfLiteSignatureRunnerGetInputTensor(train_runner, "labels");
  TfLiteTensorCopyFromBuffer(lbl_t,
    train_label, sizeof(train_label));

  // 5) Invoke one training step
  if (TfLiteSignatureRunnerInvoke(train_runner) != kTfLiteOk) {
    fprintf(stderr,"Training Invoke failed\n");
    return 1;
  }

  // 6) Read back the loss
  TfLiteTensor* loss_t =
    TfLiteSignatureRunnerGetOutputTensor(train_runner, "loss");
  float loss_val;
  TfLiteTensorCopyToBuffer(loss_t, &loss_val, sizeof(loss_val));

  printf("Training step done — loss = %.6f\n", loss_val);

  // 7) Clean up
  TfLiteSignatureRunnerDelete(train_runner);
  TfLiteInterpreterDelete(interp);
  return 0;
}
