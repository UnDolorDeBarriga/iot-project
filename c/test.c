#include <stdio.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/c/c_api.h>

int main()
{
    int numThreads = 4;

    TfLiteModel *model = TfLiteModelCreateFromFile("../models/test.tflite");

    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, numThreads);
    TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);

    float x[] = {15.0f};

    TfLiteTensor *inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    TfLiteTensorCopyFromBuffer(inputTensor, x, sizeof(x));

    TfLiteInterpreterInvoke(interpreter);

    float y[1];

    const TfLiteTensor *outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    TfLiteTensorCopyToBuffer(outputTensor, y, sizeof(y));

    printf("%.4f\n", y[0]);

    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return 0;
}