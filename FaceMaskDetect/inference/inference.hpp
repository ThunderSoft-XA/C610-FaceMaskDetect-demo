#include <iostream>
#include "snpe/RunSnpe/snpe_inference.hpp"
#include "tflite/RunTflite/tflite_inference.h"

#include "common.h"

using namespace std;

namespace ai2nference
{
class Inference
{
private:
    /* data */
    string model_path;

public:
    Inference(/* args */) {}
    ~Inference(){}

    virtual RetState initRuntime();
    virtual RetState deInit();

    virtual void doInference();
};


} // namespace infer

