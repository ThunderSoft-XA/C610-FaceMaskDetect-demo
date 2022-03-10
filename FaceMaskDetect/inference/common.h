#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>

namespace ai2nference
{
typedef  enum _InferType {
    SNPE,
    TFLITE
} InferType;

// In fact, I think of the principal-agent and snpe Tensorflow runtime is the same concept, 
// and the more I think runtime
typedef enum {
    CPU = 0,
    GPU = 1,
	DSP = 2,       //DSP   HEXAGON
    APU = 3,
    NNAPI = 4    //Coordinate the CPU, GPU, DSP and so on to get with high performance
}RunTime;

typedef enum _DataFormat{
    NCHW = 0,
    NHWC = 1
} DataFormat;

// typedef struct __ai_conf {
//     std::string  model_name;
// 	RunTime runtime;
// 	float input_mean;      // 127.5f
// 	float std_mean;          //  127.5f
//     string labels_file_name;    // lable.txt 
// 	string input_layer_type;    // float32, uint8
// } AIConf;

typedef enum _RetState{
    INIT_ERROR = -3,
    EXECUTE_ERROR = -2,
    INVALID_INPUT = -1,
    NO_ERROR = 0
}RetState;

} // namespace ai2nference

#endif
