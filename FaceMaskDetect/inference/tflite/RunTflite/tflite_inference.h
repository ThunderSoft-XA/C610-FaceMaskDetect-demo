#ifndef TFLITE_INFERENCE_HPP
#define TFLITE_INFERENCE_HPP

#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#include "result_process.h"
#include "../../utils/timeutil.h"
#include "../common.h"

using namespace std;
using namespace tflite;

#define LOG(x) std::cerr

namespace ai2nference
{

template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat &mat)
{   
    // cv::Mat temp;
    // mat.copyTo(temp);
	return (vector<_Tp>)(mat.reshape(1, 1));
}

template<typename _Tp>
cv::Mat convertVector2Mat(vector<_Tp> v, int channels, int rows)
{
	cv::Mat mat = cv::Mat(v);//vector ---> Single row mat
	cv::Mat dest = mat.reshape(channels, rows).clone();//PS：must clone()，or wrong
	return dest;
}

//string sqlit func
std::vector<std::string> selfSplit(std::string str, std::string pattern);

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

class InferTFLITE
{
public:
    InferTFLITE(){}
    InferTFLITE(std::string _model_path,DataFormat _format);

    void initRuntime(RunTime _runtime);
    void getTensorInfo();
    TfLiteDelegatePtrMap getDelegatesFromConf();
    template<class T> void loadTfliteData(int  image_width,int image_height,int image_channels,std::vector<T> _input_data ) {
        // get input node and output node at load model and init interpreter
        // const std::vector<int> inputs = this->interpreter->inputs();
        // const std::vector<int> outputs = this->interpreter->outputs();
        //record input,output node`s index value
        this->input_index_vec = this->interpreter->inputs();
        this->output_index_vec = this->interpreter->outputs();
        // get fact the data size info of input node
        TfLiteIntArray* dims;
        for(auto input : this->input_index_vec) {
            LOG(INFO) << "input node index value: " << input << std::endl;
            dims = this->interpreter->tensor(input)->dims;
            int wanted_height = dims->data[1];
            int wanted_width = dims->data[2];
            int wanted_channels = dims->data[3];

            std::cout << "node " << input << ": wanted_height = " << wanted_height << \
                ",wanted_width = " << wanted_width << ",wanted_channels = " << wanted_channels;
            
            switch (interpreter->tensor(input)->type) {
                case kTfLiteFloat32:
                    resize<float>(interpreter->typed_tensor<float>(input), _input_data.data(),
                                    image_height, image_width, image_channels, wanted_height,
                                    wanted_width, wanted_channels,false, 127.5f, 127.5f);
                break;
                case kTfLiteUInt8:
                    resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), _input_data.data(),
                                    image_height, image_width, image_channels, wanted_height,
                                    wanted_width, wanted_channels,false, 127.5f, 127.5f);
                break;
                default:
                    LOG(FATAL) << "cannot handle input type "
                                << interpreter->tensor(input)->type << " yet";
                    exit(-1);
            }
        }
        
    }

    template<class T> void doInference(std::vector<std::vector<T>>*_result)
    {
        struct timeval start_time, stop_time;
        gettimeofday(&start_time, nullptr);

        
        if (this->interpreter->Invoke() != kTfLiteOk) {
            LOG(FATAL) << "Failed to invoke tflite!\n";
            return ;
        }

        gettimeofday(&stop_time, nullptr);
        LOG(INFO) << "invoked \n";
        LOG(INFO) << "average time: "
                << (get_us(stop_time) - get_us(start_time)) / (/*s->loop_count*/1 * 1000)
                << " ms \n";

        const float threshold = 0.001f;

        std::vector<std::pair<float, int>> top_results;
        
        //Multiple output
        unsigned int output_node_index = 0;
        for(auto output_node : this->output_index_vec) {
            std::cout << "this is output node " << output_node << std::endl;
            TfLiteIntArray* output_dims = interpreter->tensor(output_node)->dims;
            // assume output dims to be something like (1, 1, ... ,size)
            // this output_dims->size ,example ,1*1*10*2 ----->size = 4
            //then,output_dims->data,---->data[0] = 1,data[2] = 10, data[3] = 3
            // auto output_size = output_dims->data[output_dims->size - 1];
            auto output_size = 1;
            std::vector<int> output_dims_vec;
            for(int i = 0; i < output_dims->size; i++ ){
                output_dims_vec.push_back(output_dims->data[i]);
                output_size *= output_dims->data[i];
            }
            std::cout << "output size " << output_size << std::endl;
            std::vector<T> result_vec;
            const long count = output_size;  // NOLINT(runtime/int)
            switch (interpreter->tensor(output_node)->type) {
            case kTfLiteFloat32:
                for (int i = 0; i < count; ++i) {
                    float value;
                    //0 == outputs()[index] == actually output node index
                    //example, output1`s index 123,output1`s index 124,output1`s index 125,so, the index in (0~2),124 = outputs()[1] 
                    value = this->interpreter->typed_output_tensor<float>(output_node_index)[i];
                    result_vec.push_back(value);
                }
                
                break;
            case kTfLiteUInt8:
                uint8_t value;
                for (int i = 0; i < count; ++i) {
                    value = this->interpreter->typed_output_tensor<float>(0)[i] / 255.0;
                    result_vec.push_back(value);
                }
                get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                                    output_size, 5, threshold,
                                    &top_results, false);
                break;
            default:
                LOG(FATAL) << "cannot handle output type "
                            << interpreter->tensor(output_node)->type << " yet";
                exit(-1);
            }
            _result->push_back(result_vec);
            if(output_node_index < this->output_index_vec.size()){
                output_node_index++;
            }
        }
        
        std::cout << "result vector size is " << _result->size()<< std::endl;
    }

    TfLiteType getTensorType(int node_index){
        return this->interpreter->tensor(node_index)->type;
    }

private:
    string model_name;
    RunTime runtime;
    DataFormat data_format;
    std::vector<int> input_index_vec;
    std::vector<int> output_index_vec;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
};
    
} // namespace ai2nference


#endif