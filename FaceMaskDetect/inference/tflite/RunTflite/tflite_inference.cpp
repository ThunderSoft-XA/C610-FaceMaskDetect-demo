#include "../../utils/timeutil.h"
#include "tflite_inference.h"
#include "result_process.h"

#define LOG(x) std::cerr

using namespace tflite;

namespace ai2nference {

//string split function
std::vector<std::string> selfSplit(std::string str, std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern;//extro string for easy opration
    unsigned int size = str.size();
    for (unsigned int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

/**
 * constructor provide two construction type,
 * construct by parameters after parsing the configuration file
 * */
InferTFLITE::InferTFLITE(std::string _model_path,DataFormat _format)
{
    this->model_name = _model_path;
    this->data_format = _format;
}

void InferTFLITE::initRuntime(RunTime _runtime)
{
    this->runtime = _runtime;
    if (this->model_name.empty()) {
        LOG(ERROR) << "no model file name\n";
        return;
    }
    std::cout << "model path: " << this->model_name << std::endl;
    this->model = tflite::FlatBufferModel::BuildFromFile(this->model_name.c_str());
    if(! this->model) {
        LOG(FATAL) << "\nFailed to mmap model " << this->model_name << "\n";
        return;
    }
    this->model->error_reporter();
    LOG(INFO) << "resolved reporter\n";

    tflite::InterpreterBuilder(*this->model,this->resolver)(&this->interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        return;
    }

    // the line code for indicating it that you will use old NNAPI
    this->interpreter->UseNNAPI(false);
    this->interpreter->SetAllowFp16PrecisionForFp32(false);

    auto delegates_ = getDelegatesFromConf();
    for (const auto& delegate : delegates_) {
        if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk) {
            LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
        } else {
            LOG(INFO) << "Applied " << delegate.first << " delegate.";
        }
    }

    if (this->interpreter->AllocateTensors() != kTfLiteOk) {
            LOG(FATAL) << "Failed to allocate tensors!";
    }

    if (false) {
        PrintInterpreterState(interpreter.get());
    }
    
}

// the function for setting Delegates of hardware inference.
// Notice,now ,GPU Delegate temporarily unavailable in c610 open kit.
TfLiteDelegatePtrMap InferTFLITE::getDelegatesFromConf()
{
    TfLiteDelegatePtrMap delegates;
    if(RunTime::GPU == this->runtime) {
        auto delegate = tflite::evaluation::CreateGPUDelegate();
        if (!delegate) {
            LOG(INFO) << "GPU acceleration is unsupported on this platform.";  
        } else {
            delegates.emplace("GPU", std::move(delegate));
        }
    }

    if (RunTime::NNAPI == this->runtime) {
        auto delegate = tflite::evaluation::CreateNNAPIDelegate();
        if (!delegate) {
            LOG(INFO) << "NNAPI acceleration is unsupported on this platform.";
        } else {
            delegates.emplace("NNAPI", std::move(delegate));
        }
    }

    if (RunTime::DSP == this->runtime) {
        const std::string libhexagon_path("/usr/lib");
        auto delegate = tflite::evaluation::CreateHexagonDelegate(libhexagon_path, false);
        if (!delegate) {
            LOG(INFO) << "Hexagon acceleration is unsupported on this platform.";
        } else {
            delegates.emplace("Hexagon", std::move(delegate));
        }
    }

    return delegates;
}

void InferTFLITE::getTensorInfo() 
{
    // show tflite model all tensor info
    if(NULL != this->interpreter) {
        LOG(INFO) << "tensors size: " << this->interpreter->tensors_size() << "\n";
        LOG(INFO) << "nodes size: " << this->interpreter->nodes_size() << "\n";
        LOG(INFO) << "inputs: " << this->interpreter->inputs().size() << "\n";
        LOG(INFO) << "input(0) name: " << this->interpreter->GetInputName(0) << "\n";

        int t_size = this->interpreter->tensors_size();
        for (int i = 0; i < t_size; i++) {
            
            if (this->interpreter->tensor(i)->name)
                LOG(INFO) << i << ": " << this->interpreter->tensor(i)->name << ", "
                        << this->interpreter->tensor(i)->bytes << ", "
                        << this->interpreter->tensor(i)->type << ", "
                        << this->interpreter->tensor(i)->params.scale << ", "
                        << this->interpreter->tensor(i)->params.zero_point << "\n";      
        }

    } else {
        std::cout << "create the model " << this->model_name << "`s interpreter failed !" << std::endl;
    }
}

}