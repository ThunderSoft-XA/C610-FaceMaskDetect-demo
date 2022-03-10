#ifndef INFERENCE_FACTORY_HPP
#define INFERENCE_FACTORY_HPP

#include "inference.hpp"
#include "common.h"

#include "snpe/RunSnpe/snpe_inference.hpp"

#include "tflite/RunTflite/tflite_inference.h"
#include "tflite/RunTflite/result_process.h"
#include "tflite/RunTflite/result_process_imp.h"

namespace ai2nference
{

class inferencefactory{
public:
    static inferencefactory* getInstance() {
        static inferencefactory* factory;
        return factory;
    }

    Inference* createInference(InferType _type){
        Inference* infer;
        switch (_type) {
        case InferType::SNPE:
            /* code */
            // infer = new InferSNPE();
            break;
        case InferType::TFLITE:
            /* code */
            // infer = new InferTFLITE();
            break;
        
        default:
            break;
        }
        return infer;
    }

private:
    inferencefactory(InferType _inferType);

};
    
} // namespace infer




#endif