#include <iostream>

#include "RunTflite/result_process.h"
#include "RunTflite/tflite_inference.h"
#include "../common.h"


int main(int argc,char **argv)
{
    ai2nference::InferTFLITE* tflite_inference = new ai2nference::InferTFLITE(argv[1],ai2nference::DataFormat::NHWC);
    tflite_inference->initRuntime(ai2nference::RunTime::CPU);
    cv::Mat src_mat = cv::imread(argv[2]);
    cv::imshow("source mat",src_mat);
    cv::waitKey(0);
    std::vector<std::vector<float>> inference_result;
    std::vector<uchar> object_mat_vec = ai2nference::convertMat2Vector<uchar>(src_mat);
    tflite_inference->loadTfliteData(src_mat.rows,src_mat.cols,src_mat.channels(),object_mat_vec);
    tflite_inference->doInference<float>(src_mat,&inference_result);

    return 0;
}