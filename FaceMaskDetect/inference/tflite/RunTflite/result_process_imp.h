#include <algorithm>
#include <functional>
#include <queue>
#include <opencv2/opencv.hpp>

#include "absl/memory/memory.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

using namespace std;
using namespace cv;
using namespace tflite;

namespace ai2nference {

extern bool input_floating;

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
template <class T>
void get_top_n(T* prediction, int prediction_size, size_t num_results,
               float threshold, std::vector<std::pair<float, int>>* top_results,
               bool input_floating) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      top_result_pq;

  const long count = prediction_size;  // NOLINT(runtime/int)
  for (int i = 0; i < count; ++i) {
    float value;
    if (input_floating)
      value = prediction[i];
    else
      value = prediction[i] / 255.0;
    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

// for getting face /iris landmark (x,y,z) point value
template <class T>
void landmark_result(T* prediction, int prediction_size, std::vector<cv::Point3_<T>>* results,bool input_floating)
{
    const long count = prediction_size;  // NOLINT(runtime/int)
    cv::Point3_<T> point_value;
    for (int index = 0; index < (count + 1) /3; ++index) {
        if (input_floating) {
            point_value = cv::Point3_<T>(prediction[3*index],prediction[3*index+1],prediction[3*index+2]);
        } else {
            point_value = cv::Point3_<T>(prediction[3*index] /255.0,prediction[3*index+1]/255.0,prediction[3*index+2]/255.0);
        }
        results->push_back(point_value);
    }
}

//convert original size image to model wanted size image, from tflite example label_image
//in fact,the function is a whole tflite model inference process
template <class T>
void resize(T* out, uint8_t* _input, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, bool input_floating, float input_mean, float std_mean) {

    int number_of_pixels = image_height * image_width * image_channels;
    std::unique_ptr<Interpreter> interpreter(new Interpreter);

    int base_index = 0;
    
    // two inputs: input and new_sizes
    interpreter->AddTensors(2, &base_index);
    // one output
    interpreter->AddTensors(1, &base_index);
    // set input and output tensors
    interpreter->SetInputs({0, 1});
    interpreter->SetOutputs({2});

    if(out == nullptr) {
        std::cout << "the out data point is null" << std::endl; 
    }

    // set parameters of tensors
    TfLiteQuantizationParams quant;
    interpreter->SetTensorParametersReadWrite(
        0, kTfLiteFloat32, "input",
        {1, image_height, image_width, image_channels}, quant);
    interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                                quant);
    interpreter->SetTensorParametersReadWrite(
        2, kTfLiteFloat32, "output",
        {1, wanted_height, wanted_width, wanted_channels}, quant);

        

    ops::builtin::BuiltinOpResolver resolver;
    const TfLiteRegistration* resize_op = resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR, 1);
    auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(malloc(sizeof(TfLiteResizeBilinearParams)));
    params->align_corners = false;
    params->half_pixel_centers = false;
    interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,nullptr);

    interpreter->AllocateTensors();
    
    // fill input image
    // input[] are integers, cannot do memcpy() directly
    auto input = interpreter->typed_tensor<float>(0);
    for (int i = 0; i < number_of_pixels; i++) {
        input[i] = _input[i];
    }

    
    // fill new_sizes
    interpreter->typed_tensor<int>(1)[0] = wanted_height;
    interpreter->typed_tensor<int>(1)[1] = wanted_width;

    interpreter->Invoke();

    
    // int fact_input = interpreter->inputs()[0];
    auto output = interpreter->typed_tensor<float>(2);
    auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;
    
    for (int i = 0; i < output_number_of_pixels; i++) {
      if (input_floating) {
        out[i] = (output[i] - input_mean) / std_mean;
      } else {
        out[i] = (uint8_t)output[i];
      }
    }
    
}

}