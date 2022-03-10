#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <string>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "opencv2/opencv.hpp"
#include "RunTflite/tflite_inference.h"

#include "gst_pipe/gstpipe.hpp"
#include "gst_pipe/gstpipefactory.hpp"

#include "pb_conf/gstreamer.pb.h"
#include "pb_conf/aiconf.pb.h"

#include "examples/param_parse.hpp"
#include "face_mask_box.hpp"
#include "face_mask.hpp"

using namespace std;
using namespace cv;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

using namespace gstpipe;
using namespace ai2nference;
using namespace face_mask;

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

std::vector<gstpipe::GstPipe*> gst_pipe_vec;
std::vector <InferTFLITE *> ai_inference_vec;
cv::Mat* show_mat;

void preview(cv::Mat& imgframe)
{
    cv::Mat showframe;
    cv::cvtColor(imgframe,showframe,CV_BGR2RGBA);
    cv::imshow("sink", showframe);
    cv::waitKey(1);
    return ;
}

void bodyInference()
{
    std::cout << "enter body inference thread........" << std::endl; 
    FaceMask facemask;
    facemask.generateAnchors();
    for(;;) {
        for(auto ai_inference : ai_inference_vec) {
            if(ai_inference == nullptr) {
                continue;
            }

            cv::Mat helmet_mat;
            if(false == gst_pipe_vec[0]->getFrameData(gst_pipe_vec[0],helmet_mat)) {
                continue;
            }
            // cv::cvtColor(helmet_mat_source, helmet_mat,cv::COLOR_BGR2RGB);    helmet tflite model need BGR style
            vector<uchar> helmet_mat_vec = convertMat2Vector<uchar>(helmet_mat);
            std::cout << "loading tflite inference data ........" << std::endl;
            // vector<float> helmet_mat_vec_float(helmet_mat_vec.begin(), helmet_mat_vec.end());
            ai_inference->loadTfliteData<uchar>(helmet_mat.rows,helmet_mat.cols,helmet_mat.channels(),helmet_mat_vec);
            std::vector<std::vector<float>> inference_result;
            ai_inference->doInference<float>(&inference_result);

            std::vector<FaceMaskBox>* facemask_boxes;
            std::cout << "decode inference result data ........" << std::endl;
            facemask_boxes = facemask.detectFaceMasks(inference_result);
            std::cout << "final valid detect box number: " << facemask_boxes->size()  <<std::endl;
            if(facemask_boxes->size() > 0) {
                for(auto facemask_box : *facemask_boxes) {
                    printf("box info: left x top = %f x %f\n",facemask_box.left(),facemask_box.top());
                    if(facemask_box.deleted) {
                        continue;
                    }
                    char score[5];
                    cv::Scalar color;
                    cv::Rect2f* result_box = facemask_box.transform2Rect();
                    sprintf(score,"%4f",facemask_box.score);
                    std::string title = facemask_box.title + ":" + score;
                    if(facemask_box.cls == 0) {
                        color = cv::Scalar(0,255,0);
                    } else {
                        color = cv::Scalar(0,0,255);
                    }

                    cv::putText(helmet_mat, title,cv::Point2f(result_box->x + 10, result_box->y +10), cv::FONT_HERSHEY_SIMPLEX, 0.8, color);
                    rectangle(helmet_mat,*result_box, color, 1);
                }
            }
            std::cout << "decode inference result data and draw mat finished !!!" << std::endl;
            show_mat = new cv::Mat(helmet_mat);
        }
    }
}

void showMat() {
    cv::Mat* last_mat;
    for( ; ;) {
        if(nullptr == show_mat || last_mat == show_mat) {
            continue;
        }
        preview(*show_mat);
        last_mat = show_mat;
    }
}

int main(int argc, char ** argv)
{
    GMainLoop *main_loop = g_main_loop_new(NULL,false);

    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    parse::Parse parma_set = parse::parseArgs(argc,argv);

    if(parma_set.be_fill == false) {
        std::cout << "don`t parse required parma" << std::endl;
        parse::printHelp();
        return -1;
    }

    gstcfg::DataSourceSet data_source_set;
    aicfg::AISet ai_config_set;
    {
        // Read the existing config file.
        int fd = open(parma_set.config_file.c_str(), O_RDONLY);
        FileInputStream* input = new FileInputStream(fd);
        if (!google::protobuf::TextFormat::Parse(input, &data_source_set)) {
            cerr << "Failed to parse gstreamer data source." << endl;
            delete input;
            close(fd);
            return -1;
        }

        fd = open(parma_set.model_file.c_str(), O_RDONLY);
        input = new FileInputStream(fd);
        if (!google::protobuf::TextFormat::Parse(input, &ai_config_set)) {
            cerr << "Failed to parse gstreamer data source." << endl;
            delete input;
            close(fd);
            return -1;
        }
    }

    for(int i = 0; i < ai_config_set.config_size(); i++) {
        std::cout << "AI runtime config info " << i << ",as follow:" <<std::endl;
        const aicfg::AIConfig& ai_config = ai_config_set.config(i);
        ai2nference::InferTFLITE *ai_inference = new InferTFLITE(ai_config.model_path(),(ai2nference::DataFormat)ai_config.data_format());
        ai_inference_vec.push_back(ai_inference);
        std::cout << "loading tflite model ......" <<std::endl;
        ai_inference->initRuntime((ai2nference::RunTime)ai_config.runtime());
    }
    std::cout << "tflite init runtime env finished ......" << std::endl;

    int stream_count = data_source_set.data_source_size();
    gstpipe::GstPipeFactory* pipe_factory = gstpipe::GstPipeFactory::getInstance();
    for (int i = 0; i < stream_count; i++) {
        const gstcfg::DataSource& data_source = data_source_set.data_source(i);
        gstpipe::GstType gsttype = (gstpipe::GstType)data_source.gst_type();
        gstpipe::GstPipe* gst_pipe;
        gst_pipe = pipe_factory->createPipeLine(gsttype);

        gst_pipe->setPipeName(data_source.gst_name());
        gst_pipe->setSinkName(data_source.sink_name());
        gst_pipe->setGstType((GstType)data_source.gst_type());
        gst_pipe->setWidth(data_source.data_info().width());
        gst_pipe->setHeight(data_source.data_info().height());
        gst_pipe->setDecodeType(data_source.data_info().decode());
        gst_pipe->setFormat(data_source.data_info().format());
        gst_pipe->setFramerate(data_source.data_info().framerate());
        gst_pipe->setPath(data_source.gst_path());
        gst_pipe->setNeedCalib(data_source.neeed_calib());
        gst_pipe->setHwDec(data_source.enable_ai());

        gst_pipe_vec.emplace_back(gst_pipe);
    }

    for(auto gst_pipe : gst_pipe_vec) {
        gst_pipe->Init(argc,argv);
        std::thread gst_thread([=]{
            gst_pipe->runGst();
        });
        gst_thread.join();
    }

    std::thread snpeInferenceThread(bodyInference);
    snpeInferenceThread.detach();

    std::thread showThread(showMat);
    showThread.join();

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    g_main_loop_run(main_loop);
    g_main_loop_unref(main_loop);

    return 0;
}
