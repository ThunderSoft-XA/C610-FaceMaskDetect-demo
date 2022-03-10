#include <iostream>
#include <sstream>
#include "gstpipe.hpp"
#include <opencv2/opencv.hpp>

namespace gstpipe {

class GstRtsp: public GstPipe
{
public:
    GstRtsp(){}
    ~GstRtsp();

    void createPipeLine();
     bool getFrameData(void* _pipe,cv::Mat& _dst);
};

void GstRtsp::createPipeLine()
{
    std::ostringstream cameraPath;
    cameraPath << "rtspsrc location=" << getPath() << " latency=0 tcp-timeout=500 drop-on-latency=true ntp-sync=true" << " ! ";
    cameraPath << "queue ! rtp" << getDecodeType() << "depay ! "<< getDecodeType() << "parse ! queue ! ";
    if(isHwDec()) {
        cameraPath << "omx" << getDecodeType() << "dec " << " ! ";
    } else {
        cameraPath << "avdec_" << getDecodeType() << " ! ";
    }
    cameraPath << "videoscale ! video/x-raw,width=" << getWidth() << ",height=" << getHeight() <<  " ! appsink name=" << getSinkName() << " sync=false  max-lateness=0 max-buffers=1 drop=true";
    setPipeStr(cameraPath.str());
    std::cout << "rtsp Pipeline: " << getPipeStr() << std::endl;
}

bool GstRtsp::getFrameData(void* _pipe,cv::Mat& _dst)
{
    std::shared_ptr<GstSample> sample;
    GstRtsp* pCam = reinterpret_cast<GstRtsp*>(_pipe);

    pCam->getGstSample(sample);
    if(NULL == sample || NULL == sample.get()) {
        std::cout << "the rtsp pipe sample is null or invaild" << std::endl;
        return false;
    }

    GstCaps *sample_caps = gst_sample_get_caps(sample.get());
    if( sample_caps == NULL) {
        return false;
    }
    gint sample_width,sample_height;

    GstStructure *structure = gst_caps_get_structure(sample_caps,0);
    gst_structure_get_int(structure,"width",&sample_width);
    gst_structure_get_int(structure,"height",&sample_height);

    GstBuffer *gst_buffer = gst_sample_get_buffer(sample.get());
    if (NULL == gst_buffer || !sample_width || !sample_height) {
         return false;
    }
    cout <<"==sample width="<< sample_width <<",sample height = " << sample_height <<  endl;
    GstMapInfo sample_map;
    gst_buffer_map(gst_buffer,&sample_map,GST_MAP_READ);
    std::cout << "smaple map size: " << sample_map.size << std::endl;

    unsigned char *data_buffer = (unsigned char*)malloc(sizeof(unsigned char)*sample_map.size);
    cv::Mat sample_mat;
    if(data_buffer != nullptr) {
        memset(data_buffer,0x00, sample_map.size);
        memcpy(data_buffer, (guchar *)sample_map.data, sample_map.size);
        (new cv::Mat(sample_width, sample_height, CV_8UC3, data_buffer, 0))->copyTo(_dst);	
    }

    return true;
}

    
} // namespace gstpipe
