#ifndef FACE_MASK_HPP
#define FACE_MASK_HPP

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "face_mask_box.hpp"

using namespace std;
using namespace cv;

namespace face_mask {

class FaceMask {
public:
    FaceMask() {
        int anchor_total = 0;
        for(int i = 0; i < 5; i++) {
            anchor_total += feature_map_sizes[i] * feature_map_sizes[i];
        }
        anchor_total *= 4;

        // anchors = (float **)new float[anchor_total][4];
        anchors  = std::vector<std::vector<float>>(anchor_total, std::vector<float>(4));
    }

    std::string model_file;
    float conf_threshold = 0.5f;
    float IOU_threshold = 0.4f;

    const int feature_map_sizes[5] = {33, 17, 9, 5, 3};
    const float anchor_sizes[5][2] = {{0.04f, 0.056f}, 
        {0.08f, 0.11f}, {0.16f, 0.22f}, 
        {0.32f, 0.45f}, {0.64f, 0.72f}};
    const float anchor_ratios[3] ={1.0f, 0.62f, 0.42f};

    // float** anchors;
    std::vector<std::vector<float>> anchors;

    std::vector<FaceMaskBox>* detectFaceMasks(std::vector<std::vector<float>>& result);
    void generateAnchors();

private:
    void decodeBBox(std::vector<FaceMaskBox> boxes);
    void nms(std::vector<FaceMaskBox> boxes, float threshold, String method);
};

// loc 1,5972,4  cls 1,5972,2
std::vector<FaceMaskBox>* FaceMask::detectFaceMasks(std::vector<std::vector<float>>& result)
{
    std::vector<FaceMaskBox>* filteredBoxes = new std::vector<FaceMaskBox>();
    std::cout << "enter detect facemask........" << std::endl;
    for(int i = 0; i < 5972; i++) {
            int idxCls = -1;
            if(result[1][2 * i] > result[1][2 * i + 1]) {
                idxCls = 0;
            } else {
                idxCls = 1;
            }

            if(result[1][2 * i + idxCls] > conf_threshold) {
                FaceMaskBox* box = new FaceMaskBox();
                // core
                box->score = result[1][2 * i + idxCls];
                // box
                box->box[0] = result[0][4 * i];
                box->box[1] = result[0][4 * i + 1];
                box->box[2] = result[0][4 * i + 2];
                box->box[3] = result[0][4 * i + 3];

                box->cls = idxCls;

                if(idxCls == 0) {
                    box->title = "Mask";
                } else {
                    box->title = "Without_Mask";
                }

                box->index = i;

                filteredBoxes->push_back(*box);
            }
        }

        //parse Box params
        std::cout << "decode face mask box........" << std::endl;
        decodeBBox(*filteredBoxes);

        //NMS
        std::cout << "NMS processing ........" << std::endl;
        nms(*filteredBoxes, IOU_threshold, "Union");

        return filteredBoxes;
}

void FaceMask::generateAnchors() 
{
    int index = 0;
    for(int i=0; i<5; i++) {
        std::cout << "enter generate anchor function`s first for loop" << std::endl;
        float center[feature_map_sizes[i]];
        for(int j=0; j<feature_map_sizes[i]; j++) {
            center[j] = 1.0f * (float)(-feature_map_sizes[i]/2 + j) / (float)feature_map_sizes[i] + 0.5f;
        }
        float offset[4][4];
        for(int j=0; j<2; j++) {
            float ratio = anchor_ratios[0];
            float width = anchor_sizes[i][j] * (float)sqrt((double)ratio);
            float height = anchor_sizes[i][j] / (float)sqrt((double)ratio);
            //offset[j] =  {-width / 2.0f, -height / 2.0f, width / 2.0f, height / 2.0f};
            offset[j][0] = -width / 2.0f;
            offset[j][1] = -height / 2.0f;
            offset[j][2] = width / 2.0f;
            offset[j][3] = height / 2.0f;

        }
        for(int j=0; j<2; j++) {
            float s1 = anchor_sizes[i][0];
            float ratio = anchor_ratios[1+j];
            float width = s1 * (float)sqrt((double)ratio);
            float height = s1 / (float)sqrt((double)ratio);
            //offset[2+j] = new float[] {-width / 2.0f, -height / 2.0f, width / 2.0f, height / 2.0f};
            offset[2 + j][0] = -width / 2.0f;
            offset[2 + j][1] = -height / 2.0f;
            offset[2 + j][2] = width / 2.0f;
            offset[2 + j][3] = height / 2.0f;
        }
        for(int y=0; y<feature_map_sizes[i]; y++) {
            for(int x=0; x<feature_map_sizes[i]; x++) {
                for(int j=0; j<4; j++) {
                    //anchors[index] = new float[]{center[x]+offset[j][0], center[y]+offset[j][1], center[x]+offset[j][2], center[y]+offset[j][3]};
                    anchors[index][0] = center[x]+offset[j][0];
                    anchors[index][1] = center[y]+offset[j][1];
                    anchors[index][2] = center[x]+offset[j][2];
                    anchors[index][3] = center[y]+offset[j][3];
                    printf("%f  ---- %f ---- % f ---- %f \n",anchors[index][0],anchors[index][1],anchors[index][2],anchors[index][3]);
                    index++;
                }
            }
        }
    }
    std::cout << "successfully generate anchor array" << std::endl;
}

void FaceMask::decodeBBox(std::vector<FaceMaskBox> boxes)
{
    std::cout << "enter decode box process function......box count: " << boxes.size() << std::endl;
    for(unsigned int i = 0; i < boxes.size(); i++) {
        FaceMaskBox box = boxes[i];
        float anchor_center_x = (anchors[box.index][0] + anchors[box.index][2])/2;
        float anchor_center_y = (anchors[box.index][1] + anchors[box.index][3])/2;
        float anchor_w = anchors[box.index][2] - anchors[box.index][0];
        float anchor_h = anchors[box.index][3] - anchors[box.index][1];

        float predict_center_x = box.box[0] * 0.1f * anchor_w + anchor_center_x;
        float predict_center_y = box.box[1] * 0.1f * anchor_h + anchor_center_y;
        float predict_w = (float)exp((double)box.box[2] * 0.2) * anchor_w;
        float predict_h = (float)exp((double)box.box[3] * 0.2) * anchor_h;

        box.box[0] = predict_center_x - predict_w / 2;
        box.box[1] = predict_center_y - predict_h / 2;
        box.box[2] = predict_center_x + predict_w / 2;
        box.box[3] = predict_center_y + predict_h / 2;
    }
    std::cout << "exit decode box process function !!!"<< std::endl;
}

void FaceMask::nms(std::vector<FaceMaskBox> boxes, float threshold, String method)
{
    // NMS.compare one by one
    // int delete_cnt = 0;
    std::cout << "enter NMS process function......box count: " << boxes.size() << std::endl;

    for (unsigned int i = 0; i < boxes.size(); i++) {
        FaceMaskBox box = boxes[i];
        if (!box.deleted) {
            // score<0,current rect will been deleted.
            for (unsigned int j = i + 1; j < boxes.size(); j++) {
                FaceMaskBox box2 = boxes[j];
                if ((!box2.deleted) && (box2.cls==box.cls)) {
                    float x1 = max(box.box[0], box2.box[0]);
                    float y1 = max(box.box[1], box2.box[1]);
                    float x2 = min(box.box[2], box2.box[2]);
                    float y2 = min(box.box[3], box2.box[3]);
                    if (x2 < x1 || y2 < y1) {
                        continue;
                    }
                    float areaIoU = (x2 - x1 + 0.001) * (y2 - y1 + 0.001);
                    float iou = 0.0f;
                    if ("Union" == method) {
                        iou = 1.0f * areaIoU / (box.area() + box2.area() - areaIoU);
                    }else if ("Min" == method) {
                        iou = 1.0f * areaIoU / (min(box.area(), box2.area()));
                    }
                    if (iou >= threshold) { // delete rect of prob value smaller 
                        if (box.score > box2.score) {
                            box2.deleted = true;
                        }else {
                            box.deleted = true;
                        }
                    }
                }
            }
        }
    }
    std::cout << "exit NMS process function......" << std::endl;
}

}

#endif