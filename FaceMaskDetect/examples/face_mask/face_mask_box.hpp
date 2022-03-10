#ifndef FACE_MASK_BOX_HPP
#define FACE_MASK_BOX_HPP

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define IMAGE_WIDTH 260
#define IMAGE_HEIGHT 260


namespace face_mask {

class FaceMaskBox {

public:
    FaceMaskBox() {
        this->deleted  = false;
        this->title = "";
    }

    ~FaceMaskBox() {
        // delete this->box;
    }

    float box[4] = {0,0,0,0};   // left:box[0],top:box[1],right:box[2],bottom:box[3]
    float score;
    int index;
    bool deleted;
    int cls;
    string title;

    float left() {
        return box[0];
    }
    float top() {
        return box[1];
    }

    float right() {
        return box[2];
    }
    float bottom() {
        return box[3];
    }
    float width() {
        return box[2] - box[0];
    }
    float height() {
        return box[3] - box[1];
    }   

    cv::Rect2f* transform2Rect() {
        int padding = 5;
        float overlayViewHeight = 260;
        float sizeMultiplier = min((float) 260 / (float) 1,
                overlayViewHeight / (float) 1);

        float offsetX = (260 - 1 * sizeMultiplier) / 2;
        float offsetY = (overlayViewHeight - 1 * sizeMultiplier) / 2;

        float left = max((float)padding, sizeMultiplier * box[0] + offsetX);
        float top = max(offsetY + padding, sizeMultiplier * box[1] + offsetY);

        float right = min(box[2] * sizeMultiplier, (float)260 - padding);
        float bottom = min(box[3] * sizeMultiplier + offsetY, (float)260 - padding);

        return new cv::Rect2f(left, top, right - left, bottom - top);
    }

    int area() {
        return (int)(this->width() * this->height());
    }
};

} // namespace face_mask

#endif