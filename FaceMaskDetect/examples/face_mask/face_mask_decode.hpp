#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define RUN_TYPE_IMAGE		0
#define RUN_TYPE_CAMARA		1

#define USE_RUN_TYPE RUN_TYPE_CAMARA

void blobFromImagesFromOpencv(InputArrayOfArrays images_, OutputArray blob_, double scalefactor,
                    Size size, const Scalar& mean_, bool swapRB, bool crop, int ddepth)
{
    // CV_TRACE_FUNCTION();
    CV_CheckType(ddepth, ddepth == CV_32F || ddepth == CV_8U, "Blob depth should be CV_32F or CV_8U");
    // 数据类型默认是CV_32F, 不支持CV_8U
    if (ddepth == CV_8U)
    {
        CV_CheckEQ(scalefactor, 1.0, "Scaling is not supported for CV_8U blob depth");
        CV_Assert(mean_ == Scalar() && "Mean subtraction is not supported for CV_8U blob depth");
    }

    std::vector<Mat> images;
    images_.getMatVector(images);
    CV_Assert(!images.empty());
    // 相当于遍历一个batch里的图片，预处理
    for (size_t i = 0; i < images.size(); i++)
    {
        Size imgSize = images[i].size();
        if (size == Size())
            size = imgSize;
        if (size != imgSize)
        {	
            if(crop)	// 按照宽高比resize，再裁剪；这里的resize方式与torchvision不一样
            {
              float resizeFactor = std::max(size.width / (float)imgSize.width,
                                            size.height / (float)imgSize.height);
              resize(images[i], images[i], Size(), resizeFactor, resizeFactor, INTER_LINEAR);
              Rect crop(Point(0.5 * (images[i].cols - size.width),
                              0.5 * (images[i].rows - size.height)),
                        size);
              images[i] = images[i](crop);
            }
            else	// 直接resize
              resize(images[i], images[i], size, 0, 0, INTER_LINEAR);
        }
        if(images[i].depth() == CV_8U && ddepth == CV_32F)
            images[i].convertTo(images[i], CV_32F);
        Scalar mean = mean_;
        if (swapRB)
            std::swap(mean[0], mean[2]);
		// 先减去均值再尺度缩放，没有协方差的处理；
		// torchvision里是先尺度缩放，在减均值，再除以协方差。
        images[i] -= mean;
        images[i] *= scalefactor;
    }

    size_t nimages = images.size();
    Mat image0 = images[0];
    int nch = image0.channels();
    CV_Assert(image0.dims == 2);
    if (nch == 3 || nch == 4)
    {
    	// 创建四维矩阵 blob  
        int sz[] = { (int)nimages, nch, image0.rows, image0.cols }; // NCHW
        blob_.create(4, sz, ddepth);
        Mat blob = blob_.getMat();
        Mat ch[4];

        for(size_t i = 0; i < nimages; i++ )
        {
            const Mat& image = images[i];
            CV_Assert(image.depth() == blob_.depth());
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
            CV_Assert(image.size() == image0.size());

            for( int j = 0; j < nch; j++ )
                ch[j] = Mat(image.rows, image.cols, ddepth, blob.ptr((int)i, j));  // ch[j]指向blob
            if(swapRB)
                std::swap(ch[0], ch[2]);
            // 讲image各通道劈开，填入ch中，于是blob也就有数据了
            split(image, ch);
        }
    }
    else
    {
    	// 单通道
       CV_Assert(nch == 1);
       int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
       blob_.create(4, sz, ddepth);
       Mat blob = blob_.getMat();

       for(size_t i = 0; i < nimages; i++ )
       {
           const Mat& image = images[i];
           CV_Assert(image.depth() == blob_.depth());
           nch = image.channels();
           CV_Assert(image.dims == 2 && (nch == 1));
           CV_Assert(image.size() == image0.size());

           image.copyTo(Mat(image.rows, image.cols, ddepth, blob.ptr((int)i, 0)));
       }
    }
}

vector<vector<float>> generate_anchors(const vector<float> &ratios, const vector<int> &scales, vector<float> &anchor_base)
{
	vector<vector<float>> anchors;
	for (int idx = 0; idx < scales.size(); idx++) {
		vector<float> bbox_coords;
		int s = scales[idx];
		vector<float> cxys;
		vector<float> center_tiled;
		for (int i = 0; i < s; i++) {
			float x = (0.5 + i) / s;
			cxys.push_back(x);
		}

		for (int i = 0; i < s; i++) {
			float x = (0.5 + i) / s;
			for (int j = 0; j < s; j++) {
				for (int k = 0; k < 8; k++) {
					center_tiled.push_back(cxys[j]);
					center_tiled.push_back(x);
					//printf("%f %f ", cxys[j], x);
				}
				//printf("\n");
			}
			//printf("\n");
		}

		vector<float> anchor_width_heights;
		for (int i = 0; i < anchor_base.size(); i++) {
			float scale = anchor_base[i] * pow(2, idx);
			anchor_width_heights.push_back(-scale / 2.0);
			anchor_width_heights.push_back(-scale / 2.0);
			anchor_width_heights.push_back(scale / 2.0);
			anchor_width_heights.push_back(scale / 2.0);
			//printf("%f %f %f %f\n", -scale / 2.0, -scale / 2.0, scale / 2.0, scale / 2.0);
		}

		for (int i = 0; i < anchor_base.size(); i++) {
			float s1 = anchor_base[0] * pow(2, idx);
			float ratio = ratios[i + 1];
			float w = s1 * sqrt(ratio);
			float h = s1 / sqrt(ratio);
			anchor_width_heights.push_back(-w / 2.0);
			anchor_width_heights.push_back(-h / 2.0);
			anchor_width_heights.push_back(w / 2.0);
			anchor_width_heights.push_back(h / 2.0);
			//printf("s1:%f, ratio:%f w:%f h:%f\n", s1, ratio, w, h);
			//printf("%f %f %f %f\n", -w / 2.0, -h / 2.0, w / 2.0, h / 2.0);
		}

		int index = 0;
		//printf("\n");
		for (float &a : center_tiled) {
			float c = a + anchor_width_heights[(index++) % anchor_width_heights.size()];
			bbox_coords.push_back(c);
			//printf("%f ", c);
		}

		//printf("bbox_coords.size():%d\n", bbox_coords.size());
		int anchors_size = bbox_coords.size() / 4;
		for (int i = 0; i < anchors_size; i++) {
			vector<float> f;
			for (int j = 0; j < 4; j++) {
				f.push_back(bbox_coords[i * 4 + j]);
			}
			anchors.push_back(f);
		}
	}

	return anchors;
}

vector<cv::Rect2f> decode_bbox(vector<vector<float>> &anchors, float *raw)
{
	vector<cv::Rect2f> rects;
	float v[4] = { 0.1, 0.1, 0.2, 0.2 };

	int i = 0;
	for (vector<float>& k : anchors) {
		float acx = (k[0] + k[2]) / 2;
		float acy = (k[1] + k[3]) / 2;
		float cw = (k[2] - k[0]);
		float ch = (k[3] - k[1]);

		float r0 = raw[i++] * v[i % 4];
		float r1 = raw[i++] * v[i % 4];
		float r2 = raw[i++] * v[i % 4];
		float r3 = raw[i++] * v[i % 4];

		float centet_x = r0 * cw + acx;
		float centet_y = r1 * ch + acy;

		float w = exp(r2) * cw;
		float h = exp(r3) * ch;
		float x = centet_x - w / 2;
		float y = centet_y - h / 2;
		rects.push_back(cv::Rect2f(x, y, w, h));
	}

	return rects;
}

typedef struct FaceInfo {
	Rect2f rect;
	float score;
	int id;
} FaceInfo;

bool increase(const FaceInfo & a, const FaceInfo & b) {
	return a.score > b.score;
}

std::vector<int> do_nms(std::vector<FaceInfo>& bboxes, float thresh, char methodType) {
	std::vector<int> bboxes_nms;
	if (bboxes.size() == 0) {
		return bboxes_nms;
	}
	std::sort(bboxes.begin(), bboxes.end(), increase);

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
	bool all_merged = false;

	while (!all_merged) {
		while (select_idx < num_bbox && mask_merged[select_idx] == 1)
			select_idx++;
		if (select_idx == num_bbox) {
			all_merged = true;
			continue;
		}

		bboxes_nms.push_back(bboxes[select_idx].id);
		mask_merged[select_idx] = 1;

		Rect2f &select_bbox = bboxes[select_idx].rect;
		float area1 =(select_bbox.width + 1) * (select_bbox.height + 1);

		select_idx++;
#pragma omp parallel for num_threads(8)
		for (int32_t i = select_idx; i < num_bbox; i++) {
			if (mask_merged[i] == 1)
				continue;

			Rect2f & bbox_i = bboxes[i].rect;
			float x = std::max<float>(select_bbox.x,bbox_i.x);
			float y = std::max<float>(select_bbox.y, bbox_i.y);
			float w = std::min<float>(select_bbox.width + select_bbox.x, bbox_i.x + bbox_i.width) - x + 1;
			float h = std::min<float>(select_bbox.height + select_bbox.y, bbox_i.y + bbox_i.height) - y + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = (bbox_i.width + 1) * (bbox_i.height + 1);
			float area_intersect = w * h;

			switch (methodType) {
			case 'u':
				if (area_intersect / (area1 + area2 - area_intersect) > thresh)
					mask_merged[i] = 1;
				break;
			case 'm':
				if (area_intersect / std::min(area1, area2) > thresh)
					mask_merged[i] = 1;
				break;
			default:
				break;
			}
		}
	}
	return bboxes_nms;
}

vector<int> single_class_non_max_suppression(vector<cv::Rect2f> &rects, float *confidences, int c_len, vector<int> &classes, vector <float>&bbox_max_scores)
{
	vector<int> keep_idxs;

	float conf_thresh = 0.5;
	float iou_thresh = 0.5;
	int keep_top_k = -1;
	if (rects.size() <= 0) {
		return keep_idxs;
	}

	for (int i = 0; i < c_len; i += 2) {
		float max = confidences[i];
		int classess = 0;
		if (max < confidences[i + 1]) {
			max = confidences[i + 1];
			classess = 1;
		}
		classes.push_back(classess);
		bbox_max_scores.push_back(max);
	}

	vector <FaceInfo>infos;
	for (int i = 0; i < bbox_max_scores.size(); i++) {
		if (bbox_max_scores[i] > conf_thresh) {
			FaceInfo info;
			info.rect = rects[i];
			info.score = bbox_max_scores[i];
			info.id = i;
			infos.push_back(info);
		}
	}

	keep_idxs = do_nms(infos, iou_thresh, 'u');
	return keep_idxs;
}
