#ifndef matching2D_hpp
#define matching2D_hpp

#include <cstdio>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

class Matching2D {
public:
    Matching2D() = default;
    ~Matching2D() = default;

    // keypoint detector
    void DetKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void DetKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void DetKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, const std::string& detectorType);

    // keypoint descriptor
    void DescKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors,
                       std::string descriptorType);
    void MatchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef,
                          cv::Mat &descSource, cv::Mat &descRef,
                          std::vector<cv::DMatch> &matches, std::string descriptorType,
                          std::string matcherType, std::string selectorType);

    // helper functions
    void DisplayKeypoints(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, const std::string& detectorType);
    void DisplayMatches(const DataFrame& current, const DataFrame& last, const std::vector<cv::DMatch>& matches);
    void CropKeypoints(const cv::Rect& rect, std::vector<cv::KeyPoint>& keypoints);
};

#endif /* matching2D_hpp */
