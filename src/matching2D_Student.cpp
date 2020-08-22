#include <numeric>
#include "matching2D.hpp"

#include <vector>
#include <string>

// Find best matches for keypoints in two camera images based on several matching methods
void Matching2D::MatchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef,
                                  cv::Mat &descSource, cv::Mat &descRef,
                                  std::vector<cv::DMatch> &matches, std::string descriptorType,
                                  std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void Matching2D::DescKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType =="BRISK") {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else {

        //...
    }

    // perform feature description
    double t = cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void Matching2D::DetKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    // compute detector parameters based on image size
    //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int blockSize = 4;
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = cv::getTickCount();
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize,
                            false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms\n";
}

void Matching2D::DetKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) {
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {
                    // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
}

/**
 * Implement FAST, BRISK, ORB, AKAZE, SIFT keypoint detection.
 * Refer to https://docs.opencv.org/4.1.0/d5/d51/group__features2d__main.html
 * and https://docs.opencv.org/4.1.0/d0/d13/classcv_1_1Feature2D.html
 * @param keypoints
 * @param img
 * @param detectorType
 * @param bVis
 */
void Matching2D::DetKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                                    const std::string& detectorType){
    cv::Ptr<cv::Feature2D> detector;
    if (detectorType == "FAST") {
        // difference between intensity of the central pixel and pixels of a circle around this pixel
        int threshold = 30;
        bool bNMS = true; // perform non-maxima suppression on keypoints
        // TYPE_9_16, TYPE_7_12, TYPE_5_8
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    } else if (detectorType == "BRISK") {
        int thresh = 30; // AGAST detection threshold score.
        int octaves = 3; // detection octaves.
        float patternScale = 1.0f;
        detector = cv::BRISK::create(thresh, octaves, patternScale);
    } else if (detectorType == "ORB") {
        int nfeatures = 500; // The maximum number of features to retain.
        float scaleFactor = 1.2f; // Pyramid decimation ratio, greater than 1.
        int nlevels = 8; // the number of pyramid levels.
        int edgeThreshold = 31; // Size of the border where the features are not detected.
        int firstLevel = 0; // The level of pyramid to put source image to.
        int WTA_K = 2; // The number of points that produce each element of the oriented BRIEF descriptor.
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31; // Size of patch used by the oriented BRIEF descriptor.
        int fastThreshold = 20;
        detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
                                   WTA_K, scoreType, patchSize, fastThreshold);
    } else if (detectorType == "AKAZE") {
        cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptorSize = 0; // Size of the descriptor in bits. 0->Full.
        int descriptorChannels = 3; // Number of channels in the descriptor (1,2,3).
        float threshold = 0.001f; // Detector response threshold to accept point.
        int nOctaves = 4; // Maximum octave evolution of the image.
        int nOctaveLayers = 4; // Default number of sublevels per scale level.
        cv::KAZE::DiffusivityType diffusivityType = cv::KAZE::DIFF_PM_G2;
        detector = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels,
                                     threshold, nOctaves, nOctaveLayers, diffusivityType);
    } else if (detectorType == "SIFT") {
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;
        detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                                 edgeThreshold, sigma);
    }

    double t = cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << detectorType << " with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms\n";
}

void Matching2D::DisplayKeypoints(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, const std::string& detectorType){
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string windowName = detectorType + " Detection Results";
    cv::namedWindow(windowName, 5);
    imshow(windowName, visImage);
    cv::waitKey(0);
}

void
Matching2D::DisplayMatches(const DataFrame &current, const DataFrame &last, const std::vector<cv::DMatch> &matches) {
    cv::Mat matchImg = (current.cameraImg).clone();
    cv::drawMatches(last.cameraImg, last.keypoints,
                    current.cameraImg, current.keypoints,
                    matches, matchImg,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::string windowName = "Matching keypoints between two camera images";
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, matchImg);
    cv::waitKey(0); // wait for key to be pressed
}

void Matching2D::CropKeypoints(const cv::Rect &rect, std::vector<cv::KeyPoint> &keypoints) {
    // remove the keypoint from keypoints if rect does not contain it.
    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(),
                                   [&](const cv::KeyPoint& keyPoint){return !rect.contains(keyPoint.pt);}),
                    keypoints.end());
}
