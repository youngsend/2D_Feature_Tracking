/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
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
#include "matching2D.hpp"

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */
    Matching2D matching2D;

    // data location
    std::string dataPath = "../";

    // camera
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        uint current_index = imgIndex % dataBufferSize;

        if (dataBuffer.size() < dataBufferSize) {
            // so that I do not need to reserve dataBufferSize space.
            dataBuffer.push_back(frame);
        } else {
            dataBuffer[current_index] = frame;
        }

        std::cout << "#1 : LOAD IMAGE INTO BUFFER done\n";

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        std::string detectorType = "ORB";

        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based
        /// selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType == "SHITOMASI") {
            matching2D.DetKeypointsShiTomasi(keypoints, imgGray);
        } else if (detectorType == "HARRIS") {
            matching2D.DetKeypointsHarris(keypoints, imgGray);
        } else {
            matching2D.DetKeypointsModern(keypoints, imgGray, detectorType);
        }

        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        matching2D.CropKeypoints(vehicleRect, keypoints);
        matching2D.DisplayKeypoints(keypoints, img, detectorType);

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts) {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            std::cout << " NOTE: Keypoints have been limited!\n";
        }

        // push keypoints and descriptor for current frame to end of data buffer
        dataBuffer[current_index].keypoints = keypoints;
        std::cout << "#2 : DETECT KEYPOINTS done\n";

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection
        /// based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        std::string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        matching2D.DescKeypoints(dataBuffer[current_index].keypoints,
                                 dataBuffer[current_index].cameraImg, descriptors,
                                 descriptorType);

        // push descriptors for current frame to end of data buffer
        dataBuffer[current_index].descriptors = descriptors;

        std::cout << "#3 : EXTRACT DESCRIPTORS done\n";

        // wait until at least two images have been processed
        if (dataBuffer.size() > 1){
            uint last_index = (imgIndex - 1) % dataBufferSize;

            /* MATCH KEYPOINT DESCRIPTORS */

            std::vector<cv::DMatch> matches;
            std::string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            std::string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            std::string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with
            /// t=0.8 in file matching2D.cpp

            matching2D.MatchDescriptors(dataBuffer[last_index].keypoints, dataBuffer[current_index].keypoints,
                                        dataBuffer[last_index].descriptors, dataBuffer[current_index].descriptors,
                                        matches, descriptorType, matcherType, selectorType);

            // store matches in current data frame
            dataBuffer[current_index].kptMatches = matches;

            std::cout << "#4 : MATCH KEYPOINT DESCRIPTORS done\n";

            // visualize matches between current and previous image
//            matching2D.DisplayMatches(dataBuffer[current_index], dataBuffer[last_index], matches);
        }

    } // eof loop over all images

    return 0;
}
