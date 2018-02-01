#include "getAffine.h"

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <algorithm>
#include <exception>

using namespace cv;

std::vector< DMatch > matchDescriptors( Mat descriptors1, Mat descriptors2)
{
    BFMatcher matcher(NORM_L2, false);
    std::vector< DMatch > matches;
    matcher.match( descriptors1, descriptors2, matches );
    return matches;
}

void validateInformationGain(std::vector< DMatch >& matches)
{
    if(matches.size() < 3)
    {
        throw std::runtime_error("Not enough matches to estimate affine transformation");
    }
}

Mat leastSquaresAffineEstimation(std::vector<Point2f>& originalPoints, std::vector<Point2f>& transformedPoints)
{
    /*
    instead of A =  | x_1 y_1 0   0    1 0 |, M = | m_1 |, b = | u_1 |
                    |          ...         |      | m_2 |      | ...
                    | 0   0   x_1  y_1 0 1 |      | m_3 |      | v_1 |
                    |          ...         |      | m_4 |      | ... |
                                                  | t_1 |
                                                  | t_2 |
    we use the denser form:
    A =  | x_1 y_1 1 |, M = | m_1 m_3 |, b = | u_1 v_1 |
         | x_2 y_2 1 |      | m_2 m_4 |      | u_2 v_2 |
                            | t_1 t_2 |
    */
    Mat A = Mat(static_cast<int>(originalPoints.size()), 2, CV_32FC1, originalPoints.data());
    cv::hconcat(A, Mat::ones(static_cast<int>(originalPoints.size()), 1, CV_32FC1), A);
    
    Mat b = Mat(static_cast<int>(transformedPoints.size()), 2, CV_32FC1, transformedPoints.data());
    
    cv::Mat A_t =  cv::Mat(A.cols, A.rows, CV_32FC1); 
    cv::transpose(A, A_t);
    
    Mat affineTransformation = (A_t * A).inv() * A_t * b;
    transpose(affineTransformation, affineTransformation);
    
    return affineTransformation;
}

//returns Mat CV_64FC1
Mat getAffineTransformation(Mat image, Mat transformedImage, AffineEstimator type, std::size_t numOfPoints/* = 3*/)
{
    std::vector<KeyPoint> imageKeypoints;
    std::vector<KeyPoint> transformedKeypoints;
    Mat imageDescriptors;
    Mat transformedDescriptors;
    
    static Ptr<xfeatures2d::SIFT> detectorExtractor = cv::xfeatures2d::SIFT::create();
    detectorExtractor->detectAndCompute(image, noArray(), imageKeypoints, imageDescriptors);
    detectorExtractor->detectAndCompute(transformedImage, noArray(), transformedKeypoints, transformedDescriptors);
    
    std::vector< DMatch > matches = matchDescriptors(imageDescriptors, transformedDescriptors);
    
    validateInformationGain(matches);
        
    std::vector<Point2f> originalPoints;
    std::vector<Point2f> transformedPoints;
    
    std::size_t maximumValidNumOfPoints;
    if(AffineEstimator::TRIANGULATION == type)
    {
        maximumValidNumOfPoints = 3;
    }
    else
    {
        maximumValidNumOfPoints = (numOfPoints > matches.size() ? matches.size() : numOfPoints);
    }
    
    std::sort(matches.begin(), matches.end(), [](DMatch a, DMatch b) { return a.distance < b.distance; });
    for (int i = 0; i < maximumValidNumOfPoints; ++i)
    {
        Point2f pt1 = imageKeypoints[matches[i].queryIdx].pt;
        Point2f pt2 = transformedKeypoints[matches[i].trainIdx].pt;
        originalPoints.push_back(pt1);
        transformedPoints.push_back(pt2);
    }
    
    if(AffineEstimator::TRIANGULATION == type)
    {
        return cv::getAffineTransform(originalPoints, transformedPoints);
    }
    else if(AffineEstimator::LEAST_SQUARES == type)
    {
        Mat affineEstimation = leastSquaresAffineEstimation(originalPoints, transformedPoints);
        affineEstimation.convertTo(affineEstimation, CV_64FC1); // to provide a uniform interface with the cv::getAffineTransform
        return affineEstimation;
    }
    else
    {
        throw std::domain_error("Not implemented affine estimator");
    }
}
