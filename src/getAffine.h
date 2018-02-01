#ifndef GETAFFINE_H
#define GETAFFINE_H

#include <vector>
#include <opencv2/opencv.hpp>

enum class AffineEstimator { TRIANGULATION, LEAST_SQUARES };


std::vector< cv::DMatch > matchDescriptors( cv::Mat descriptors1, cv::Mat descriptors2);

void validateInformationGain(std::vector< cv::DMatch >& matches);

cv::Mat leastSquaresAffineEstimation(std::vector<cv::Point2f>& originalPoints, std::vector<cv::Point2f>& transformedPoints);

cv::Mat getAffineTransformation(cv::Mat image, cv::Mat transformedImage, AffineEstimator type, std::size_t numOfPoints = 3); //returns Mat CV_64FC1

#endif //GETAFFINE_H
