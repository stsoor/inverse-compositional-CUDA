#include "getAffine.h"
#include "types.h"

#include <iostream>
#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <exception>

#include <vector>
#include <fstream>

struct inputType
{
    cv::Mat image;
    std::vector<cv::Mat> templates;
    std::vector<cv::Mat> affineEstimates;
    float epsilon;
    int maxIteration;
};

void printManual();

bool validateInput(int argc);

cv::Mat readImage(const char* filePath);

cv::Mat getRegressionAffineEstimate(cv::Mat templateImage, cv::Mat image, std::size_t usedPoints);

cv::Mat getTriangulationAffineEstimate(cv::Mat templateImage, cv::Mat image);

inputType readInput(const char* filePath);

extern "C" void inverseCompositional( float* imageArray
                                    , float* templateImageArray
                                    , float* affineParameterEstimates
                                    , const std::size_t imageHeight
                                    , const std::size_t imageWidth
                                    , dimensions* dimensionsArray
                                    , const std::size_t templateNum
                                    , const float epsilon
                                    , const int maxIteration
                                    );

int main( int argc, char** argv )
{
    if( !validateInput(argc) )
    {
        return -1;
    }
    
    inputType input = readInput(argv[1]);
    
    std::size_t templateNum = input.templates.size();
    
    std::size_t templateSizeSum = 0;
    for(auto& t : input.templates)
    {
        templateSizeSum += t.rows * t.cols;
    }
    
    float* imageArray = new float[input.image.rows * input.image.cols];
    float* templateImageArray = new float[templateSizeSum];
    float* affineParameterEstimates = new float[2 * 3 * templateNum];
    dimensions* dimensionsArray = new dimensions[templateNum];
    
    for(int y = 0; y < input.image.rows; ++y)
    {
        for(int x = 0; x < input.image.cols; ++x)
        {
            imageArray[y * input.image.cols + x] = input.image.at<unsigned char>(y, x) / 255.0f;
        }
    }
    
    std::size_t globalPosition = 0;
    for(int i = 0; i < templateNum; ++i)
    {
        dimensionsArray[i].id = i;
        dimensionsArray[i].position = globalPosition;
        dimensionsArray[i].rows = input.templates[i].rows;
        dimensionsArray[i].cols = input.templates[i].cols;
        
        for(int y = 0; y < input.templates[i].rows; ++y)
        {
            for(int x = 0; x < input.templates[i].cols; ++x)
            {
                templateImageArray[globalPosition++] = input.templates[i].at<unsigned char>(y, x) / 255.0f;
            }
        }
    }
    
    globalPosition = 0;
    for(int i = 0; i < templateNum; ++i)
    {
        for(int y = 0; y < input.affineEstimates[i].rows; ++y)
        {
            for(int x = 0; x < input.affineEstimates[i].cols; ++x)
            {
                affineParameterEstimates[globalPosition++] = static_cast<float>(input.affineEstimates[i].at<double>(y, x));
            }
        }
    }
    
    
    //gpu code
    inverseCompositional( imageArray
                        , templateImageArray
                        , affineParameterEstimates
                        , input.image.rows
                        , input.image.cols
                        , dimensionsArray
                        , templateNum
                        , input.epsilon
                        , input.maxIteration
                        );
    
    delete[] imageArray;
    delete[] templateImageArray;
    delete[] affineParameterEstimates;
    delete[] dimensionsArray;

    return 0;
}

void printManual()
{
    std::cout << "usage:" << std::endl;
    std::cout << " - align.exe input file" << std::endl;
}

bool validateInput(int argc)
{
    if( !(2 == argc) )
    {
        printManual();
        return false;
    }
    return true;
}

cv::Mat readImage(const char* filePath)
{
    cv::Mat image = cv::imread( filePath, CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
    
    if( !image.data )
    {
        throw std::ios_base::failure("Runtime error during reading in the initial image");
    }
    
    image.convertTo(image, CV_8UC1); // 8UC1 is needed for SIFT detection / parameter estimation and a uniform container is needed for later conversions
    
    return image;
}

cv::Mat getRegressionAffineEstimate(cv::Mat templateImage, cv::Mat image, std::size_t usedPoints)
{
    return getAffineTransformation(templateImage, image, AffineEstimator::LEAST_SQUARES, usedPoints); 
}

cv::Mat getTriangulationAffineEstimate(cv::Mat templateImage, cv::Mat image)
{
    return getAffineTransformation(templateImage, image, AffineEstimator::TRIANGULATION); 
}

inputType readInput(const char* filePath)
{
    inputType input;
    
    std::ifstream file; 
    file.open(filePath);
    
    int templateNum;
    file >> templateNum;
    
    input.templates.reserve(templateNum);
    input.affineEstimates.reserve(templateNum);
    
    file >> input.epsilon;
    file >> input.maxIteration;
    
    file.get(); //next char is a space
    std::string imageFilePath;
    getline(file, imageFilePath);
    
    input.image = readImage(imageFilePath.c_str());
    
    for(int i = 0; i < templateNum; ++i)
    {
        int mode;
        file >> mode;
        
        if(1 == mode)
        {
            std::size_t estimationMatchNum;
            file >> estimationMatchNum;
            
            file.get(); //next char is a space
            getline(file, imageFilePath);
            input.templates.push_back(readImage(imageFilePath.c_str()));
            
            input.affineEstimates.push_back(getRegressionAffineEstimate(input.templates.back(), input.image, estimationMatchNum));
        }
        else if(2 == mode)
        {
            file.get(); //next char is a space
            getline(file, imageFilePath);
            input.templates.push_back(readImage(imageFilePath.c_str()));
            
            input.affineEstimates.push_back(getTriangulationAffineEstimate(input.templates.back(), input.image));
        }
        else if(3 == mode)
        {
            cv::Mat_<double> estimate(2,3);
            
            file >> estimate.at<double>(0,0);
            file >> estimate.at<double>(0,1);
            file >> estimate.at<double>(0,2);
            file >> estimate.at<double>(1,0);
            file >> estimate.at<double>(1,1);
            file >> estimate.at<double>(1,2);
            
            input.affineEstimates.push_back(estimate);
            
            file.get(); //next char is a space
            getline(file, imageFilePath);
            input.templates.push_back(readImage(imageFilePath.c_str()));
        }
        else
        {
            std::runtime_error("Encountered unsupported mode during input file handling.");
        }
    }
    
    file.close();
    
    return input;
}
