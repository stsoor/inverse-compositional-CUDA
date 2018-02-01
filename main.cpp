//cl /EHsc /W4 /Fealign.exe getAffine.cpp main.cpp /I "C:\Libs\OpenCV\OpenCV3.3.0\include" /link /LIBPATH:"C:\Libs\OpenCV\OpenCV3.3.0\x64\vc15\lib\" opencv_world330.lib

#include "getAffine.h"

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
                                    , const std::size_t templateImageHeight
                                    , const std::size_t templateImageWidth
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
    std::cout << argv[1] << std::endl;
    
    //if(image.type() != CV_8UC1 || templateImage.type() != CV_8UC1) -> we could throw an error if we would only implement for 8bit images
    
    float* imageArray = new float[input.image.rows * input.image.cols];
    float* templateImageArray = new float[input.templates[0].rows * input.templates[0].cols];
    float* affineParameterEstimates = new float[2 * 3];
    
    for(int y = 0; y < input.image.rows; ++y)
    {
        for(int x = 0; x < input.image.cols; ++x)
        {
            imageArray[y * input.image.cols + x] = input.image.at<unsigned char>(y, x) / 255.0f;
        }
    }
    for(int y = 0; y < input.templates[0].rows; ++y)
    {
        for(int x = 0; x < input.templates[0].cols; ++x)
        {
            templateImageArray[y * input.templates[0].cols + x] = input.templates[0].at<unsigned char>(y, x) / 255.0f;
        }
    }
    
    
      for(int y = 0; y < input.affineEstimates[0].rows; ++y)
    {
        for(int x = 0; x < input.affineEstimates[0].cols; ++x)
        {
            affineParameterEstimates[y * 3 + x] = static_cast<float>(input.affineEstimates[0].at<double>(y, x));
        }
    }
    
    
    //gpu code
    inverseCompositional( imageArray
                        , templateImageArray
                        , affineParameterEstimates
                        , input.image.rows
                        , input.image.cols
                        , input.templates[0].rows
                        , input.templates[0].cols
                        , input.epsilon
                        , input.maxIteration
                        );
    
    delete[] imageArray;
    delete[] templateImageArray;
    delete[] affineParameterEstimates;
    

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
    cv::Mat image = cv::imread( filePath, CV_LOAD_IMAGE_GRAYSCALE ); //TODO CV_LOAD_IMAGE_ANYDEPTH és convertTo float
    
    if( !image.data )
    {
        throw std::ios_base::failure("Runtime error during reading in the initial image");
    }
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
    std::cout << templateNum << std::endl;
    
    input.templates.reserve(templateNum);
    input.affineEstimates.reserve(templateNum);
    
    file >> input.epsilon;
    file >> input.maxIteration;
    std::cout << input.epsilon << std::endl;
    std::cout << input.maxIteration << std::endl;
    
    file.get(); //next char is a space
    std::string imageFilePath;
    getline(file, imageFilePath);
    std::cout << imageFilePath << std::endl;
    
    input.image = readImage(imageFilePath.c_str());
    
    for(int i = 0; i < templateNum; ++i)
    {
        int mode;
        file >> mode;
    std::cout << mode << std::endl;
        
        if(1 == mode)
        {
            std::size_t estimationMatchNum;
            file >> estimationMatchNum;
    std::cout << estimationMatchNum << std::endl;
            
            file.get(); //next char is a space
            getline(file, imageFilePath);
            input.templates.push_back(readImage(imageFilePath.c_str()));
    std::cout << imageFilePath << std::endl;
            
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
