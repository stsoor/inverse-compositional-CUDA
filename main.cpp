//cl /EHsc /W4 /Fealign.exe getAffine.cpp main.cpp /I "C:\Libs\OpenCV\OpenCV3.3.0\include" /link /LIBPATH:"C:\Libs\OpenCV\OpenCV3.3.0\x64\vc15\lib\" opencv_world330.lib

#include "getAffine.h"

#include <iostream>
#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <exception>

using namespace cv;

void printManual();

bool validateInput(int argc);

Mat readImage(const char* filePath);

Mat convertTransformationArguments(char** argv);

cv::Mat createTemplate(cv::Mat image, char** argv);

cv::Mat applyAffine(cv::Mat image, char** argv);

Rect convertRoiArgmunets(char** argv);

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
    
    Mat image;
    Mat templateImage;
    
    image = readImage(argv[1]);
    if(5 == argc)
    {
        templateImage = readImage(argv[2]);
    }
    else if(14 == argc)
    {
        templateImage = createTemplate(image, argv);
        image = applyAffine(image, argv);
    }
    else
    {
        std::cout << "Wrong number of arguments." << std::endl;
        exit(0);
    }
    
    //TRIANGULATION or LEAST_SQUARES
    Mat affineEstimate = getAffineTransformation(templateImage, image, AffineEstimator::LEAST_SQUARES, 7); // PARAMETER
    
    std::cout << "initial affine estimate: " << std::endl << affineEstimate << std::endl << std::endl;
    
    //if(image.type() != CV_8UC1 || templateImage.type() != CV_8UC1) -> we could throw an error if we would only implement for 8bit images
    
    float* imageArray = new float[image.rows * image.cols];
    float* templateImageArray = new float[templateImage.rows * templateImage.cols];
    float* affineParameterEstimates = new float[2 * 3];
    
    for(int y = 0; y < image.rows; ++y)
    {
      for(int x = 0; x < image.cols; ++x)
        imageArray[y * image.cols + x] = image.at<unsigned char>(y, x) / 255.0f;
    }
    for(int y = 0; y < templateImage.rows; ++y)
    {
      for(int x = 0; x < templateImage.cols; ++x)
        templateImageArray[y * templateImage.cols + x] = templateImage.at<unsigned char>(y, x) / 255.0f;
    }
    
    for(int x = 0; x < affineEstimate.cols; ++x)
    {
      for(int y = 0; y < affineEstimate.rows; ++y)
        affineParameterEstimates[x * 2 + y] = static_cast<float>(affineEstimate.at<double>(y, x));
    }
    //affineParameterEstimates[0] -= 1.0;
    //affineParameterEstimates[3] -= 1.0;
    
    
    //gpu code
    inverseCompositional( imageArray
                        , templateImageArray
                        , affineParameterEstimates
                        , image.rows
                        , image.cols
                        , templateImage.rows
                        , templateImage.cols
                        , static_cast<float>(atof(argv[argc-2])) // epsilon PARAMETER
                        , atoi(argv[argc-1]) // max iteration PARAMETER
                        );
    waitKey(0);
    
    delete[] imageArray;
    delete[] templateImageArray;
    delete[] affineParameterEstimates;
    

    return 0;
}

void printManual()
{
    std::cout << "use one of the following options:" << std::endl;
    std::cout << " - align.exe image template" << std::endl;
    std::cout << " - align.exe image p1 p2 t1 p3 p4 t2  roi_x roi_y roi_width roi_height (parameters for an affine transformation and the template roi)" << std::endl;
}

bool validateInput(int argc)
{
    if( !(5 == argc || 14 == argc) )
    {
        printManual();
        return false;
    }
    return true;
}

Mat readImage(const char* filePath)
{
    Mat image = imread( filePath, CV_LOAD_IMAGE_GRAYSCALE );
    
    if( !image.data )
    {
        throw std::ios_base::failure("Runtime error during reading in the initial image");
    }
    return image;
}

Mat convertTransformationArguments(char** argv)
{
    Mat argumentMatrix(2,3, CV_64FC1);
    for(int i = 0; i < 6; ++i)
    {
        argumentMatrix.at<double>(i / 3, i % 3) = atof(argv[i + 2]); //first element is name of exe and path to image
    }
    return argumentMatrix;
}

Mat createTemplate(Mat image, char** argv)
{
    Rect roi = convertRoiArgmunets(argv);
    Mat templateImage;
    image(roi).copyTo(templateImage);
    namedWindow("template"); //TODO delete
    imshow("template", templateImage); //TODO delete
    return templateImage;
}

Mat applyAffine(Mat image, char** argv)
{
    Mat transformation = convertTransformationArguments(argv);
    Mat transformed;
    warpAffine(image, transformed, transformation, Size(image.cols, image.rows));
    namedWindow("transformed"); //TODO delete
    imshow("transformed", transformed); //TODO delete
    return transformed;
}

Rect convertRoiArgmunets(char** argv)
{
    cv::Rect roi;
    roi.x = atoi(argv[8]);
    roi.y = atoi(argv[9]);
    roi.width = atoi(argv[10]);
    roi.height = atoi(argv[11]);
    return roi;
}