#include <helper_math.h>
#include <helper_functions.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

#include "types.h"


cv::Mat buildImageMat(float* intensityArray, const std::size_t height, const std::size_t width)
{
    cv::Mat_<float> image(static_cast<int>(height), static_cast<int>(width));
    
    for(std::size_t row = 0; row != height; ++row)
    {
        for(std::size_t col = 0; col != width; ++col)
        {
             image.at<float>(static_cast<int>(row), static_cast<int>(col)) = intensityArray[row * width + col];
        }   
    }
    
    return image;
}

void initWarp(float* W, float* p)
{
	W[0 * 3 + 0] = 1.0 + p[0];
	W[0 * 3 + 1] = p[2];
	W[0 * 3 + 2] = p[4];
    
	W[1 * 3 + 0] = p[1];
	W[1 * 3 + 1] = 1.0 + p[3];
	W[1 * 3 + 2] = p[5];
}

void getWarpInvert(float* W, float* out)
{
    float p1 = W[0 * 3 + 0] - 1.0;
    float p2 = W[1 * 3 + 0];
    float p3 = W[0 * 3 + 1];
    float p4 = W[1 * 3 + 1] - 1.0;
    float p5 = W[0 * 3 + 2];
    float p6 = W[1 * 3 + 2];
    
    float det = (1 + p1) * (1 + p4) - p2 * p3;
    
    out[0 * 3 + 0] = 1.0 + (-p1 - p1 * p4 + p2 * p3) / det;
    out[0 * 3 + 1] = (-p3) / det;
    out[0 * 3 + 2] = (-p5 - p4 * p5 + p3 * p6) / det;
    out[1 * 3 + 0] = (-p2) / det;
    out[1 * 3 + 1] = 1.0 + (-p4 - p1 * p4 + p2 * p3) / det;
    out[1 * 3 + 2] = (-p6 - p1 * p6 + p2 * p5) / det;
}

void updateWarp(float* W, float* idW)
{
    float  p[6] = {  W[0 * 3 + 0] - 1.0f,   W[1 * 3 + 0],   W[0 * 3 + 1],    W[1 * 3 + 1] - 1.0f,   W[0 * 3 + 2],   W[1 * 3 + 2]};
    float dp[6] = {idW[0 * 3 + 0] - 1.0f, idW[1 * 3 + 0], idW[0 * 3 + 1],  idW[1 * 3 + 1] - 1.0f, idW[0 * 3 + 2], idW[1 * 3 + 2]};
        
        
    float newP[6] = {
        p[0] + dp[0] + p[0] * dp[0] + p[2] * dp[1],
        p[1] + dp[1] + p[1] * dp[0] + p[3] * dp[1],
        p[2] + dp[2] + p[0] * dp[2] + p[2] * dp[3],
        p[3] + dp[3] + p[1] * dp[2] + p[3] * dp[3],
        p[4] + dp[4] + p[0] * dp[4] + p[2] * dp[5],
        p[5] + dp[5] + p[1] * dp[4] + p[3] * dp[5]
    };
    
    initWarp(W, newP);
}

__host__ __device__ float interpolate(float* image, std::size_t imageHeight, std::size_t imageWidth, float y, float x)
{
  float xd, yd;  
  float k1 = modff(x,&xd);
  float k2 = modff(y,&yd);
  int xi = int(xd);
  int yi = int(yd);

  int f1 = xi < imageHeight-1;  // Check that pixels to the right  
  int f2 = yi < imageWidth-1; // and to down direction exist.

  float px1 = image[( yi ) * imageWidth + xi];
  float px2 = image[( yi ) * imageWidth + xi+1];
  float px3 = image[(yi+1) * imageWidth + xi];
  float px4 = image[(yi+1) * imageWidth + xi+1];      
  
  // Interpolate pixel intensity.
  float interpolated_value = 
        (1.0-k1)*(1.0-k2)*px1 +
  (     f1     ? ( k1*(1.0-k2)*px2 ) : 0) +
  (     f2     ? ( (1.0-k1)*k2*px3 ) : 0) + 
  ( (f1 && f2) ? ( k1*k2*px4 ) : 0);

  return interpolated_value;
}

float norm(float* m, std::size_t rows, std::size_t cols)
{
    float squareSum = 0.0f;
    
    for(std::size_t row = 0; row < rows; ++row)
    {
        for(std::size_t col = 0; col < cols; ++col)
        {
            float elem = m[row * cols + col];
            
            squareSum += elem * elem;
        }
    }
    
    return sqrt(squareSum);
}

//gridDim.x = maximum template image height
//gridDim.y = active template number
//blockDim.x = maximum template image width
__global__ void getB(
                      dimensions* templateDimensions,
                      float* image,
                      const unsigned int imageHeight,
                      const unsigned int imageWidth,
                      float* templateImage,
                      float* steepestDescent,
                      float* W,
                      float* b
                    )
{
    std::size_t row = blockIdx.x;
    std::size_t col = threadIdx.x;
    
    std::size_t templateImageWidth = templateDimensions[blockIdx.y].cols;
    
    if(templateDimensions[blockIdx.y].rows <= row || templateDimensions[blockIdx.y].cols <= col)
    {
        return;
    }
    
    
	// Set vector X with pixel coordinates (x,y,1)
    float X[3] = { static_cast<float>(col), static_cast<float>(row), 1.0f};

    std::size_t startPosition = templateDimensions[blockIdx.y].id * 2 * 3;
	// Warp Z=W*X
    float Z[3] = { W[startPosition + 0 * 3 + 0] * X[0] + W[startPosition + 0 * 3 + 1] * X[1] + W[startPosition + 0 * 3 + 2] * X[2],
                   W[startPosition + 1 * 3 + 0] * X[0] + W[startPosition + 1 * 3 + 1] * X[1] + W[startPosition + 1 * 3 + 2] * X[2],
                                              0 * X[0] +                                   0 * X[1] +                     1 * X[2]
                 };

    // pixel coordinates in the coordinate frame of I.
    float col2 = Z[0];
    float row2 = Z[1];

	// Get the nearest integer pixel coords (x2i;y2i).
	int col2i = int(floor(col2));
	int row2i = int(floor(row2));

	if(col2i >= 0 && col2i < imageWidth && // check if pixel is inside I.
		row2i >= 0 && row2i < imageHeight)
	{
		// Calculate intensity of a transformed pixel with sub-pixel accuracy
		// using bilinear interpolation.
		float I2 = interpolate(image, imageHeight, imageWidth, row2, col2);
        
        startPosition = templateDimensions[blockIdx.y].position;
		// Calculate image difference D = I(W(x,p))-T(x).
		float D = I2 - templateImage[startPosition + row * templateImageWidth + col];

		// Add a term to b matrix.
        atomicAdd(&(b[templateDimensions[blockIdx.y].id*6 + 0]), steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 0] * D);
        atomicAdd(&(b[templateDimensions[blockIdx.y].id*6 + 1]), steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 1] * D);
        atomicAdd(&(b[templateDimensions[blockIdx.y].id*6 + 2]), steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 2] * D);
        atomicAdd(&(b[templateDimensions[blockIdx.y].id*6 + 3]), steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 3] * D);
        atomicAdd(&(b[templateDimensions[blockIdx.y].id*6 + 4]), steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 4] * D);
        atomicAdd(&(b[templateDimensions[blockIdx.y].id*6 + 5]), steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 5] * D);
	}
}

//gridDim.x = maximum template image height
//gridDim.y = active template number
//blockDim.x = maximum template image width
__global__ void precompute(dimensions* templateDimensions, float* gradientX, float* gradientY, float* steepestDescent, float* h)
{
    std::size_t row = blockIdx.x;
    std::size_t col = threadIdx.x;
    
    std::size_t templateImageWidth = templateDimensions[blockIdx.y].cols;
    
    if(templateDimensions[blockIdx.y].rows <= row || templateDimensions[blockIdx.y].cols <= col)
    {
        return;
    }
    
    std::size_t startPosition = templateDimensions[blockIdx.y].position;
    // Evaluate gradient of T.
    float Tx = gradientX[startPosition + row * templateImageWidth + col];
    float Ty = gradientY[startPosition + row * templateImageWidth + col];
    
    // Calculate steepest descent image's element.
    steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 0] = Tx * col;
    steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 1] = Ty * col;
    steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 2] = Tx * row;
    steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 3] = Ty * row;
    steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 4] = Tx;
    steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + 5] = Ty;
    
    // Add a term to Hessian.
    for(int i = 0; i < 6; ++i)
    {
        for(int j = 0; j < 6; ++j)
        {
            atomicAdd(&(h[templateDimensions[blockIdx.y].id *6*6 + i * 6 + j]), steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + i] * steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + j]);
        }
    }
}

extern "C"
void inverseCompositional( float* imageArray
                         , float* templateImageArray
                         , float* affineParameterEstimates
                         , const std::size_t imageHeight
                         , const std::size_t imageWidth
                         , dimensions* dimensionsArray
                         , const std::size_t templateNum
                         , const float epsilon
                         , const int maxIteration
                         )
{
	cv::Mat gradientY;   // Gradient of T in X direction.
	cv::Mat gradientX;   // Gradient of T in Y direction.

									// Here we will store matrices.
	float*   W;                     // Current value of warp W(x,p)
    W = affineParameterEstimates;   // We use the input array directly
	float*  dW = new float[templateNum * 2 * 3];  // Warp update.
	float* idW = new float[templateNum * 2 * 3];  // Inverse warp.

	float* b = new float[templateNum * 6];                     // Vector in the right side of the system of linear equations.
	float* deltaP = new float[templateNum * 6];               // Parameter update value.

    
    std::size_t templateSizeSum = 0;
    std::size_t maxTemplateRowSize = 0;
    std::size_t maxTemplateColSize = 0;
    for(int i = 0; i < templateNum; ++i)
    {
        templateSizeSum += dimensionsArray[i].rows * dimensionsArray[i].cols;
        
        if(maxTemplateRowSize < dimensionsArray[i].rows)
        {
            maxTemplateRowSize = dimensionsArray[i].rows;
        }
        if(maxTemplateColSize < dimensionsArray[i].cols)
        {
            maxTemplateColSize = dimensionsArray[i].cols;
        }
    }
    
    float* steepestDescent = new float[templateSizeSum * 6];
    
    std::size_t templateImageArraySize = templateSizeSum * sizeof(float);
    std::size_t steepestDescentArraySize = templateSizeSum * 6 * sizeof(float);
    std::size_t dimensionsArraySize = templateNum * sizeof(dimensions);
    std::size_t hArraySize = templateNum * 6 * 6 * sizeof(float);
    
  
    float* gpuH = NULL;
    cudaMalloc((void **)&gpuH, hArraySize);
    float* gpuSteepestDescentImage = NULL;
    cudaMalloc((void **)&gpuSteepestDescentImage, steepestDescentArraySize);
    dimensions* gpuTemplateDimensions = NULL;
    cudaMalloc((void **)&gpuTemplateDimensions, dimensionsArraySize);
    float* gpuGradientX = NULL;
    cudaMalloc((void **)&gpuGradientX, templateImageArraySize);
    float* gpuGradientY = NULL;
    cudaMalloc((void **)&gpuGradientY, templateImageArraySize);
    
    std::vector<cv::Mat> invertedHessians;
    float* hArray = new float[templateNum * 6*6];
    
    for(int i = 0; i < templateNum * 6 * 6; ++i)
    {
        hArray[i] = 0.0f;
    }
    
    cudaMemcpy(gpuH, hArray, hArraySize, cudaMemcpyHostToDevice);
    
    cudaMemcpy(gpuTemplateDimensions, dimensionsArray, dimensionsArraySize, cudaMemcpyHostToDevice);
    
    for(int i = 0; i < templateNum; ++i)
    {
        
        std::size_t templateImageHeight = dimensionsArray[i].rows;
        std::size_t templateImageWidth = dimensionsArray[i].cols;
        
        cv::Mat templateImageMat = buildImageMat(templateImageArray + dimensionsArray[i].position, templateImageHeight, templateImageWidth);
        
		// Create images.
        gradientY = cv::Mat(templateImageHeight, templateImageWidth, CV_32FC1);
        gradientX = cv::Mat(templateImageHeight, templateImageWidth, CV_32FC1);
    
        cv::Scharr(templateImageMat, gradientY, -1, 0, 1, 1.0 / 32.0); // 1 / 32 to normalize results
        cv::Scharr(templateImageMat, gradientX, -1, 1, 0, 1.0 / 32.0);
        
        cudaMemcpy(gpuGradientX + dimensionsArray[i].position, reinterpret_cast<float*>(gradientX.data), templateImageHeight * templateImageWidth * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuGradientY + dimensionsArray[i].position, reinterpret_cast<float*>(gradientY.data), templateImageHeight * templateImageWidth * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    precompute<<<dim3(maxTemplateRowSize, templateNum), maxTemplateColSize>>>(gpuTemplateDimensions, gpuGradientX, gpuGradientY, gpuSteepestDescentImage, gpuH);
        
    cudaMemcpy(hArray, gpuH, hArraySize, cudaMemcpyDeviceToHost);
    
    cudaFree(gpuH);
    cudaFree(gpuGradientX);
    cudaFree(gpuGradientY);
    
    std::size_t startPosition = 0;
    for(int i = 0; i < templateNum; ++i)
    {
        
        cv::Mat_<float> H(6,6);
        H = buildImageMat(hArray + startPosition, 6, 6);
        
        invertedHessians.push_back(H.inv());
        
        startPosition += 6 * 6;
    }
    
        
    delete[] hArray;
    
    
	/*
	 *   Iteration stage.
	 */
    
    // copy images, estimate to gpu memory
    std::size_t imageArraySize = imageWidth * imageHeight * sizeof(float);
    std::size_t wArraySize = templateNum * 2 * 3 * sizeof(float);
    std::size_t bArraySize = templateNum * 6 * 1 * sizeof(float);
  
    float* gpuImage = NULL;
    cudaMalloc((void **)&gpuImage, imageArraySize);
    float* gpuTemplateImages = NULL;
    cudaMalloc((void **)&gpuTemplateImages, templateImageArraySize);
    float* gpuW = NULL;
    cudaMalloc((void **)&gpuW, wArraySize);
    float* gpuB = NULL;
    cudaMalloc((void **)&gpuB, bArraySize);
    
    cudaMemcpy(gpuImage, imageArray, imageArraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuTemplateImages, templateImageArray, templateImageArraySize, cudaMemcpyHostToDevice);

    std::size_t activeTemplates = templateNum; // templates for which the algorithm has not finished yet
    
	int iter = 0;
    while(iter < maxIteration && 0 < activeTemplates)
	{
		++iter;
        
        cudaMemcpy(gpuTemplateDimensions, dimensionsArray, activeTemplates * sizeof(dimensions), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuW, W, wArraySize, cudaMemcpyHostToDevice);
        
        for(int i = 0; i < templateNum * 6; ++i)
        {
            b[i] = 0.0f;
        }
        
        cudaMemcpy(gpuB, b, bArraySize, cudaMemcpyHostToDevice);
			
        getB<<<dim3(maxTemplateRowSize, activeTemplates), maxTemplateColSize>>>(gpuTemplateDimensions, gpuImage, imageHeight, imageWidth, gpuTemplateImages, gpuSteepestDescentImage, gpuW, gpuB);

        cudaMemcpy(b, gpuB, bArraySize, cudaMemcpyDeviceToHost);
        
        maxTemplateRowSize = 0;
        maxTemplateColSize = 0;
        for(int i = 0; i < activeTemplates; ++i)
        {
            startPosition = dimensionsArray[i].id * 6;
            
            //deltaP = iH * b;
            for(int row = 0; row < 6; ++row)
            {
                deltaP[startPosition + row] = 0.0f;
                for(int col = 0; col < 6; ++col)
                {
                    deltaP[startPosition + row] += invertedHessians[dimensionsArray[i].id].at<float>(row, col) * b[startPosition + col]; // can use the same startPosition for b because it has the same dimension
                }
            }
            
            initWarp(dW + startPosition, deltaP + startPosition); // can use the same startPosition for dW because it has the same dimension (2*3 vs 6 but both stored in 1D)
            
            // Invert warp.
            getWarpInvert(dW + startPosition, idW + startPosition);
            
            //W o idW;
            updateWarp(W + startPosition, idW + startPosition);
    
            // Check termination critera.
            if(norm(deltaP + startPosition, 6, 1) <= epsilon)
            {
                dimensions tmp;
                tmp.id       = dimensionsArray[i].id;
                tmp.position = dimensionsArray[i].position;
                tmp.rows     = dimensionsArray[i].rows;
                tmp.cols     = dimensionsArray[i].cols;
                
                dimensionsArray[i].id       = dimensionsArray[activeTemplates - 1].id;
                dimensionsArray[i].position = dimensionsArray[activeTemplates - 1].position;
                dimensionsArray[i].rows     = dimensionsArray[activeTemplates - 1].rows;
                dimensionsArray[i].cols     = dimensionsArray[activeTemplates - 1].cols;
                
                dimensionsArray[activeTemplates - 1].id       = tmp.id;
                dimensionsArray[activeTemplates - 1].position = tmp.position;
                dimensionsArray[activeTemplates - 1].rows     = tmp.rows;
                dimensionsArray[activeTemplates - 1].cols     = tmp.cols;
                
                --activeTemplates;
                --i; //check at the same position again
            }
            else
            {
                if(maxTemplateRowSize < dimensionsArray[i].rows)
                {
                    maxTemplateRowSize = dimensionsArray[i].rows;
                }
                if(maxTemplateColSize < dimensionsArray[i].cols)
                {
                    maxTemplateColSize = dimensionsArray[i].cols;
                }
            }
        }
	}
    
    std::cout << "Finished in " << iter << " iterations." << std::endl << std::endl;
    
    delete[] dW;
    delete[] idW;
    //for W we use the input array directly
    
	delete[] b;
	delete[] deltaP;
    
    cudaFree(gpuB);
    cudaFree(gpuW);
    cudaFree(gpuImage);
    cudaFree(gpuSteepestDescentImage);
    cudaFree(gpuTemplateImages);
    cudaFree(gpuTemplateDimensions);
}
