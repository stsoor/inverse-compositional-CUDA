#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

#include "types.h"


// building cv::Mat from (part of) a 1D array
cv::Mat buildImageMat(float* intensityArray, const std::size_t height, const std::size_t width);

// fills out a 1D array (W) with the unique 2x3 part of the warp matrix using the p parameter array
void initWarp(float* W, float* p);

//fills out a 1D array (out) based on the unique 2x3 part of a warp matrix stored in 1D (W)
void getWarpInvert(float* W, float* out);

// filling out a 1D array (W out) with the unique 2x3 part of the updated warp (W = W o W^-1)
void updateWarp(float* W, float* idW);

// bilinear interpolation
// image is a 1D array
__host__ __device__ float interpolate(float* image, std::size_t imageHeight, std::size_t imageWidth, float y, float x);

// L2 or frobenius norm
// m is a 1D array of a vector or matrix
float norm(float* m, std::size_t rows, std::size_t cols);

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
                    );

//gridDim.x = maximum template image height
//gridDim.y = active template number
//blockDim.x = maximum template image width
__global__ void precompute(dimensions* templateDimensions, float* gradientX, float* gradientY, float* steepestDescent, float* h);

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

    
	float*   W;                     // Current value of warp W(x,p)
    W = affineParameterEstimates;   // We use the input array directly
	float*  dW = new float[templateNum * 2 * 3];  // Warp update.
	float* idW = new float[templateNum * 2 * 3];  // Inverse warp.

	float* b = new float[templateNum * 6];                    // Vector in the right side of the system of linear equations. ([T'*(dW/dp)]^T*[I(W(p)) - T()])
	float* deltaP = new float[templateNum * 6];               // Parameter update value. (H^-1*b)

    // will need for gpu memory allocation
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
    
    // gpu memory allocations for calculating Hessian matrices
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
    
    // there is no memset for floats so we have to copy a zero-array to the gpu
    for(int i = 0; i < templateNum * 6 * 6; ++i)
    {
        hArray[i] = 0.0f;
    }
    
    cudaMemcpy(gpuH, hArray, hArraySize, cudaMemcpyHostToDevice);
    
    cudaMemcpy(gpuTemplateDimensions, dimensionsArray, dimensionsArraySize, cudaMemcpyHostToDevice);
    
        
	// calculate gradients (could do on gpu too) and pushing to gpu
    for(int i = 0; i < templateNum; ++i)
    {
        
        std::size_t templateImageHeight = dimensionsArray[i].rows;
        std::size_t templateImageWidth = dimensionsArray[i].cols;
        
        cv::Mat templateImageMat = buildImageMat(templateImageArray + dimensionsArray[i].position, templateImageHeight, templateImageWidth);
        
        gradientY = cv::Mat(templateImageHeight, templateImageWidth, CV_32FC1);
        gradientX = cv::Mat(templateImageHeight, templateImageWidth, CV_32FC1);
    
        cv::Scharr(templateImageMat, gradientY, -1, 0, 1, 1.0 / 32.0); // 1 / 32 to normalize results
        cv::Scharr(templateImageMat, gradientX, -1, 1, 0, 1.0 / 32.0);
        
        // copying to the same position where the template is in the templateImage
        cudaMemcpy(gpuGradientX + dimensionsArray[i].position, reinterpret_cast<float*>(gradientX.data), templateImageHeight * templateImageWidth * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuGradientY + dimensionsArray[i].position, reinterpret_cast<float*>(gradientY.data), templateImageHeight * templateImageWidth * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // calculating hessians
    precompute<<<dim3(maxTemplateRowSize, templateNum), maxTemplateColSize>>>(gpuTemplateDimensions, gpuGradientX, gpuGradientY, gpuSteepestDescentImage, gpuH);
    
    // copy back the hessians to gpu to invert them
    cudaMemcpy(hArray, gpuH, hArraySize, cudaMemcpyDeviceToHost);
    
    // we will only need the hessians (already on cpu) and the steepest images (still on gpu)
    cudaFree(gpuH);
    cudaFree(gpuGradientX);
    cudaFree(gpuGradientY);
    
    // invert all hessians
    std::size_t startPosition = 0;
    for(int i = 0; i < templateNum; ++i)
    {
        
        cv::Mat_<float> H(6,6);
        H = buildImageMat(hArray + startPosition, 6, 6);
        
        invertedHessians.push_back(H.inv());
        
        startPosition += 6 * 6;
    }
    
    // we copied all hessians already, no need for this array
    delete[] hArray;
    
    
	/*
	 *   Iteration stage.
	 */
    
    // copying to gpu everything that is needed for iteration stage
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
        
        // the first <activeTemplates> elements of dimensionsArray hold the relevant information/descriptor for the active templates
        // on the gpu we use these to descriptors to split work and decide where the necessary parts of each array are
        // this array changes in each iteration based on the active templates
        cudaMemcpy(gpuTemplateDimensions, dimensionsArray, activeTemplates * sizeof(dimensions), cudaMemcpyHostToDevice);
        
        // W changes in each iteration too (that is the purpose of the whole application)
        // we could copy only the relevant Ws but that would need a loop and a lot of communication so i guess copying the whole stuff is optimal
        cudaMemcpy(gpuW, W, wArraySize, cudaMemcpyHostToDevice);
        
        // still no memset for floats so we have to copy a zero-array
        for(int i = 0; i < templateNum * 6; ++i)
        {
            b[i] = 0.0f;
        }
        
        cudaMemcpy(gpuB, b, bArraySize, cudaMemcpyHostToDevice);
	    
        //calculating the b vectors (6x1) for each active template (b = [T'*(dW/dp)]^T*[I(W(p)) - T()] for each template)
        getB<<<dim3(maxTemplateRowSize, activeTemplates), maxTemplateColSize>>>(gpuTemplateDimensions, gpuImage, imageHeight, imageWidth, gpuTemplateImages, gpuSteepestDescentImage, gpuW, gpuB);

        //copying b vectors back to CPU
        // we could copy only the relevant bs but that would need a loop and a lot of communication so i guess copying the whole stuff is optimal
        cudaMemcpy(b, gpuB, bArraySize, cudaMemcpyDeviceToHost);
        
        //calculating delta_p vectors (delta_p = H^-1*b for each template)
        maxTemplateRowSize = 0;
        maxTemplateColSize = 0;
        for(int i = 0; i < activeTemplates; ++i)
        {
            //starting point of deltaP_i in deltaP
            startPosition = dimensionsArray[i].id * 6;
            
            //deltaP = iH * b;
            for(int row = 0; row < 6; ++row)
            {
                deltaP[startPosition + row] = 0.0f;
                for(int col = 0; col < 6; ++col)
                {
                    deltaP[startPosition + row] += invertedHessians[dimensionsArray[i].id].at<float>(row, col) * b[startPosition + col]; // can use the same startPosition for b because it has the same dimension (and order)
                }
            }
            
            initWarp(dW + startPosition, deltaP + startPosition); // can use the same startPosition for dW because it has the same dimension (2*3 vs 6 but both stored in 1D)
            
            // Invert warp.
            getWarpInvert(dW + startPosition, idW + startPosition);
            
            // W = W o idW;
            updateWarp(W + startPosition, idW + startPosition);
    
            // Check if current template finished
            if(norm(deltaP + startPosition, 6, 1) <= epsilon)
            {
                // if it finished then we swap its dimensions with the dimensions of the last active template
                // we could just simply overwrite the current one (and lose its information altogether - we won't need it anymore)
                // yaaay you found some optimization - i just couldn't get myself to delete poor dimensions eternally
                
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
                --i; // we need to check at the same position among dimensions again (the swapped value might finished too)
                // if this was the last one then because of the decreased activeTemplates the outer iteration will ncrement so don't worry
            }
            else
            {
                // if it did not finish then it is in the ballot of determining the thread number in the next iteration
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
    
    std::clog << "Finished in " << iter << " iterations." << std::endl << std::endl;
    
    // clean up everything
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

// building cv::Mat from (part of) a 1D array
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

// fills out a 1D array (W) with the unique 2x3 part of the warp matrix using the p parameter array
void initWarp(float* W, float* p)
{
	W[0 * 3 + 0] = 1.0 + p[0];
	W[0 * 3 + 1] = p[2];
	W[0 * 3 + 2] = p[4];
    
	W[1 * 3 + 0] = p[1];
	W[1 * 3 + 1] = 1.0 + p[3];
	W[1 * 3 + 2] = p[5];
}

//fills out a 1D array (out) based on the unique 2x3 part of a warp matrix stored in 1D (W)
void getWarpInvert(float* W, float* out)
{
    float p1 = W[0 * 3 + 0] - 1.0;
    float p2 = W[1 * 3 + 0];
    float p3 = W[0 * 3 + 1];
    float p4 = W[1 * 3 + 1] - 1.0;
    float p5 = W[0 * 3 + 2];
    float p6 = W[1 * 3 + 2];
    
    // for affine warps (using the same parametrization as in the baker-matthews unifying article) there is a closed form for inverting the warp
    float det = (1 + p1) * (1 + p4) - p2 * p3;
    
    out[0 * 3 + 0] = 1.0 + (-p1 - p1 * p4 + p2 * p3) / det;
    out[0 * 3 + 1] = (-p3) / det;
    out[0 * 3 + 2] = (-p5 - p4 * p5 + p3 * p6) / det;
    out[1 * 3 + 0] = (-p2) / det;
    out[1 * 3 + 1] = 1.0 + (-p4 - p1 * p4 + p2 * p3) / det;
    out[1 * 3 + 2] = (-p6 - p1 * p6 + p2 * p5) / det;
}

// filling out a 1D array (W out) with the unique 2x3 part of the updated warp (W = W o W^-1)
void updateWarp(float* W, float* idW)
{
    float  p[6] = {  W[0 * 3 + 0] - 1.0f,   W[1 * 3 + 0],   W[0 * 3 + 1],    W[1 * 3 + 1] - 1.0f,   W[0 * 3 + 2],   W[1 * 3 + 2]};
    float dp[6] = {idW[0 * 3 + 0] - 1.0f, idW[1 * 3 + 0], idW[0 * 3 + 1],  idW[1 * 3 + 1] - 1.0f, idW[0 * 3 + 2], idW[1 * 3 + 2]};
        
    // composition formula is provided in the baker-matthews unifying article and can be calculated by hand too
    float newP[6] = {
        p[0] + dp[0] + p[0] * dp[0] + p[2] * dp[1],
        p[1] + dp[1] + p[1] * dp[0] + p[3] * dp[1],
        p[2] + dp[2] + p[0] * dp[2] + p[2] * dp[3],
        p[3] + dp[3] + p[1] * dp[2] + p[3] * dp[3],
        p[4] + dp[4] + p[0] * dp[4] + p[2] * dp[5],
        p[5] + dp[5] + p[1] * dp[4] + p[3] * dp[5]
    };
    
    initWarp(W, newP); // creating the warp based on the new parameters
}

// bilinear interpolation
// image is a 1D array
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

// L2 or frobenius norm
// m is a 1D array of a vector or matrix
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
    
    /// we are using the maximum active height/width dimensions so it is possible that it is not relevant to this template
    if(templateDimensions[blockIdx.y].rows <= row || templateDimensions[blockIdx.y].cols <= col)
    {
        return;
    }
    
    
	// Set vector X with pixel coordinates (x,y,1)
    float X[3] = { static_cast<float>(col), static_cast<float>(row), 1.0f};

    std::size_t startPosition = templateDimensions[blockIdx.y].id * 2 * 3; // position at which the W of this template starts
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
        
        startPosition = templateDimensions[blockIdx.y].position; // position at which the template image starts
		// Calculate image difference D = I(W(x,p))-T(x).
		float D = I2 - templateImage[startPosition + row * templateImageWidth + col];

		// Add a term to b matrix.
        // steepest descent starts exactly at 6 times in the steepestDescent array where the templates start in the templateImage array (they are in the same order)
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
    
    // we are using the maximum height/width dimensions so it is possible that it is not relevant to this template 
    if(templateDimensions[blockIdx.y].rows <= row || templateDimensions[blockIdx.y].cols <= col)
    {
        return;
    }
    
    std::size_t startPosition = templateDimensions[blockIdx.y].position; // where the template image starts
    // Evaluate gradient of T.
    float Tx = gradientX[startPosition + row * templateImageWidth + col];
    float Ty = gradientY[startPosition + row * templateImageWidth + col];
    
    // Calculate steepest descent image's element. (Each element attained by multiplying the gradient row with the jacobi matrix)
    // steepest descent starts exactly at 6 times in the steepestDescent array where the templates start in the templateImage array (they are in the same order)
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
            // everything is in the order of ids so h of T_i is starting at id*6*6
            atomicAdd(&(h[templateDimensions[blockIdx.y].id*6*6 + i * 6 + j]), steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + i] * steepestDescent[startPosition*6 + 6 * (row * templateImageWidth + col) + j]);
        }
    }
}
