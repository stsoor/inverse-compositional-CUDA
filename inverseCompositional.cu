#include <helper_math.h>
#include <helper_functions.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <vector>


#define HUBER_LOSS 0.80 //Huber loss function parameter, to redunce the influence of outliers
#define sobelKernelSize 3

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

void buildTransformationFromInput(float* W, float* p)
{
    W[0 * 3 + 0] = p[0];
    W[0 * 3 + 1] = p[2];
    W[0 * 3 + 2] = p[4];
    W[1 * 3 + 0] = p[1];
    W[1 * 3 + 1] = p[3];
    W[1 * 3 + 2] = p[5];
    W[2 * 3 + 0] = 0.0f;
    W[2 * 3 + 1] = 0.0f;
    W[2 * 3 + 2] = 1.0f;
}

void init_warp(float* W, float* p)
{
	W[0 * 3 + 0] = 1.0 + p[0];
	W[0 * 3 + 1] = p[2];
	W[0 * 3 + 2] = p[4];
    
	W[1 * 3 + 0] = p[1];
	W[1 * 3 + 1] = 1.0 + p[3];
	W[1 * 3 + 2] = p[5];
    
	W[2 * 3 + 0] = 0.0;
	W[2 * 3 + 1] = 0.0;
	W[2 * 3 + 2] = 1.0;
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
    out[2 * 3 + 0] = 0.0;
    out[2 * 3 + 1] = 0.0;
    out[2 * 3 + 2] = 1.0;
}

void update_warp(float* W, float* idW)
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
    
    init_warp(W, newP);
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

//gridDim.x = template image height
//blockDim.x = template image width
__global__ void getB(
                      float* image,
                      const unsigned int imageHeight,
                      const unsigned int imageWidth,
                      float* templateImage,
                      float* steepestDescent,
                      float* W,
                      float* b
                    )
{
    //int pixelId = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    int templateImageWidth = blockDim.x;
    
	// Set vector X with pixel coordinates (x,y,1)
    float X[3] = { static_cast<float>(col), static_cast<float>(row), 1.0f};

	// Warp Z=W*X
    float Z[3] = { W[0 * 3 + 0] * X[0] + W[0 * 3 + 1] * X[1] + W[0 * 3 + 2] * X[2],
                   W[1 * 3 + 0] * X[0] + W[1 * 3 + 1] * X[1] + W[1 * 3 + 2] * X[2],
                   W[2 * 3 + 0] * X[0] + W[2 * 3 + 1] * X[1] + W[2 * 3 + 2] * X[2]
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
        
		// Calculate image difference D = I(W(x,p))-T(x).
		float D = I2 - templateImage[row * templateImageWidth + col];

		// Add a term to b matrix.
        
        atomicAdd(&(b[0]), steepestDescent[6 * (row * templateImageWidth + col) + 0] * D);
        atomicAdd(&(b[1]), steepestDescent[6 * (row * templateImageWidth + col) + 1] * D);
        atomicAdd(&(b[2]), steepestDescent[6 * (row * templateImageWidth + col) + 2] * D);
        atomicAdd(&(b[3]), steepestDescent[6 * (row * templateImageWidth + col) + 3] * D);
        atomicAdd(&(b[4]), steepestDescent[6 * (row * templateImageWidth + col) + 4] * D);
        atomicAdd(&(b[5]), steepestDescent[6 * (row * templateImageWidth + col) + 5] * D);
	}
}

void displayAlignment(float* image, const std::size_t imageHeight, const std::size_t imageWidth, float* templateImage, const std::size_t templateImageHeight, const std::size_t templateImageWidth, float* W)
{
    cv::Mat imageMat = buildImageMat(image, imageHeight, imageWidth);
    
    for (int row = 0; row < templateImageHeight; ++row)
	{
		for (int col = 0; col < templateImageWidth; ++col)
		{
            float X[3] = { static_cast<float>(col), static_cast<float>(row), 1.0f};
            
            float Z[3] = { W[0 * 3 + 0] * X[0] + W[0 * 3 + 1] * X[1] + W[0 * 3 + 2] * X[2],
                           W[1 * 3 + 0] * X[0] + W[1 * 3 + 1] * X[1] + W[1 * 3 + 2] * X[2],
                           W[2 * 3 + 0] * X[0] + W[2 * 3 + 1] * X[1] + W[2 * 3 + 2] * X[2]
                         };
            
            float col2 = Z[0];
            float row2 = Z[1];

			int row2i = int(floor(row2));
			int col2i = int(floor(col2));

			if (row2i >= 0 && row2i < imageHeight && // check if pixel is inside I.
				col2i >= 0 && col2i < imageWidth)
			{
                float I2 = interpolate(image, imageHeight, imageWidth, row2, col2);

				imageMat.at<float>(row2i, col2i) = templateImage[row * templateImageWidth + col];
                
				if (row == 0 || col == 0 || col == templateImageWidth - 1 || row == templateImageHeight - 1)
				{
					imageMat.at<float>(row2i, col2i) = 1.0f;
				}
			}
		}
	}
    
	cv::imshow("Result", imageMat);
	cv::waitKey(0);
}


extern "C"
void inverseCompositional( float* imageArray
                         , float* templateImageArray
                         , float* affineParameterEstimates
                         , const std::size_t imageHeight
                         , const std::size_t imageWidth
                         , const std::size_t templateImageHeight
                         , const std::size_t templateImageWidth
                         , const float epsilon
                         , const int maxIteration
                         )
{
    cv::Mat image = buildImageMat(imageArray, imageHeight, imageWidth);
    cv::Mat templateImageMat = buildImageMat(templateImageArray, templateImageHeight, templateImageWidth);
    
	// Find the 2-D similarity transform that best aligns the two images (uniform scale, rotation and translation)
	cv::Mat debug;

	cv::Mat template_gradient_row;   // Gradient of I in X direction.
	cv::Mat template_gradient_col;   // Gradient of I in Y direction.

									// Here we will store matrices.
	float*   W = new float[3 * 3];  // Current value of warp W(x,p)
	float*  dW = new float[3 * 3];  // Warp update.
	float* idW = new float[3 * 3];  // Warp update.

	cv::Mat_<float> H(6,6);         // Approximate Hessian. - has to be inverted online
	float b[6];                     // Vector in the right side of the system of linear equations.
	float delta_p[6];               // Parameter update value.

							  // Create images.
	template_gradient_row = cv::Mat(templateImageMat.rows, templateImageMat.cols, CV_32FC1);
	template_gradient_col = cv::Mat(templateImageMat.rows, templateImageMat.cols, CV_32FC1);
    
    float* steepestDescent = new float[6 * templateImageHeight * templateImageWidth];

	//The "magic number" appearing at the end in the following is simply the inverse 
	//of the absolute sum of the weights in the matrix representing the Scharr filter.
	cv::Scharr(templateImageMat, template_gradient_row, -1, 0, 1, 1.0 / 32.0);
	cv::Scharr(templateImageMat, template_gradient_col, -1, 1, 0, 1.0 / 32.0);
    
	H = cv::Mat::zeros(6, 6, CV_32FC1);    
    	
	// Walk through pixels in the template T.
	for(int col = 0; col < templateImageWidth; ++col)
	{
		for(int row = 0; row < templateImageHeight; ++row)
		{
			// Evaluate gradient of T.
			float Tx = template_gradient_col.at<float>(row, col);	
			float Ty = template_gradient_row.at<float>(row, col);	
			
			// Calculate steepest descent image's element.
            steepestDescent[6 * (row * templateImageWidth + col) + 0] = Tx * col;
            steepestDescent[6 * (row * templateImageWidth + col) + 1] = Ty * col;
            steepestDescent[6 * (row * templateImageWidth + col) + 2] = Tx * row;
            steepestDescent[6 * (row * templateImageWidth + col) + 3] = Ty * row;
            steepestDescent[6 * (row * templateImageWidth + col) + 4] = Tx;
            steepestDescent[6 * (row * templateImageWidth + col) + 5] = Ty;
            
			// Add a term to Hessian.
			for(int i = 0; i < 6; ++i)
			{
				for(int j = 0; j < 6; ++j)
				{
                    H.at<float>(i, j) += steepestDescent[6 * (row * templateImageWidth + col) + i] * steepestDescent[6 * (row * templateImageWidth + col) + j];
				}
			}
		}
	}
    
    // Invert Hessian.
    cv::Mat iH = H.inv();

	/*
	 *   Iteration stage.
	 */
     
    buildTransformationFromInput(W, affineParameterEstimates);    
    
    // copy images, estimate to gpu memory
    std::size_t imageArraySize = imageWidth * imageHeight * sizeof(float);
    std::size_t templateImageArraySize = templateImageWidth * templateImageHeight * sizeof(float);
    std::size_t steepestDescentArraySize = 6 * templateImageWidth * templateImageHeight * sizeof(float);
    std::size_t wArraySize = 3 * 3 * sizeof(float);
    std::size_t bArraySize = 6 * 1 * sizeof(float);
  
    float* gpuImage = NULL;
    cudaMalloc((void **)&gpuImage, imageArraySize);
    float* gpuTemplateImage = NULL;
    cudaMalloc((void **)&gpuTemplateImage, templateImageArraySize);
    float* gpuSteepestDescentImage = NULL;
    cudaMalloc((void **)&gpuSteepestDescentImage, steepestDescentArraySize);
    float* gpuW = NULL;
    cudaMalloc((void **)&gpuW, wArraySize);
    float* gpuB = NULL;
    cudaMalloc((void **)&gpuB, bArraySize);
    
    cudaMemcpy(gpuImage, imageArray, imageArraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuTemplateImage, templateImageArray, templateImageArraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuSteepestDescentImage, steepestDescent, steepestDescentArraySize, cudaMemcpyHostToDevice);

	// Iterate
	int iter=0; // number of current iteration
    while(iter < maxIteration)
	{
		++iter; // Increment iteration counter		
        
        cudaMemcpy(gpuW, W, wArraySize, cudaMemcpyHostToDevice);
        
        for(int i = 0; i < 6; ++i)
        {
            b[i] = 0.0f;
        }
        
        cudaMemcpy(gpuB, b, bArraySize, cudaMemcpyHostToDevice);
			
        getB<<<templateImageHeight, templateImageWidth>>>(gpuImage, imageHeight, imageWidth, gpuTemplateImage, gpuSteepestDescentImage, gpuW, gpuB);

        cudaMemcpy(b, gpuB, bArraySize, cudaMemcpyDeviceToHost);
        
		// Find parameter increment.
        //delta_p = iH * b;
        for(int row = 0; row < 6; ++row)
        {
            delta_p[row] = 0.0f;
            for(int col = 0; col < 6; ++col)
            {
                delta_p[row] += iH.at<float>(row,col) * b[col];
            }
        }

		init_warp(dW, delta_p);
        
		// Invert warp.
		getWarpInvert(dW, idW);
        
        //W o idW;
		update_warp(W, idW);

		// Check termination critera.
		if(norm(delta_p, 6, 1) <= epsilon) break;
	}
    
    std::cout << "Finished in " << iter << " iterations." << std::endl << std::endl;
    
    std::cout << "[ " << W[0] << ",\t" << W[1] << ",\t" << W[2] << ";" << std::endl << "  " << W[3] << ",\t" << W[4] << ",\t" << W[5] << ";" << std::endl << " " << W[6] << ",\t" << W[7] << ",\t" << W[8] << " ]" << std::endl;
    
    displayAlignment(imageArray, imageHeight, imageWidth, templateImageArray, templateImageHeight, templateImageWidth, W);
    
    delete[] W;
    delete[] dW;
    delete[] idW;
    
    
    cudaFree(gpuB);
    cudaFree(gpuW);
    cudaFree(gpuImage);
    cudaFree(gpuSteepestDescentImage);
    cudaFree(gpuTemplateImage);
}
