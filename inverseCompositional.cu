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

void buildTransformation(cv::Mat& W, float* p)
{
    W.at<float>(0,0) = p[0];
    W.at<float>(0,1) = p[2];
    W.at<float>(0,2) = p[4];
    W.at<float>(1,0) = p[1];
    W.at<float>(1,1) = p[3];
    W.at<float>(1,2) = p[5];
    W.at<float>(2,0) = 0.0f;
    W.at<float>(2,1) = 0.0f;
    W.at<float>(2,2) = 1.0f;
}


// Our warp matrix looks like this one:
//
//  ! 1  -wz  tx !
//  ! wz  1   ty !
//  ! 0   0   1  !
//
void init_warp(cv::Mat& W, float wz, float tx, float ty)
{
	W.at<float>(0, 0) = 1;
	W.at<float>(1, 0) = wz;
	W.at<float>(2, 0) = 0;
    
	W.at<float>(0, 1) = -wz;
	W.at<float>(1, 1) = 1;
	W.at<float>(2, 1) = 0;
    
	W.at<float>(0, 2) = tx;
	W.at<float>(1, 2) = ty;
	W.at<float>(2, 2) = 1;
}

template <class T>
T interpolate(cv::Mat& image, float y, float x)
{
  float xd, yd;  
  float k1 = modff(x,&xd);
  float k2 = modff(y,&yd);
  int xi = int(xd);
  int yi = int(yd);

  int f1 = xi < image.rows-1;  // Check that pixels to the right  
  int f2 = yi < image.cols-1; // and to down direction exist.

  T px1 = image.at<T>(yi  , xi);
  T px2 = image.at<T>(yi  , xi+1);
  T px3 = image.at<T>(yi+1, xi);
  T px4 = image.at<T>(yi+1, xi+1);      
  
  // Interpolate pixel intensity.
  T interpolated_value = 
  (1.0-k1)*(1.0-k2)*px1 +
  (f1 ? ( k1*(1.0-k2)*px2 ):0) +
  (f2 ? ( (1.0-k1)*k2*px3 ):0) +            
  ((f1 && f2) ? ( k1*k2*px4 ):0);

  return interpolated_value;
}

float norm(cv::Mat m)
{
    float squareSum = 0.0f;
    
    for(std::size_t row = 0; row < m.rows; ++row)
    {
        for(std::size_t col = 0; col < m.cols; ++col)
        {
            float elem = m.at<float>(static_cast<int>(row), static_cast<int>(col));
            
            squareSum += elem * elem;
        }
    }
    
    return sqrt(squareSum);
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
    cv::Mat target = buildImageMat(imageArray, imageHeight, imageWidth);
    cv::Mat source = buildImageMat(templateImageArray, templateImageHeight, templateImageWidth);
    
	// Find the 2-D similarity transform that best aligns the two images (uniform scale, rotation and translation)
	cv::Mat debug;

	cv::Mat source_gradient_row;    // Gradient of I in X direction.
	cv::Mat source_gradient_col;    // Gradient of I in Y direction.

									// Here we will store matrices.
	cv::Mat_<float> W(3,3);         // Current value of warp W(x,p)
	cv::Mat_<float> dW(3,3);        // Warp update.
	cv::Mat_<float> idW(3,3);       // Warp update.
	cv::Mat_<float> X(3,1);         // Point in coordinate frame of source.
	cv::Mat_<float> Z(3,1);         // Point in coordinate frame of target.

	cv::Mat_<float> H(3,3);         // Approximate Hessian.
	cv::Mat_<float> b(3,1);         // Vector in the right side of the system of linear equations.
	cv::Mat_<float> delta_p(3,1);   // Parameter update value.

							  // Create images.
	source_gradient_row = cv::Mat(source.rows, source.cols, CV_32FC1);
	source_gradient_col = cv::Mat(source.rows, source.cols, CV_32FC1);
    
    float* steepest_descent = new float[3 * templateImageHeight * templateImageWidth];

	//The "magic number" appearing at the end in the following is simply the inverse 
	//of the absolute sum of the weights in the matrix representing the Scharr filter.
	cv::Scharr(source, source_gradient_row, -1, 0, 1, 1.0 / 32.0);
	cv::Scharr(source, source_gradient_col, -1, 1, 0, 1.0 / 32.0);
    
	H = cv::Mat::zeros(3, 3, CV_32FC1);    
    
	int u, v;	// (u,v) - pixel coordinates in the coordinate frame of T.
	float u2, v2; // (u2,v2) - pixel coordinates in the coordinate frame of I.
	
	// Walk through pixels in the template T.
	int i, j;
	for(i=0; i< templateImageWidth; i++)
	{
		u = i;

		for(j=0; j < templateImageHeight; j++)
		{
			v = j;

			// Evaluate gradient of T.
			float Tx = source_gradient_col.at<float>(v, u);	
			float Ty = source_gradient_row.at<float>(v, u);	
			
			// Calculate steepest descent image's element.
            steepest_descent[3 * (v * templateImageWidth + u) + 0] = -v*Tx+u*Ty;
            steepest_descent[3 * (v * templateImageWidth + u) + 1] = Tx;
            steepest_descent[3 * (v * templateImageWidth + u) + 2] = Ty;
            
			// Add a term to Hessian.
			int l,m;
			for(l=0;l<3;l++)
			{
				for(m=0;m<3;m++)
				{
                    H.at<float>(l, m) += steepest_descent[3 * (v * templateImageWidth + u) + l] * steepest_descent[3 * (v * templateImageWidth + u) + m];
				}
			}
		}
	}
    
    // Invert Hessian.
    cv::Mat iH = H.inv();

	/*
	 *   Iteration stage.
	 */
    
    buildTransformation(W, affineParameterEstimates);
    
    cv::Mat R = (cv::Mat_<float>(3,3) << sqrt(2.0)/2.0, -sqrt(2.0)/2.0, 0, sqrt(2.0)/2.0, sqrt(2.0)/2.0, 0, 0, 0, 1);
    W.at<float>(0,2) += 100;
    W.at<float>(1,2) -= 100;
    
    W = W * R;
    

	// Here we will store current value of mean error.
	float mean_error=0;

	// Iterate
	int iter=0; // number of current iteration
    while(iter < maxIteration)
	{
		iter++; // Increment iteration counter

        target.copyTo(debug);
        
		mean_error = 0; // Set mean error value with zero

		int pixel_count=0; // Count of processed pixels
		
		b = cv::Mat::zeros(3, 1, CV_32FC1); // Set b matrix with zeroes
			
		// Walk through pixels in the template T.
		int i, j;
		for(i=0; i<templateImageWidth; i++)
		{
			int u = i;

			for(j=0; j< templateImageHeight; j++)
			{
				int v = j;

				// Set vector X with pixel coordinates (u,v,1)
                X = (cv::Mat_<float>(3,1) << u, v, 1);

				// Warp Z=W*X
                Z = W * X;

				// Get coordinates of warped pixel in coordinate frame of I.
                u2 = Z.at<float>(0,0);
                v2 = Z.at<float>(1,0);

				// Get the nearest integer pixel coords (u2i;v2i).
				int u2i = int(floor(u2));
				int v2i = int(floor(v2));

				if(u2i >= 0 && u2i < imageWidth && // check if pixel is inside I.
					v2i >= 0 && v2i < imageHeight)
				{
					pixel_count++;

					// Calculate intensity of a transformed pixel with sub-pixel accuracy
					// using bilinear interpolation.
					float I2 = interpolate<float>(target, v2, u2);
                    
                    debug.at<float>(v2i,u2i) = source.at<float>(v,u);
                    //if(v == 0 && u == 0 || v == 0 && u == source.cols-1 || v == source.rows-1 && u == 0 || v == source.rows-1 && u == source.cols-1)
                    if(v == 0 || u == 0 || u == source.cols-1 || v == source.rows-1)
                    {
                        debug.at<float>(v2i,u2i) = 1.0;
                    }

					// Calculate image difference D = I(W(x,p))-T(x).
					float D = I2 - source.at<float>(v, u);

					// Update mean error value.
					mean_error += fabs(D);

					// Add a term to b matrix.
					b.at<float>(0,0) += steepest_descent[3 * (v * templateImageWidth + u) + 0] * D;
					b.at<float>(1,0) += steepest_descent[3 * (v * templateImageWidth + u) + 1] * D;
					b.at<float>(2,0) += steepest_descent[3 * (v * templateImageWidth + u) + 2] * D;					
				}	
			}
		}

		// Finally, calculate resulting mean error.
		if(pixel_count!=0)
			mean_error /= pixel_count;

		// Find parameter increment.
        delta_p = iH * b;
        
		float delta_wz = delta_p.at<float>(0, 0);
		float delta_tx = delta_p.at<float>(1, 0);
		float delta_ty = delta_p.at<float>(2, 0);

		init_warp(dW, delta_wz, delta_tx, delta_ty);
		// Invert warp.
		idW = dW.inv();

        dW = idW * W;
        dW.copyTo(W);

		// Print diagnostic information to screen.
		printf("iter=%d mean_error=%f\n", iter, mean_error);
        
        cv::imshow("Debug", debug);
        cv::waitKey(24);

		// Check termination critera.
		if(fabs(delta_wz)<=epsilon && fabs(delta_tx)<=epsilon && fabs(delta_ty)<=epsilon) break;
	}
    
    std::cout << W << std::endl;
    
    cv::imshow("Debug", debug);
    cv::waitKey(0);
}
