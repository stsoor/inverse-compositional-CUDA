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

void buildTransformationFromInput(cv::Mat& W, float* p)
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

void init_warp(cv::Mat& W, cv::Mat p)
{
	W.at<float>(0, 0) = 1.0 + p.at<float>(0,0);
	W.at<float>(0, 1) = p.at<float>(2,0);
	W.at<float>(0, 2) = p.at<float>(4,0);
    
	W.at<float>(1, 0) = p.at<float>(1,0);
	W.at<float>(1, 1) = 1.0 + p.at<float>(3,0);
	W.at<float>(1, 2) = p.at<float>(5,0);
    
	W.at<float>(2, 0) = 0.0;
	W.at<float>(2, 1) = 0.0;
	W.at<float>(2, 2) = 1.0;
}

cv::Mat getWarpInvert(cv::Mat W)
{
    float p1 = W.at<float>(0,0) - 1.0;
    float p2 = W.at<float>(1,0);
    float p3 = W.at<float>(0,1);
    float p4 = W.at<float>(1,1) - 1.0;
    float p5 = W.at<float>(0,2);
    float p6 = W.at<float>(1,2);
    
    float det = (1 + p1) * (1 + p4) - p2 * p3;
    
    cv::Mat invW = (cv::Mat_<float>(3,3) <<
                                            1.0 + (-p1 - p1 * p4 + p2 * p3) / det, (-p3) / det, (-p5 - p4 * p5 + p3 * p6) / det,
                                            (-p2) / det, 1.0 + (-p4 - p1 * p4 + p2 * p3) / det, (-p6 - p1 * p6 + p2 * p5) / det,
                                            0.0, 0.0, 1.0
                   );
    
    return invW;
}

void update_warp(cv::Mat& W, cv::Mat idW)
{
    cv::Mat p =  (cv::Mat_<float>(6,1) <<   W.at<float>(0,0) - 1.0,   W.at<float>(1,0),   W.at<float>(0,1),   W.at<float>(1,1) - 1.0,   W.at<float>(0,2),   W.at<float>(1,2) );
    cv::Mat dp = (cv::Mat_<float>(6,1) << idW.at<float>(0,0) - 1.0, idW.at<float>(1,0), idW.at<float>(0,1), idW.at<float>(1,1) - 1.0, idW.at<float>(0,2), idW.at<float>(1,2) );
        
    cv::Mat_<float>newP(6,1);
    
    newP.at<float>(0,0) = p.at<float>(0,0) + dp.at<float>(0,0) + p.at<float>(0,0) * dp.at<float>(0,0) + p.at<float>(2,0) * dp.at<float>(1,0);
    newP.at<float>(1,0) = p.at<float>(1,0) + dp.at<float>(1,0) + p.at<float>(1,0) * dp.at<float>(0,0) + p.at<float>(3,0) * dp.at<float>(1,0);
    newP.at<float>(2,0) = p.at<float>(2,0) + dp.at<float>(2,0) + p.at<float>(0,0) * dp.at<float>(2,0) + p.at<float>(2,0) * dp.at<float>(3,0);
    newP.at<float>(3,0) = p.at<float>(3,0) + dp.at<float>(3,0) + p.at<float>(1,0) * dp.at<float>(2,0) + p.at<float>(3,0) * dp.at<float>(3,0);
    newP.at<float>(4,0) = p.at<float>(4,0) + dp.at<float>(4,0) + p.at<float>(0,0) * dp.at<float>(4,0) + p.at<float>(2,0) * dp.at<float>(5,0);
    newP.at<float>(5,0) = p.at<float>(5,0) + dp.at<float>(5,0) + p.at<float>(1,0) * dp.at<float>(4,0) + p.at<float>(3,0) * dp.at<float>(5,0);
    
    init_warp(W, newP);
}

float interpolate(cv::Mat& image, float y, float x)
{
  float xd, yd;  
  float k1 = modff(x,&xd);
  float k2 = modff(y,&yd);
  int xi = int(xd);
  int yi = int(yd);

  int f1 = xi < image.rows-1;  // Check that pixels to the right  
  int f2 = yi < image.cols-1; // and to down direction exist.

  float px1 = image.at<float>(yi  , xi);
  float px2 = image.at<float>(yi  , xi+1);
  float px3 = image.at<float>(yi+1, xi);
  float px4 = image.at<float>(yi+1, xi+1);      
  
  // Interpolate pixel intensity.
  float interpolated_value = 
        (1.0-k1)*(1.0-k2)*px1 +
  (     f1     ? ( k1*(1.0-k2)*px2 ) : 0) +
  (     f2     ? ( (1.0-k1)*k2*px3 ) : 0) +            
  ( (f1 && f2) ? ( k1*k2*px4 ) : 0);

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
    cv::Mat templateImageMat = buildImageMat(templateImageArray, templateImageHeight, templateImageWidth);
    
	// Find the 2-D similarity transform that best aligns the two images (uniform scale, rotation and translation)
	cv::Mat debug;

	cv::Mat template_gradient_row;    // Gradient of I in X direction.
	cv::Mat template_gradient_col;    // Gradient of I in Y direction.

									// Here we will store matrices.
	cv::Mat_<float> W(3,3);         // Current value of warp W(x,p)
	cv::Mat_<float> dW(3,3);        // Warp update.
	cv::Mat_<float> idW(3,3);       // Warp update.
	cv::Mat_<float> X(3,1);         // Point in coordinate frame of template.
	cv::Mat_<float> Z(3,1);         // Point in coordinate frame of image.

	cv::Mat_<float> H(6,6);         // Approximate Hessian.
	cv::Mat_<float> b(6,1);         // Vector in the right side of the system of linear equations.
	cv::Mat_<float> delta_p(6,1);   // Parameter update value.

							  // Create images.
	template_gradient_row = cv::Mat(templateImageMat.rows, templateImageMat.cols, CV_32FC1);
	template_gradient_col = cv::Mat(templateImageMat.rows, templateImageMat.cols, CV_32FC1);
    
    float* steepest_descent = new float[6 * templateImageHeight * templateImageWidth];

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
            steepest_descent[6 * (row * templateImageWidth + col) + 0] = Tx * col;
            steepest_descent[6 * (row * templateImageWidth + col) + 1] = Ty * col;
            steepest_descent[6 * (row * templateImageWidth + col) + 2] = Tx * row;
            steepest_descent[6 * (row * templateImageWidth + col) + 3] = Ty * row;
            steepest_descent[6 * (row * templateImageWidth + col) + 4] = Tx;
            steepest_descent[6 * (row * templateImageWidth + col) + 5] = Ty;
            
			// Add a term to Hessian.
			for(int i = 0; i < 6; ++i)
			{
				for(int j = 0; j < 6; ++j)
				{
                    H.at<float>(i, j) += steepest_descent[6 * (row * templateImageWidth + col) + i] * steepest_descent[6 * (row * templateImageWidth + col) + j];
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

	// Here we will store current value of mean error.
	float mean_error=0;

	// Iterate
	int iter=0; // number of current iteration
    while(iter < maxIteration)
	{
		++iter; // Increment iteration counter

        target.copyTo(debug);
        
		mean_error = 0; // Set mean error value with zero

		int pixel_count = 0; // Count of processed pixels
		
		b = cv::Mat::zeros(6, 1, CV_32FC1); // Set b matrix with zeroes
			
		// Walk through pixels in the template T.
		for(int col = 0; col < templateImageWidth; ++col)
		{
			for(int row = 0; row < templateImageHeight; ++row)
			{
				// Set vector X with pixel coordinates (x,y,1)
                X = (cv::Mat_<float>(3,1) << col, row, 1);

				// Warp Z=W*X
                Z = W * X;

				// Get coordinates of warped pixel in coordinate frame of I.
                float col2, row2; // pixel coordinates in the coordinate frame of I.
                col2 = Z.at<float>(0,0);
                row2 = Z.at<float>(1,0);

				// Get the nearest integer pixel coords (x2i;y2i).
				int col2i = int(floor(col2));
				int row2i = int(floor(row2));

				if(col2i >= 0 && col2i < imageWidth && // check if pixel is inside I.
					row2i >= 0 && row2i < imageHeight)
				{
					++pixel_count;

					// Calculate intensity of a transformed pixel with sub-pixel accuracy
					// using bilinear interpolation.
					float I2 = interpolate(target, row2, col2);
                    
                    debug.at<float>(row2i, col2i) = templateImageMat.at<float>(row,col);
                    if(row == 0 || col == 0 || col == templateImageMat.cols-1 || row == templateImageMat.rows-1)
                    {
                        debug.at<float>(row2i,col2i) = 1.0;
                    }

					// Calculate image difference D = I(W(x,p))-T(x).
					float D = I2 - templateImageMat.at<float>(row, col);

					// Update mean error value.
					mean_error += fabs(D);

					// Add a term to b matrix.
					b.at<float>(0,0) += steepest_descent[6 * (row * templateImageWidth + col) + 0] * D;
					b.at<float>(1,0) += steepest_descent[6 * (row * templateImageWidth + col) + 1] * D;
					b.at<float>(2,0) += steepest_descent[6 * (row * templateImageWidth + col) + 2] * D;	
					b.at<float>(3,0) += steepest_descent[6 * (row * templateImageWidth + col) + 3] * D;	
					b.at<float>(4,0) += steepest_descent[6 * (row * templateImageWidth + col) + 4] * D;	
					b.at<float>(5,0) += steepest_descent[6 * (row * templateImageWidth + col) + 5] * D;					
				}
			}
		}

		// Finally, calculate resulting mean error.
		if(pixel_count!=0)
			mean_error /= pixel_count;

		// Find parameter increment.
        delta_p = iH * b;

		init_warp(dW, delta_p);
        
		// Invert warp.
		idW = getWarpInvert(dW);
        
        //W o idW;
		update_warp(W, idW);
		// Print diagnostic information to screen.
		printf("iter=%d mean_error=%f\n", iter, mean_error);
        
        cv::imshow("Debug", debug);
        cv::waitKey(24);

		// Check termination critera.
		if(norm(delta_p) <= epsilon) break;
	}
    
    std::cout << W << std::endl;
    
    cv::imshow("Debug", debug);
    cv::waitKey(0);
}
