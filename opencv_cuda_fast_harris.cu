#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include "timer.h"
#include <ctime>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

using namespace std;
using namespace cv;

int imageWidth;
int imageHeight;
int _threshold;


/*cudaError_t inverseImage(int * sourcePixelArray,int * destonationPixelArray);

__global__ void inverseImageKernel(int * sourcePixelArray,int * destonationPixelArray, int width, int height)
{
    int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

    if( xIdx < width && yIdx < height)
    {
        destonationPixelArray[ (yIdx * width + xIdx) * 3 + 0] = 255 - sourcePixelArray[ (yIdx * width + xIdx) * 3 + 0];
        destonationPixelArray[ (yIdx * width + xIdx) * 3 + 1] = 255 - sourcePixelArray[ (yIdx * width + xIdx) * 3 + 1];
        destonationPixelArray[ (yIdx * width + xIdx) * 3 + 2] = 255 - sourcePixelArray[ (yIdx * width + xIdx) * 3 + 2];
    }
}

__global__ void detectingLightnessKernel(int * sourcePixelArray,int * lightnessArray, int width, int height)
{
    int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

    if( xIdx < width && yIdx < height)
    {
        int lightness = 0.3 * sourcePixelArray[ (yIdx * width + xIdx) * 3 + 0] + 
                    0.59 * sourcePixelArray[ (yIdx * width + xIdx) * 3 + 1] + 
                    0.11 * sourcePixelArray[ (yIdx * width + xIdx) * 3 + 2];
        lightnessArray[ yIdx * width + xIdx] = lightness;
    }
}

__global__ void detectingPointsKernel(int *lightnessArray, int *detectedArray, int __threshold, int *detectedCounter, int width, int height)
{
    int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

    if( xIdx >= 1 && xIdx < width-1 && yIdx < height-1 && yIdx >= 1)
    {
        (*detectedCounter)++;
        int leftLineAverageLightness = (lightnessArray[ (yIdx-1) * width + xIdx-1] + lightnessArray[ (yIdx) * width + xIdx-1] + lightnessArray[ (yIdx+1) * width + xIdx-1]) / 3;                                        
        int upLineAverageLightness = (lightnessArray[ (yIdx-1) * width + xIdx-1] + lightnessArray[ (yIdx-1) * width + xIdx] + lightnessArray[ (yIdx-1) * width + xIdx+1]) / 3; 
                                    //(imageLightness[ (y-1)*imageWidth + x-1] + imageLightness[ (y-1)*imageWidth + x] + imageLightness[ (y-1)*imageWidth + x+1]) / 3;                                            
        int rightLineAverageLightness = (lightnessArray[ (yIdx-1) * width + xIdx+1] + lightnessArray[ (yIdx) * width + xIdx+1] + lightnessArray[ (yIdx+1) * width + xIdx+1]) / 3; 
                                    //(imageLightness[ (y-1)*imageWidth + x+1] + imageLightness[ y*imageWidth + x+1] + imageLightness[ (y+1)*imageWidth + x+1]) / 3;                                            
        int downLineAverageLightness = (lightnessArray[ (yIdx+1) * width + xIdx+1] + lightnessArray[ (yIdx+1) * width + xIdx] + lightnessArray[ (yIdx+1) * width + xIdx-1]) / 3;
                                    // (imageLightness[ (y+1)*imageWidth + x+1] + imageLightness[ (y+1)*imageWidth + x] + imageLightness[ (y+1)*imageWidth + x-1]) / 3;
        
        int leftUpCornerAverageLightness = ( leftLineAverageLightness + upLineAverageLightness ) / 2;
        int rightUpCornerAverageLightness = ( upLineAverageLightness + rightLineAverageLightness ) / 2;
        int leftDownCornerAverageLightness = ( leftLineAverageLightness + downLineAverageLightness ) / 2;
        int rightDownCornerAverageLightness = ( rightLineAverageLightness + downLineAverageLightness ) / 2;

        int leftDifference = abs( leftUpCornerAverageLightness - leftDownCornerAverageLightness);
        int rightDifference = abs( rightUpCornerAverageLightness - rightDownCornerAverageLightness);
        int upDifference = abs( leftUpCornerAverageLightness - rightUpCornerAverageLightness);
        int downDifference = abs( leftDownCornerAverageLightness - rightDownCornerAverageLightness);

        detectedArray[ yIdx * width + xIdx] = 0;
        if ( leftDifference >= __threshold || rightDifference >= __threshold)
            if( upDifference >= __threshold || downDifference >= __threshold){
                detectedArray[ yIdx * width + xIdx] = 1;
                //*detectedCounter++;                    
            }

        if( upDifference >= __threshold || downDifference >= __threshold )
            if ( leftDifference >= __threshold || rightDifference >= __threshold){
                detectedArray[ yIdx * width + xIdx] = 1;
               //*detectedCounter++;                    
            }
    }
}
void matToPixelArray( const Mat image, int * pixelArray);
void pixelArrayToMat(int * pixelArray, Mat * destImage);
void detectedArrayToKeyPointVector(int *detectedArray, std::vector<cv::KeyPoint>* keyPointVector);

cudaError_t gpuImageProcessing(int *sourceImagePixelArray, Mat * sourceImage, Mat * destonationImage)
{
    int *devSourceImagePixelArray;
    int *imageLightness;
    int *detected;
    std::vector<cv::KeyPoint> keyPointVector = std::vector<cv::KeyPoint>();    
    int * hostDetected = new int[imageWidth * imageHeight]; 
    int lightnessCounter = 0;
    
    cout<<"Cuda start malloc. Lightness."<<endl;
    //Cuda malloc
    cudaError_t cuerr = cudaMalloc( (void**)&imageLightness, imageWidth * imageHeight * sizeof(int) );
    if (cuerr != cudaSuccess)
    {
        printf("Cuda malloc imageLightness error: %s\n", cudaGetErrorString(cuerr));
        goto Error;        
    }
    cout<<"Cuda start malloc. Dev source image."<<endl;
    cuerr = cudaMalloc( (void**)&devSourceImagePixelArray, imageWidth * imageHeight * 3 * sizeof(int) );
    if (cuerr != cudaSuccess)
    {
        printf("Cuda malloc source image pixel array error: %s\n", cudaGetErrorString(cuerr));
        goto Error;        
    }
    cout<<"Cuda start malloc. Detected."<<endl;
    cuerr = cudaMalloc( (void**)&detected, imageWidth * imageHeight * sizeof(int) );
    if (cuerr != cudaSuccess)
    {
        printf("Cuda malloc detected error: %s\n", cudaGetErrorString(cuerr));
        goto Error;        
    }
    cout<<"Cuda start memcpy. Dev source image."<<endl;
    //Cuda memcpy source array
    cuerr = cudaMemcpy(devSourceImagePixelArray, sourceImagePixelArray, imageWidth * imageHeight * 3 * sizeof(int) , cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Source pixel array cudaMemcpy faied!\n");
        goto Error;
    }
    //Set device
    cout<<"Set cuda device"<<endl;
    cuerr = cudaSetDevice(0);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice faied!  Do you have a CUDA-capabe GPU instaed?\n");
        system("pause");
        goto Error;
    }

    //Init and start detecting Lightness kernel
    //Image lightness processing
    const int bs = 16;
    dim3 blockSize( bs, bs);
    dim3 gridSize( imageWidth / blockSize.x, imageHeight / blockSize.y);   

    detectingLightnessKernel<<< gridSize, blockSize>>>( devSourceImagePixelArray, imageLightness, imageWidth, imageHeight);

    cudaDeviceSynchronize();
    cout<<"Kernel finish"<<endl;
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        printf("FUNC ERROR: %s\n", cudaGetErrorString(cuerr));
        goto Error;
    }

    //Point detection
    int detectedCounter = 0;
    //int devDetectedCounter = 0;
    detectingPointsKernel<<< gridSize, blockSize>>>( imageLightness, detected, _threshold, &detectedCounter, imageWidth, imageHeight);
    cudaDeviceSynchronize();
    cout<<"Kernel finish"<<endl;
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        printf("FUNC ERROR: %s\n", cudaGetErrorString(cuerr));
        goto Error;
    }
    printf("Detection counter: %d\n", detectedCounter);

    cout<<"Pre cuda memcpy dest image"<<endl;   
    cuerr = cudaMemcpy( hostDetected, detected, imageWidth * imageHeight * sizeof(int), cudaMemcpyDeviceToHost);
    if ( cuerr  != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy faied!\n");
        goto Error;
    }   
    
    detectedArrayToKeyPointVector(hostDetected, &keyPointVector);        
    drawKeypoints( *sourceImage, keyPointVector, *destonationImage, Scalar( 0, 255, 255 ), DrawMatchesFlags::DEFAULT );
    
Error:
    cudaFree(devSourceImagePixelArray);
    cudaFree(imageLightness);
    cudaFree(detected);
    free( hostDetected);
    return cuerr;    
}


void inverseImageCPU(int *sourceImagePixelArray, int * destonationImagePixelArray)
{
    for(int y=0; y<imageHeight; y++)
        for(int x=0; x<imageWidth; x++)
        {
            destonationImagePixelArray[ (y * imageWidth + x) * 3 + 0] = 255 - sourceImagePixelArray[ (y * imageWidth + x) * 3 + 0];
            destonationImagePixelArray[ (y * imageWidth + x) * 3 + 1] = 255 - sourceImagePixelArray[ (y * imageWidth + x) * 3 + 1];
            destonationImagePixelArray[ (y * imageWidth + x) * 3 + 2] = 255 - sourceImagePixelArray[ (y * imageWidth + x) * 3 + 2];
        }
}

int getLightness(int *rgb)
{
    return (int) ( 0.3*rgb[0] + 0.59*rgb[1] + 0.11*rgb[2]);
}

void cpuImageProcessing(int *sourceImagePixelArray, Mat * sourceImage, Mat * destonationImage)
{
    int *imageLightness = new int [ imageWidth * imageHeight];
    int lightnessCounter = 0;
    
    //Image lightness processing
    for( int y=0; y<imageHeight; y++)
        for(int x=0; x<imageWidth; x++)
        {
            int *rgb = new int[3];
            rgb[0] = sourceImagePixelArray[ (y * imageWidth + x) * 3 + 0];
            rgb[1] = sourceImagePixelArray[ (y * imageWidth + x) * 3 + 1];
            rgb[2] = sourceImagePixelArray[ (y * imageWidth + x) * 3 + 2];
            imageLightness[ lightnessCounter++] = getLightness(rgb);
        }        

    //Check detected points
    int *detected = new int[ imageWidth * imageHeight];
    
    int maxLighntess = 0;
    int minLightness = 255;
    int detectedCounter = 0;
    
    for( int y=1; y < imageHeight-1; y++)
        for(int x=1; x < imageWidth-1; x++)
        {
            if( maxLighntess < imageLightness[ y * imageWidth + x])
                maxLighntess = imageLightness[ y * imageWidth + x];
            if( minLightness > imageLightness[ y * imageWidth + x])
                minLightness = imageLightness[ y * imageWidth + x];

            int leftLineAverageLightness = (imageLightness[ (y-1)*imageWidth + x-1] + imageLightness[ y*imageWidth + x-1] + imageLightness[ (y+1)*imageWidth + x-1]) / 3;
            int upLineAverageLightness = (imageLightness[ (y-1)*imageWidth + x-1] + imageLightness[ (y-1)*imageWidth + x] + imageLightness[ (y-1)*imageWidth + x+1]) / 3;                                            
            int rightLineAverageLightness = (imageLightness[ (y-1)*imageWidth + x+1] + imageLightness[ y*imageWidth + x+1] + imageLightness[ (y+1)*imageWidth + x+1]) / 3;                                            
            int downLineAverageLightness = (imageLightness[ (y+1)*imageWidth + x+1] + imageLightness[ (y+1)*imageWidth + x] + imageLightness[ (y+1)*imageWidth + x-1]) / 3;
                                            

            int leftUpCornerAverageLightness = ( leftLineAverageLightness + upLineAverageLightness ) / 2;
            int rightUpCornerAverageLightness = ( upLineAverageLightness + rightLineAverageLightness ) / 2;
            int leftDownCornerAverageLightness = ( leftLineAverageLightness + downLineAverageLightness ) / 2;
            int rightDownCornerAverageLightness = ( rightLineAverageLightness + downLineAverageLightness ) / 2;

            int leftDifference = abs( leftUpCornerAverageLightness - leftDownCornerAverageLightness);
            int rightDifference = abs( rightUpCornerAverageLightness - rightDownCornerAverageLightness);
            int upDifference = abs( leftUpCornerAverageLightness - rightUpCornerAverageLightness);
            int downDifference = abs( leftDownCornerAverageLightness - rightDownCornerAverageLightness);

            detected[y * imageWidth + x] = 0;
            if ( leftDifference >= _threshold || rightDifference >= _threshold)
                if( upDifference >= _threshold || downDifference >= _threshold){
                    detected[y * imageWidth + x] = 1;
                    detectedCounter++;                    
                }

            if( detected[y * imageWidth + x] != 1 &&  (upDifference >= _threshold || downDifference >= _threshold) )
                if ( leftDifference >= _threshold || rightDifference >= _threshold){
                    detected[y * imageWidth + x] = 1;
                    detectedCounter++;                    
                }
        }
        printf("Max lightness:= %d, Min lightness:= %d\n", maxLighntess, minLightness);
        printf("Detected counter:= %d", detectedCounter);

        std::vector<cv::KeyPoint> keyPointVector = std::vector<cv::KeyPoint>();
        detectedArrayToKeyPointVector(detected, &keyPointVector);        
        drawKeypoints( *sourceImage, keyPointVector, *destonationImage, Scalar( 0, 255, 255 ), DrawMatchesFlags::DEFAULT );
}
*/

//Declarations
void getLightnessArray( unsigned char * imagePixelArray, int* lightnessArray);
void detectedArrayToKeyPointVector(int *detectedArray, std::vector<cv::KeyPoint>* keyPointVector);

void cpuImageProcessing(const Mat sourceImage, Mat * destonationImage)
{
    unsigned char *imagePixelArray = (unsigned char *)sourceImage.data;

    //get image lightness array    
    int * lightnessArray = new int[ imageWidth * imageHeight];
    getLightnessArray( imagePixelArray, lightnessArray);

    //Check detected points
    int *detected = new int[ imageWidth * imageHeight];
    
    int maxLighntess = 0;
    int minLightness = 255;
    int detectedCounter = 0;
    
    for( int y=1; y < imageHeight - 1 ; y++)
        for(int x=1; x < imageWidth - 1 ; x++)
        {
            if( maxLighntess < lightnessArray[ y * imageWidth + x])
                maxLighntess = lightnessArray[ y * imageWidth + x];
            if( minLightness > lightnessArray[ y * imageWidth + x])
                minLightness = lightnessArray[ y * imageWidth + x];

            int leftLineAverageLightness = (lightnessArray[ (y-1)*imageWidth + x-1] + lightnessArray[ y*imageWidth + x-1] + lightnessArray[ (y+1)*imageWidth + x-1]) / 3;
            int upLineAverageLightness = (lightnessArray[ (y-1)*imageWidth + x-1] + lightnessArray[ (y-1)*imageWidth + x] + lightnessArray[ (y-1)*imageWidth + x+1]) / 3;                                            
            int rightLineAverageLightness = (lightnessArray[ (y-1)*imageWidth + x+1] + lightnessArray[ y*imageWidth + x+1] + lightnessArray[ (y+1)*imageWidth + x+1]) / 3;                                            
            int downLineAverageLightness = (lightnessArray[ (y+1)*imageWidth + x+1] + lightnessArray[ (y+1)*imageWidth + x] + lightnessArray[ (y+1)*imageWidth + x-1]) / 3;
                                            

            int leftUpCornerAverageLightness = ( leftLineAverageLightness + upLineAverageLightness ) / 2;
            int rightUpCornerAverageLightness = ( upLineAverageLightness + rightLineAverageLightness ) / 2;
            int leftDownCornerAverageLightness = ( leftLineAverageLightness + downLineAverageLightness ) / 2;
            int rightDownCornerAverageLightness = ( rightLineAverageLightness + downLineAverageLightness ) / 2;

            int leftDifference = abs( leftUpCornerAverageLightness - leftDownCornerAverageLightness);
            int rightDifference = abs( rightUpCornerAverageLightness - rightDownCornerAverageLightness);
            int upDifference = abs( leftUpCornerAverageLightness - rightUpCornerAverageLightness);
            int downDifference = abs( leftDownCornerAverageLightness - rightDownCornerAverageLightness);

            detected[y * imageWidth + x] = 0;
            if ( leftDifference >= _threshold || rightDifference >= _threshold)
                if( upDifference >= _threshold || downDifference >= _threshold){
                    detected[y * imageWidth + x] = 1;
                    detectedCounter++;                    
                }

            if( detected[y * imageWidth + x] != 1 &&  (upDifference >= _threshold || downDifference >= _threshold) )
                if ( leftDifference >= _threshold || rightDifference >= _threshold){
                    detected[y * imageWidth + x] = 1;
                    detectedCounter++;                    
                }
        }
        printf("Max lightness:= %d, Min lightness:= %d\n", maxLighntess, minLightness);
        printf("Detected counter:= %d\n", detectedCounter);

        std::vector<cv::KeyPoint> keyPointVector = std::vector<cv::KeyPoint>();
        detectedArrayToKeyPointVector(detected, &keyPointVector);        
        drawKeypoints( sourceImage, keyPointVector, *destonationImage, Scalar( 0, 255, 255 ), DrawMatchesFlags::DEFAULT );

}
void getLightnessArray( unsigned char * imagePixelArray, int* lightnessArray)
{
    int r, g, b;
    r = imagePixelArray[ 2];
    g = imagePixelArray[ 1];
    b = imagePixelArray[ 0];
    printf(" R:= %d G:= %d B:= %d\n", r, g, b);
    
    for( int y = 0 ; y < imageHeight ; y++)
        for( int x = 0 ; x < imageWidth ; x++)
        {
            r = imagePixelArray[ (y * imageWidth + x)*3 + 2];
            g = imagePixelArray[ (y * imageWidth + x)*3 + 1];
            b = imagePixelArray[ (y * imageWidth + x)*3 + 0];
            //printf(" R:= %d G:= %d B:= %d\n", r, g, b);
            lightnessArray[ y * imageWidth + x] =(int) ( 0.3 * r + 0.59 * g + 0.11 * b);
        }    

    
}
void detectedArrayToKeyPointVector(int *detectedArray, std::vector<cv::KeyPoint>* keyPointVector)
{
    for( int y=1; y < imageHeight-1; y++)
        for(int x=1; x < imageWidth-1; x++)
        {
            if( detectedArray[ y * imageWidth + x] == 1)
            {
                cv::KeyPoint kp( x, y, 1);
                keyPointVector->push_back(kp);
            }
        }
}

//CUDA functions
//kernels
__global__ void detectingLightnessWithoutOptimizationKernel(unsigned char* devSourceImagePixelArray,int * lightnessArray,int width,int height)
{
    int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

    if( xIdx < width && yIdx < height)
    {
        int lightness = 0.3  * devSourceImagePixelArray[ (yIdx * width + xIdx) * 3 + 2] + 
                        0.59 * devSourceImagePixelArray[ (yIdx * width + xIdx) * 3 + 1] + 
                        0.11 * devSourceImagePixelArray[ (yIdx * width + xIdx) * 3 + 0];
        lightnessArray[ yIdx * width + xIdx] = lightness;
    }
}
__global__ void detectingPointsWithoutOptimizationKernel(int *imageLightness, int *devDetectedPointArray, int __threshold,int width,int height)
{
    int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

    if( xIdx >= 1 && xIdx < width-1 && yIdx < height-1 && yIdx >= 1)
    {
        int leftLineAverageLightness = (imageLightness[ (yIdx-1) * width + xIdx-1] + imageLightness[ (yIdx) * width + xIdx-1] + imageLightness[ (yIdx+1) * width + xIdx-1]) / 3;                                        
        int upLineAverageLightness = (imageLightness[ (yIdx-1) * width + xIdx-1] + imageLightness[ (yIdx-1) * width + xIdx] + imageLightness[ (yIdx-1) * width + xIdx+1]) / 3;
        int rightLineAverageLightness = (imageLightness[ (yIdx-1) * width + xIdx+1] + imageLightness[ (yIdx) * width + xIdx+1] + imageLightness[ (yIdx+1) * width + xIdx+1]) / 3;
        int downLineAverageLightness = (imageLightness[ (yIdx+1) * width + xIdx+1] + imageLightness[ (yIdx+1) * width + xIdx] + imageLightness[ (yIdx+1) * width + xIdx-1]) / 3;
                  
        int leftUpCornerAverageLightness = ( leftLineAverageLightness + upLineAverageLightness ) / 2;
        int rightUpCornerAverageLightness = ( upLineAverageLightness + rightLineAverageLightness ) / 2;
        int leftDownCornerAverageLightness = ( leftLineAverageLightness + downLineAverageLightness ) / 2;
        int rightDownCornerAverageLightness = ( rightLineAverageLightness + downLineAverageLightness ) / 2;

        int leftDifference = abs( leftUpCornerAverageLightness - leftDownCornerAverageLightness);
        int rightDifference = abs( rightUpCornerAverageLightness - rightDownCornerAverageLightness);
        int upDifference = abs( leftUpCornerAverageLightness - rightUpCornerAverageLightness);
        int downDifference = abs( leftDownCornerAverageLightness - rightDownCornerAverageLightness);

        devDetectedPointArray[ yIdx * width + xIdx] = 0;
        if ( leftDifference >= __threshold || rightDifference >= __threshold)
            if( upDifference >= __threshold || downDifference >= __threshold){
                devDetectedPointArray[ yIdx * width + xIdx] = 1;
                //*detectedCounter++;                    
            }

        if( upDifference >= __threshold || downDifference >= __threshold )
            if ( leftDifference >= __threshold || rightDifference >= __threshold){
                devDetectedPointArray[ yIdx * width + xIdx] = 1;
               //*detectedCounter++;                    
            }
    }
}

__global__ void detectingLightnessWithOptimizationKernel(unsigned char* devSourceImagePixelArray,int * lightnessArray,int width,int height)
{
    int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

    int tid = threadIdx.x;
    __shared__ int data[3];
    data[0] = devSourceImagePixelArray[ (yIdx * width + xIdx) * 3 + 2];
    data[1] = devSourceImagePixelArray[ (yIdx * width + xIdx) * 3 + 1];
    data[2] = devSourceImagePixelArray[ (yIdx * width + xIdx) * 3 + 0];
    __syncthreads ();
    if( xIdx < width && yIdx < height)
    {
        int lightness = 0.3  * data[0]+ 
                        0.59 * data[1] + 
                        0.11 * data[2];
        __syncthreads ();
        lightnessArray[ yIdx * width + xIdx] = lightness;
    }
}
__global__ void detectingPointsWithOptimizationKernel(int *imageLightness, int *devDetectedPointArray, int __threshold,int width,int height)
{
    int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ int data[8];
    data[0] = imageLightness[ (yIdx-1) * width + xIdx-1];
    data[1] = imageLightness[ (yIdx-1) * width + xIdx];
    data[2] = imageLightness[ (yIdx-1) * width + xIdx+1];
    data[3] = imageLightness[ (yIdx) * width + xIdx+1];
    data[4] = imageLightness[ (yIdx+1) * width + xIdx+1];
    data[5] = imageLightness[ (yIdx+1) * width + xIdx];
    data[6] = imageLightness[ (yIdx+1) * width + xIdx-1];
    data[7] = imageLightness[ (yIdx) * width + xIdx-1];
    __syncthreads ();

    if( xIdx >= 1 && xIdx < width-1 && yIdx < height-1 && yIdx >= 1)
    {
        int leftLineAverageLightness = (data[0] + data[7] + data[6]) / 3;                                        
        int upLineAverageLightness = (data[0] + data[1] + data[2]) / 3;
        int rightLineAverageLightness = ( data[2] + data[3] + data[4]) / 3;
        int downLineAverageLightness = (data[4] + data[5] + data[6]) / 3;
        __syncthreads ();          
        int leftUpCornerAverageLightness = ( leftLineAverageLightness + upLineAverageLightness ) / 2;
        int rightUpCornerAverageLightness = ( upLineAverageLightness + rightLineAverageLightness ) / 2;
        int leftDownCornerAverageLightness = ( leftLineAverageLightness + downLineAverageLightness ) / 2;
        int rightDownCornerAverageLightness = ( rightLineAverageLightness + downLineAverageLightness ) / 2;

        int leftDifference = abs( leftUpCornerAverageLightness - leftDownCornerAverageLightness);
        int rightDifference = abs( rightUpCornerAverageLightness - rightDownCornerAverageLightness);
        int upDifference = abs( leftUpCornerAverageLightness - rightUpCornerAverageLightness);
        int downDifference = abs( leftDownCornerAverageLightness - rightDownCornerAverageLightness);

        devDetectedPointArray[ yIdx * width + xIdx] = 0;
        if ( leftDifference >= __threshold || rightDifference >= __threshold)
            if( upDifference >= __threshold || downDifference >= __threshold){
                devDetectedPointArray[ yIdx * width + xIdx] = 1;
                //*detectedCounter++;                    
            }

        if( upDifference >= __threshold || downDifference >= __threshold )
            if ( leftDifference >= __threshold || rightDifference >= __threshold){
                devDetectedPointArray[ yIdx * width + xIdx] = 1;
               //*detectedCounter++;                    
            }
    }
}

cudaError_t gpuImageProcessingWithOptimization(const Mat sourceImage, Mat * destonationImage)
{
    std::vector<cv::KeyPoint> keyPointVector = std::vector<cv::KeyPoint>();
    int *lightnessArray;
    int *devDetectedPointArray;
    int *detectedPointArray = new int[ imageWidth * imageHeight];
    unsigned char *devSourceImagePixelArray;

    clock_t mallocStartTime= clock();
    /*cudaError_t cuerr = cudaSetDevice(0);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice faied!  Do you have a CUDA-capabe GPU instaed?\n");
        system("pause");
        goto Error;
    }    
    */
    //Cuda malloc
    cout<<"Cuda start malloc. Source image pixel array."<<endl;
    cudaError_t cuerr = cudaMalloc( (void**)&devSourceImagePixelArray, imageWidth * imageHeight * 3 * sizeof(int) ); 
    if (cuerr != cudaSuccess)
    {
        printf("Cuda malloc source image pixel array error: %s\n", cudaGetErrorString(cuerr));
        goto Error;        
    }

    cout<<"Cuda start malloc. Lightness."<<endl;
    cuerr = cudaMalloc( (void**)&lightnessArray, imageWidth * imageHeight * sizeof(int) );
    if (cuerr != cudaSuccess)
    {
        printf("Cuda malloc imageLightness error: %s\n", cudaGetErrorString(cuerr));
        goto Error;        
    }

    cout<<"Cuda start malloc. Device detected point array."<<endl;
    cuerr = cudaMalloc( (void**)&devDetectedPointArray, imageWidth * imageHeight * sizeof(int) );
    if (cuerr != cudaSuccess)
    {
        printf("Cuda malloc detected point error: %s\n", cudaGetErrorString(cuerr));
        goto Error;        
    } 
    //Cuda memcpy source array
    cout<<"Cuda start memcpy. Dev source image."<<endl;    
    cuerr = cudaMemcpy( devSourceImagePixelArray, (unsigned char *)sourceImage.data, imageWidth * imageHeight * 3 * sizeof(unsigned char ) , cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Source pixel array cudaMemcpy faied!\n");
        goto Error;
    }
    float mallocFinishTime = clock() - mallocStartTime;
    printf("Malloc time:= %f \n", mallocFinishTime);
    
    //Init and start detecting Lightness kernel
    const int bs = 16;
    dim3 blockSize( bs, bs);
    dim3 gridSize( imageWidth / blockSize.x, imageHeight / blockSize.y);   

    cout<<"Lightness kernel start."<<endl;
    detectingLightnessWithOptimizationKernel<<< gridSize, blockSize>>>( devSourceImagePixelArray, lightnessArray, imageWidth, imageHeight);

    cudaDeviceSynchronize();
    cout<<"Lightness kernel finish."<<endl;
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        printf("FUNC ERROR: %s\n", cudaGetErrorString(cuerr));
        goto Error;
    }

    // ====================================
    //           Point detecting
    // ====================================
    cout<<"Detecting point kernel start."<<endl;
    detectingPointsWithOptimizationKernel<<< gridSize, blockSize>>>( lightnessArray, devDetectedPointArray, _threshold, imageWidth, imageHeight);
    cudaDeviceSynchronize();
    cout<<"Detecting point kernel finish."<<endl;    
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        printf("FUNC ERROR: %s\n", cudaGetErrorString(cuerr));
        goto Error;
    }

    cout<<"Pre cuda memcpy dest image"<<endl;   
    cuerr = cudaMemcpy( detectedPointArray, devDetectedPointArray, imageWidth * imageHeight * sizeof(int), cudaMemcpyDeviceToHost);
    if ( cuerr  != cudaSuccess) {
        fprintf(stderr, "Detected points cudaMemcpy faied!\n");
        goto Error;
    }   
    cout<<"Pre detected array to keypoint vector"<<endl;
    detectedArrayToKeyPointVector( detectedPointArray, &keyPointVector);     
    cout<<"Dest. image init"<<endl;
    drawKeypoints( sourceImage, keyPointVector, *destonationImage, Scalar( 0, 255, 255 ), DrawMatchesFlags::DEFAULT );
    cout<<"Dest. image init finish"<<endl;
    
Error:
    cudaFree(lightnessArray);
    cudaFree(devSourceImagePixelArray);    
    cudaFree(devDetectedPointArray);
    free( detectedPointArray );

    return cuerr;    
}

cudaError_t gpuImageProcessingWithoutOptimization(const Mat sourceImage, Mat * destonationImage)
{
    std::vector<cv::KeyPoint> keyPointVector = std::vector<cv::KeyPoint>();
    int *lightnessArray;
    int *devDetectedPointArray;
    int *detectedPointArray = new int[ imageWidth * imageHeight];
    unsigned char *devSourceImagePixelArray;

    clock_t mallocStartTime= clock();
    cudaError_t cuerr = cudaSetDevice(0);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice faied!  Do you have a CUDA-capabe GPU instaed?\n");
        system("pause");
        goto Error;
    }    
    
    //Cuda malloc
    cout<<"Cuda start malloc. Source image pixel array."<<endl;
    cuerr = cudaMalloc( (void**)&devSourceImagePixelArray, imageWidth * imageHeight * 3 * sizeof(int) ); 
    if (cuerr != cudaSuccess)
    {
        printf("Cuda malloc source image pixel array error: %s\n", cudaGetErrorString(cuerr));
        goto Error;        
    }

    cout<<"Cuda start malloc. Lightness."<<endl;
    cuerr = cudaMalloc( (void**)&lightnessArray, imageWidth * imageHeight * sizeof(int) );
    if (cuerr != cudaSuccess)
    {
        printf("Cuda malloc imageLightness error: %s\n", cudaGetErrorString(cuerr));
        goto Error;        
    }

    cout<<"Cuda start malloc. Device detected point array."<<endl;
    cuerr = cudaMalloc( (void**)&devDetectedPointArray, imageWidth * imageHeight * sizeof(int) );
    if (cuerr != cudaSuccess)
    {
        printf("Cuda malloc detected point error: %s\n", cudaGetErrorString(cuerr));
        goto Error;        
    } 
    //Cuda memcpy source array
    cout<<"Cuda start memcpy. Dev source image."<<endl;    
    cuerr = cudaMemcpy( devSourceImagePixelArray, (unsigned char *)sourceImage.data, imageWidth * imageHeight * 3 * sizeof(unsigned char ) , cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Source pixel array cudaMemcpy faied!\n");
        goto Error;
    }
    float mallocFinishTime = clock() - mallocStartTime;
    printf("Malloc time:= %f \n", mallocFinishTime);
    
    //Init and start detecting Lightness kernel
    const int bs = 16;
    dim3 blockSize( bs, bs);
    dim3 gridSize( imageWidth / blockSize.x, imageHeight / blockSize.y);   

    cout<<"Lightness kernel start."<<endl;
    detectingLightnessWithoutOptimizationKernel<<< gridSize, blockSize>>>( devSourceImagePixelArray, lightnessArray, imageWidth, imageHeight);

    cudaDeviceSynchronize();
    cout<<"Lightness kernel finish."<<endl;
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        printf("FUNC ERROR: %s\n", cudaGetErrorString(cuerr));
        goto Error;
    }

    // ====================================
    //           Point detecting
    // ====================================
    cout<<"Detecting point kernel start."<<endl;
    detectingPointsWithoutOptimizationKernel<<< gridSize, blockSize>>>( lightnessArray, devDetectedPointArray, _threshold, imageWidth, imageHeight);
    cudaDeviceSynchronize();
    cout<<"Detecting point kernel finish."<<endl;    
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        printf("FUNC ERROR: %s\n", cudaGetErrorString(cuerr));
        goto Error;
    }

    cout<<"Pre cuda memcpy dest image"<<endl;   
    cuerr = cudaMemcpy( detectedPointArray, devDetectedPointArray, imageWidth * imageHeight * sizeof(int), cudaMemcpyDeviceToHost);
    if ( cuerr  != cudaSuccess) {
        fprintf(stderr, "Detected points cudaMemcpy faied!\n");
        goto Error;
    }   
    cout<<"Pre detected array to keypoint vector"<<endl;
    detectedArrayToKeyPointVector( detectedPointArray, &keyPointVector);     
    cout<<"Dest. image init"<<endl;
    drawKeypoints( sourceImage, keyPointVector, *destonationImage, Scalar( 0, 255, 255 ), DrawMatchesFlags::DEFAULT );
    cout<<"Dest. image init finish"<<endl;
    
Error:
    cudaFree(lightnessArray);
    cudaFree(devSourceImagePixelArray);    
    cudaFree(devDetectedPointArray);
    free( detectedPointArray );

    return cuerr;    
}

int main(int argc, char** argv)
{
    //Check params
    if( argc !=3 ){
        cout<<"Enter [image] [treshold]"<<endl;
        system("pause");
        return 0;
    }
    //Initialize image
    Mat image = imread(argv[1]);
    if( image.empty()){
        cout << "Cannot load image!" << endl;
        system("pause");
        return -1;
    }
    _threshold = atoi( argv[2]);
    
    imageHeight = image.rows;
    imageWidth  = image.cols;

    cout<<"Image width: "<<imageWidth<<" Image height: " << imageHeight<<endl;
    Mat destonationImage;

    //Opencv processing
    const clock_t opencvStartTime = clock();    
    SurfFeatureDetector detector( _threshold * 10 );
    std::vector<KeyPoint> keypoints_object_1;
    detector.detect( image, keypoints_object_1);
    drawKeypoints( image, keypoints_object_1, destonationImage, Scalar( 0, 255, 255 ), DrawMatchesFlags::DEFAULT );
    float opencvTime = clock() - opencvStartTime;
    printf("OpenCV processing time: %f\n", opencvTime);
    if( destonationImage.empty()){
        printf("Destonation image is empty");
        system("pause");
    }
    else{
        imwrite("opencv_image.jpg", destonationImage);
        namedWindow("OpenCV", CV_WINDOW_AUTOSIZE);        
        Size s( 380, 240);
        resize( destonationImage, destonationImage, s, 0, 0, CV_INTER_AREA );
        imshow("OpenCV", destonationImage);        
        waitKey(0);            
    }


    const clock_t cpuStartTime = clock();
    cpuImageProcessing( image, &destonationImage);
    float cpuTime = clock() - cpuStartTime;
    printf("CPU processing time: %f\n", cpuTime);
    
    if( destonationImage.empty()){
        printf("Destonation image is empty");
        system("pause");
    }
    else{
        imwrite("cpu_image.jpg", destonationImage);
        namedWindow("CPU", CV_WINDOW_AUTOSIZE);        
        Size s( 380, 240);
        resize( destonationImage, destonationImage, s, 0, 0, CV_INTER_AREA );
        imshow("CPU", destonationImage);        
        waitKey(0);            
    }

    //Start cuda part without optimization;
    const clock_t gpuTime1 = clock();
    // Add vectors in parallel.
    cudaError_t cudaStatus = gpuImageProcessingWithoutOptimization(image, &destonationImage);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU processing failed");
        return 1;
    }
    float gpuProcessingTime1 = clock() - gpuTime1;
    printf("GPU processing time without optimization: %f\n", gpuProcessingTime1);
   
    if( destonationImage.empty()){
        printf("Destonation image is empty");
        system("pause");
    }
    else{
        imwrite("gpu_image_without_optimization.jpg", destonationImage);
        namedWindow("GPU without optimization", CV_WINDOW_AUTOSIZE);        
        Size s( 380, 240);
        resize( destonationImage, destonationImage, s, 0, 0, CV_INTER_AREA );
        imshow("GPU without optimization", destonationImage);        
        waitKey(0);            
    }

    //Start cuda part with shared memory optimization
    const clock_t gpuTime2 = clock();
    // Add vectors in parallel.
    cudaStatus = gpuImageProcessingWithOptimization(image, &destonationImage);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU processing failed");
        return 1;
    }
    float gpuProcessingTime2 = clock() - gpuTime2;
    printf("GPU processing time with optimization: %f\n", gpuProcessingTime2);
   
    if( destonationImage.empty()){
        printf("Destonation image is empty");
        system("pause");
    }
    else{
        imwrite("gpu_image_with_optimization.jpg", destonationImage);
        namedWindow("GPU with optimization", CV_WINDOW_AUTOSIZE);        
        Size s( 380, 240);
        resize( destonationImage, destonationImage, s, 0, 0, CV_INTER_AREA );
        imshow("GPU with optimization", destonationImage);        
        waitKey(0);            
    }   

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    
    return 0;
}
