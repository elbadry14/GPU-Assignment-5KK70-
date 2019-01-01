#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#define ROLLSIZE 16

__global__ void clean(unsigned int * e, int n)
{
    e[threadIdx.x % n] = 0;
}

void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin)
{
//Uncomment following lines to unlock a CPU version histogram
    // int i;    
    // for ( i = 0; i < nbr_bin; i++) hist_out[i] = 0;
    // for ( i = 0; i < img_size; i++) hist_out[img_in[i]] ++;

//Uncomment following lines to unlock a GPU version histogram
    unsigned char * d_img;
    unsigned int * d_hist;
    cudaMalloc(&d_img,  img_size * sizeof(unsigned char));
    cudaMalloc(&d_hist, nbr_bin * sizeof(unsigned int));
    clean<<<1,nbr_bin>>>(d_hist, nbr_bin);
    cudaMemcpy(d_img, img_in, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);


    int numThreads = 256;    
    int serialNum = 1024;
    int numBlocks = (img_size / (numThreads*serialNum)) + 1;
    histogram_gpu_son<<<numBlocks, numThreads, ROLLSIZE*256*sizeof(unsigned int)>>>(d_img, d_hist, img_size, serialNum);

    cudaMemcpy(hist_out, d_hist, sizeof(int)*nbr_bin, cudaMemcpyDeviceToHost);
    cudaFree(d_hist);
    cudaFree(d_img);
    return;
}

__global__ void histogram_gpu_son(unsigned char * d_img, unsigned int * d_hist,  int img_size,  int serialNum)
{
    // __shared__ unsigned int aa[ROLLSIZE][256];
    extern __shared__ unsigned int aa[];
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int i;

    for(i = 0; i < ROLLSIZE; i++) aa[(i << 8) + threadIdx.x] = 0;
    __syncthreads();

    int end = (x+1)*serialNum;
    if (end >= img_size) end = img_size;
    
    for(i = x*serialNum; i < end; i++) atomicAdd(&(aa[((threadIdx.x >> 4 ) << 8) +  d_img[i]]), 1);
    __syncthreads();

    unsigned int s;
    for(s = 16 / 2; s > 0; s >>= 1) {
        //Only when numThreads == 256
        for(i = 0; i < s; i++) aa[(i << 8) + threadIdx.x] += aa[((i+s) << 8) + threadIdx.x];


        // if (threadIdx.x < s) {
            // for(i = 0; i < 256; i++) {
            //     aa[threadIdx.x][i] += aa[threadIdx.x + s][i];
            // }
        // }
        __syncthreads();
    }

    atomicAdd(&(d_hist[threadIdx.x]),aa[threadIdx.x]);
    return;
}

void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin)
{
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0) min = hist_in[i++];

    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if (lut[i] < 0) lut[i] = 0;
        if (lut[i] > 255) lut[i] = 255;
    }

    // for(i = 0; i < img_size; i++) {
    //     img_out[i] = (unsigned char) lut[img_in[i]];
    // }

    unsigned char * d_in;
    unsigned char * d_out;
    int * d_lut;

    cudaMalloc(&d_in,   img_size * sizeof(unsigned char));
    cudaMalloc(&d_out,  img_size * sizeof(unsigned char));
    cudaMalloc(&d_lut,  nbr_bin * sizeof(int));
    cudaMemcpy(d_in, img_in,    img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut,      nbr_bin * sizeof(int), cudaMemcpyHostToDevice);
    
    /* Get the result image */
    // int numThreads = 256;
    int numThreads = 1024;    
    // int numBlocks = (img_size / numThreads) + 1;
    int numBlocks = (img_size/numThreads) + 1;
    histogram_equalization_gpu_son<<<numBlocks, numThreads>>>(d_in, d_out, d_lut, img_size, (img_size / (numThreads * numBlocks)) + 1);
    cudaMemcpy(img_out, d_out, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_lut);
}

__global__ void histogram_equalization_gpu_son (unsigned char * d_in, unsigned char * d_out, int * d_lut, 
    int img_size,  int serialNum)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    if (x >= img_size) return;
    
    d_out[x] = (unsigned char) d_lut[d_in[x]];    
}



