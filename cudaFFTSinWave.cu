#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <chrono>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>


typedef float2 Complex;

#define NX 1024
#define BATCH 1 


int main()
{
    cudaError_t err = cudaSuccess;
    float mulApl = 1.0;
    float mulFre = 2*M_PI;
    float divFreq = 20.0;

    cufftHandle plan;
    cufftComplex *data = nullptr; //cufftComplex is single-precision, floating-point Complex

    //allocate GPU memory
    if(cudaMalloc((void **)&data, sizeof(cufftComplex)*NX*BATCH) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for data, error = %s \n ", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //create signal on host
    Complex* h_signal = new Complex [NX*BATCH];

    for(int i = 0; i < NX; i++)
    {
        h_signal[i].x = mulApl*sin(mulFre*i / divFreq);
        h_signal[i].y = 0;
    }

    std::ofstream input_file;
    std::ofstream output_file;

    input_file.open("input_file.dat");
    output_file.open("output_file.dat");


    for(int i = 0; i < NX; i++)
    {
        input_file << h_signal[i].x << '\n';
    }

    
    

    //allocate memory on GPU
    if(cudaMemcpy(data, h_signal, sizeof(Complex)*NX*BATCH, cudaMemcpyHostToDevice)!=cudaSuccess)
    {
        fprintf(stderr, "CuFFT error: cudaMemcpy host to device failed! \n");
        exit(EXIT_FAILURE);
    }

    

    //create a plan for 1D transform
    if(cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS) 
    {
        fprintf(stderr, "CuFFT error: cufftPlan1d failed! \n");
        exit(EXIT_FAILURE);
    }

    auto start = std::chrono::system_clock::now();

    //complex-to-complex transforms for single/double precision
    if(cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CuFFT error: cufftExecC2C failed! \n");
        exit(EXIT_FAILURE);
    }

    auto finish = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start);

    std::cout << "time taken for FFT calculation of NX samples: " << duration.count() << " nanoseconds." << '\n';

    //Synchronize device i.e. barrier
    if(cudaDeviceSynchronize() != cudaSuccess)
    {
        fprintf(stderr, "Device failed to Synchronize! error = %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    Complex* h_convolved_signal = h_signal;
    for(int i =0; i< NX; i++)
    {
        h_convolved_signal[i].x = 0;
        h_convolved_signal[i].y = 0;
    }

    if(cudaMemcpy(h_convolved_signal, data, sizeof(Complex)*NX*BATCH, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "Device to host data copy failed, error = %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < NX; i++)
    {
        // std::cout << i << ". cos: " <<  h_convolved_signal[i].x << ", sin: " << h_convolved_signal[i].y << '\n';
        double val = pow(pow(h_convolved_signal[i].x, 2) + pow(h_convolved_signal[i].y, 2), 0.5);
        output_file << val << '\n';
    }

    //clear FFT allocated resources
    if(cufftDestroy(plan)!= CUFFT_SUCCESS)
    {
        fprintf(stderr, "Failed to free plan data resources");
        exit(EXIT_FAILURE);
    }

    //clear GPU memory
    if(cudaFree(data)!= cudaSuccess)
    {
        fprintf(stderr, "Failed to free data memory, error = %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //reset GPU
    if(cudaDeviceReset() != cudaSuccess)
    {
        fprintf(stderr, "Failed to reset device, error = %s", cudaGetErrorString(err));
    }

    //clear Host memory
    delete [] h_signal;

    output_file.close();
    input_file.close();

    
    return 0; 
}