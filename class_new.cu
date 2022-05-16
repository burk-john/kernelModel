#include <algorithm>
#include <iostream>
#include <stdio.h>

using namespace std;

//FC Layer using cuda
//Ni = size of input (I)
//Nn = size of output (O)
//B  = number of batches
//
//ONES  if true I and W (weights) are filled with 1s
//      else I and W are filled with random numbers (-0.5, 0.5)

#define Ni 4096     //25088
#define Nn 25088   //4096   //4096
#define B  256

#define Blocks 128

#define ONES true

float I_vals[B][Ni];
float W_vals[Ni][Nn];
float O_vals[B][Nn];

__global__
void fc(float I[][Ni], float W[][Nn], float O[][Nn]){
    float sum = 0;

    int n_start = blockIdx.x * (Nn / gridDim.x) + threadIdx.x;
    int n_end = n_start + Nn / gridDim.x / blockDim.x;

    for(int n = n_start; n < n_end; n++){
        for(int b = 0; b < B; b++){
            for(int i = 0; i < Ni; i++){
                sum += I[b][i] * W[i][n];
            }
            O[b][n] = sum;
            sum = 0;
        }
    }
}

int main() {
    float *I, *W, *O;

    cudaMallocManaged(&I, B*Ni*sizeof(float));
    cudaMallocManaged(&W, Ni*Nn*sizeof(float));
    cudaMallocManaged(&O, B*Nn*sizeof(float));

    //Initialize I
    for(int b = 0; b < B; b++) {
        for(int i = 0; i < Ni; i++) {
            if(ONES) {
                I_vals[b][i] = 1;
            } else {
                I_vals[b][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
            }
        }
    }

    //Initialize W
    for(int i = 0; i < Ni; i++){
        for(int n = 0; n < Nn; n++){
            if(ONES) {
                W_vals[i][n] = 1;
            } else {
                W_vals[i][n] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
            }
        }
    }
    //Initialize O with zeros
    for(int b = 0; b < B; b++) {
        for(int n = 0; n < Nn; n++){
            O_vals[b][n] = 0;
        }
    }

    std::copy(&I_vals[0][0], &I_vals[0][0] + B*Ni, I);
    std::copy(&W_vals[0][0], &W_vals[0][0] + Ni*Nn, W);
    std::copy(&O_vals[0][0], &O_vals[0][0] + B*Nn, O);

    int threads = 0;
    if(Nn / Blocks > 1024){
        threads = 1024;
    } else {
        threads = Nn / Blocks;
    }

    fc<<<Blocks, threads>>>( reinterpret_cast<float (*)[Ni]>(I),
                    reinterpret_cast<float (*)[Nn]>(W),
                    reinterpret_cast<float (*)[Nn]>(O) );

    cudaDeviceSynchronize();

    //Test when I and W initialized with 1s
    int err = 0;
    for(int b = 0; b < B; b++) {
        for(int n = 0; n < Nn; n++){
            if(ONES && (reinterpret_cast<float (*)[Nn]>(O)[b][n] != Ni)){
                err++;
            }
        }
    }

    if(ONES) {
        cout<<"Number of errors: "<<err<<endl;
    } else {
        cout<<"Error could not be calculated"<<endl;
    }

    cudaFree(I);
    cudaFree(W);
    cudaFree(O);

    return 0;
}
