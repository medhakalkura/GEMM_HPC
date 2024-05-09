#include <stdio.h>
#include <time.h>
#include <math.h>
#define threshold 0.0001
#define BLOCK_SIZE 16
#define FIXME 1

void checkCUDAError(const char *msg);

// const int DSIZE = 960;
cudaEvent_t start, stop;
float tstart, elapsedTime;

// matrix multiply kernel: C = A * B
__global__ void mmtt(const float *A, const float *B, float *C, int Ni, int Nj, int Nk) {

	// Fill in code for GPU kernel
  // int i = blockDim.x * blockIdx.x + threadIdx.x; // create thread x index
  // int j = blockDim.y * blockIdx.y + threadIdx.y; // create thread y index
  // if(i<Ni && j<Nj)
  // {
  //   float sum = 0;
  //   int idx = j*Nk;
  //   for(int k = 0; k < Nk; k++)
  //   {
  //     sum += A[k*Ni + i] * B[ idx + k];
  //   }
  //   C[i*Nj + j] = sum;
  // }

  int i = blockDim.x*blockIdx.x+threadIdx.x; 
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  int tx = threadIdx.x; 
  int ty = threadIdx.y; 

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE];
  if ((i < Ni) && (j < Nj)){
    float sum = 0;
// Assumes ds is a multiple of BLOCK_SIZE (960/16 = 60)
    for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
    {
      As[ty][tx] = A[(ks+ty)*Ni+i]; 
      Bs[ty][tx] = B[j*Nk+(ks+tx)];
      __syncthreads(); 
      for (int k = ks; k < ks+BLOCK_SIZE; k++)
        sum += As[k-ks][tx] * Bs[ty][k-ks];   
      __syncthreads();
    }
    C[i*Nj+j] = sum;
  }
}

int main(){

  float *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;
  int i,j,k;
  int Ni,Nj,Nk;

  printf("Specify Matrix dimension Ni, Nj, Nk: ");
  scanf("%d %d %d", &Ni,&Nj,&Nk);
  h_A = (float *) malloc(sizeof(float)*Ni*Nk);
  h_B = (float *) malloc(sizeof(float)*Nk*Nj);
  h_C = (float *) malloc(sizeof(float)*Ni*Nj);
  h_Cref = (float *) malloc(sizeof(float)*Ni*Nj);

  for (i=0; i<Ni; i++)
   for (k=0; k<Nk; k++)
    h_A[k*Ni+i] = rand();

  for (k=0; k<Nk; k++)
   for (j=0; j<Nj; j++)
    h_B[k*Nj+j] = rand();

  for (i=0; i<Ni; i++)
   for (j=0; j<Nj; j++)
   {
    h_C[i*Nj+j] = 0;
    h_Cref[i*Nj+j] = 0;

   }

  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
//   h_Cref[i][j] += h_A[k][i]*h_B[j][k];
     h_Cref[i*Nj+j]=h_Cref[i*Nj+j]+h_A[k*Ni+i]*h_B[j*Nk+k];
  
 // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, Ni*Nk*sizeof(float));
  cudaMalloc(&d_B, Nk*Nj*sizeof(float));
  cudaMalloc(&d_C, Ni*Nj*sizeof(float));
  checkCUDAError("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, Ni*Nk*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, Nk*Nj*sizeof(float), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy H2D transfer failure");

  dim3 block(BLOCK_SIZE,BLOCK_SIZE);  
  dim3 grid(ceil(Ni/float(block.x)),ceil(Nj/float(block.y)));
  printf("Matrix size : A : %d and %d\n", Ni,Nk);
  printf("Matrix size : B : %d and %d\n", Nk,Nj);
  printf("Matrix size : C : %d and %d\n", Ni,Nj);

  for(int trial=0;trial<3;trial++)
  {
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);
   // Launch kernel
   mmtt<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
   checkCUDAError("GPU kernel launch failure");
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsedTime, start,stop);
   cudaDeviceSynchronize();
   // Copy results back to host
   cudaMemcpy(h_C, d_C, Ni*Nj*sizeof(float), cudaMemcpyDeviceToHost);
   checkCUDAError("cudaMemcpy D2H");
   for (int i = 0; i < Ni*Nj; i++) if (fabs((h_C[i]-h_Cref[i])/h_Cref[i])>threshold) {printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", i, h_C[i], h_Cref[i]); return -1;}
   printf("<BX=%d,BY=%d>: Trial %d: GFLOPS: %.2f\n",block.x,block.y,trial,2.0e-6*Ni*Nj*Nk/elapsedTime);
  }
  return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

