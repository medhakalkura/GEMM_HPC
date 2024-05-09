#define BLOCK_SIZE 16
__global__ void ab_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{

  if(Ni%BLOCK_SIZE == 0 && Nj%BLOCK_SIZE==0 && Nk%BLOCK_SIZE==0)
  {
    if ((Ni == Nj && Nj == Nk) || (Ni==Nj && Nj>Nk))
    {
      int j = (blockDim.x * blockIdx.x + threadIdx.x) * 4; 
      int i = blockDim.y*blockIdx.y*4+threadIdx.y;
      int tx = threadIdx.x; 
      int ty = threadIdx.y; 

    __shared__ float As1[BLOCK_SIZE][BLOCK_SIZE], As2[BLOCK_SIZE][BLOCK_SIZE], As3[BLOCK_SIZE][BLOCK_SIZE], As4[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs1[BLOCK_SIZE][BLOCK_SIZE], Bs2[BLOCK_SIZE][BLOCK_SIZE], Bs3[BLOCK_SIZE][BLOCK_SIZE], Bs4[BLOCK_SIZE][BLOCK_SIZE];
    if ((i < Ni) && (j < Nj)){
      float sum1 = 0, sum2=0, sum3=0, sum4=0;
      float sum5 = 0, sum6=0, sum7=0, sum8=0;
      float sum9 = 0, sum10=0, sum11=0, sum12=0;
      float sum13 = 0, sum14=0, sum15=0, sum16=0;
      for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
      {
        As1[ty][tx] = A[i*Nk+(ks+tx)];
        As2[ty][tx] = A[(i+16)*Nk+(ks+tx)]; 
        As3[ty][tx] = A[(i+32)*Nk+(ks+tx)]; 
        As4[ty][tx] = A[(i+48)*Nk+(ks+tx)]; 

        Bs1[ty][tx] = B[(ks+ty)*Nj+j];
        Bs2[ty][tx] = B[(ks+ty)*Nj+j+1];
        Bs3[ty][tx] = B[(ks+ty)*Nj+j+2];
        Bs4[ty][tx] = B[(ks+ty)*Nj+j+3];

        __syncthreads(); 
        for (int k = ks; k < ks+BLOCK_SIZE; k++)
        {
          sum1 += As1[ty][k-ks] * Bs1[k-ks][tx]; 
          sum2 += As1[ty][k-ks] * Bs2[k-ks][tx]; 
          sum3 += As1[ty][k-ks] * Bs3[k-ks][tx]; 
          sum4 += As1[ty][k-ks] * Bs4[k-ks][tx]; 

          sum5 += As2[ty][k-ks] * Bs1[k-ks][tx]; 
          sum6 += As2[ty][k-ks] * Bs2[k-ks][tx]; 
          sum7 += As2[ty][k-ks] * Bs3[k-ks][tx]; 
          sum8 += As2[ty][k-ks] * Bs4[k-ks][tx]; 

          sum9 += As3[ty][k-ks] * Bs1[k-ks][tx]; 
          sum10 += As3[ty][k-ks] * Bs2[k-ks][tx]; 
          sum11 += As3[ty][k-ks] * Bs3[k-ks][tx]; 
          sum12 += As3[ty][k-ks] * Bs4[k-ks][tx]; 

          sum13 += As4[ty][k-ks] * Bs1[k-ks][tx]; 
          sum14 += As4[ty][k-ks] * Bs2[k-ks][tx]; 
          sum15 += As4[ty][k-ks] * Bs3[k-ks][tx]; 
          sum16 += As4[ty][k-ks] * Bs4[k-ks][tx]; 
        }  
        __syncthreads();
      }
      C[i*Nj+j] = sum1;
      C[i*Nj+j+1] = sum2;
      C[i*Nj+j+2] = sum3;
      C[i*Nj+j+3] = sum4;

      C[(i+16)*Nj+j] = sum5;
      C[(i+16)*Nj+j+1] = sum6;
      C[(i+16)*Nj+j+2] = sum7;
      C[(i+16)*Nj+j+3] = sum8;

      C[(i+32)*Nj+j] = sum9;
      C[(i+32)*Nj+j+1] = sum10;
      C[(i+32)*Nj+j+2] = sum11;
      C[(i+32)*Nj+j+3] = sum12;

      C[(i+48)*Nj+j] = sum13;
      C[(i+48)*Nj+j+1] = sum14;
      C[(i+48)*Nj+j+2] = sum15;
      C[(i+48)*Nj+j+3] = sum16;
      }
    }
    else
    {
      int j = blockDim.x*blockIdx.x+threadIdx.x; 
      int i = blockDim.y*blockIdx.y+threadIdx.y;
      int tx = threadIdx.x; 
      int ty = threadIdx.y; 

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    if ((i < Ni) && (j < Nj)){
      float sum = 0;
      for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
      {
        As[ty][tx] = A[i*Nk+(ks+tx)]; 
        Bs[ty][tx] = B[(ks+ty)*Nj+j];
        __syncthreads(); 
        for (int k = ks; k < ks+BLOCK_SIZE; k++)
        {
          sum += As[ty][k-ks] * Bs[k-ks][tx];
        }  
        __syncthreads();
      }
      C[i*Nj+j] = sum;
      }
  }
}
else
{
  // int i = blockDim.x * blockIdx.x + threadIdx.x; 
  // int j = (blockDim.y * blockIdx.y + threadIdx.y)*4;
  // if(i<Ni && j<Nj)
  // {
  //   float sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
  //   for(int k = 0; k < Nk; k++)
  //   {
  //     sum1 += A[k*Ni + i] * B[ min(j,Nj-1)*Nk + k];
  //     sum2 += A[k*Ni + i] * B[ min(j+1,Nj-1)*Nk + k];
  //     sum3 += A[k*Ni + i] * B[ min(j+2,Nj-1)*Nk + k];
  //     sum4 += A[k*Ni + i] * B[ min(j+3,Nj-1)*Nk + k];
  //   }
  //   C[i*Nj + min(j,Nj)] = sum1;
  //   C[i*Nj + min(j+1,Nj)] = sum2;
  //   C[i*Nj + min(j+2,Nj)] = sum3;
  //   C[i*Nj + min(j+3,Nj)] = sum4;
  // }
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if(i<Ni && j<Nj)
  {
    float sum = 0;
    for(int k = 0; k < Nk; k++)
    {
      sum += A[i*Nk + k] * B[ k*Nj+j];
    }
    C[i*Nj + j] = sum;
  }

    //   int j = blockDim.x*blockIdx.x+threadIdx.x; 
    //   int i = blockDim.y*blockIdx.y+threadIdx.y;
    //   int tx = threadIdx.x; 
    //   int ty = threadIdx.y; 
    //   int Ni_new = Ni - (Ni%(BLOCK_SIZE*4));
    //   int Nj_new = Nj - (Nj%(BLOCK_SIZE*4));
    //   int Nk_new = Nk - (Nk%(BLOCK_SIZE*4));

    // __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    // __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // if ((i < Ni_new) && (j < Nj_new)){
    //   float sum = 0;
    //   for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
    //   {
    //     As[ty][tx] = A[i*Nk+(ks+tx)]; 
    //     Bs[ty][tx] = B[(ks+ty)*Nj+j];
    //     __syncthreads(); 
    //     for (int k = ks; k < ks+BLOCK_SIZE; k++)
    //     {
    //       sum += As[ty][k-ks] * Bs[k-ks][tx];
    //     }  
    //     __syncthreads();
    //   }
    //   C[i*Nj+j] = sum;
    //   }
}
}

__global__ void ab_gpu_horizontal(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk, int Ni_new, int Nj_new)
{
      int j = blockDim.x*blockIdx.x+threadIdx.x; 
      int i = blockDim.y*blockIdx.y+threadIdx.y;
      int tx = threadIdx.x; 
      int ty = threadIdx.y; 

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    if ((i < Ni_new) && (j < Nj_new)){
      float sum = 0;
      for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
      {
        As[ty][tx] = A[i*Nk+(ks+tx)]; 
        Bs[ty][tx] = B[(ks+ty)*Nj+j];
        __syncthreads(); 
        for (int k = ks; k < ks+BLOCK_SIZE; k++)
        {
          sum += As[ty][k-ks] * Bs[k-ks][tx];
        }  
        __syncthreads();
      }
      C[i*Nj+j] = sum;
      }
}
__global__ void ab_gpu_vertical(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk,int Ni_new, int Nj_new)
{

}
__global__ void ab_gpu_small_square(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if(i<Ni && j<Nj)
  {
    float sum = 0;
    for(int k = 0; k < Nk; k++)
    {
      sum += A[i*Nk + k] * B[ k*Nj+j];
    }
    C[i*Nj + j] = sum;
  }
}

__global__ void abT_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  if(Ni%BLOCK_SIZE == 0 && Nj%BLOCK_SIZE==0 && Nk%BLOCK_SIZE==0)
  {
    if((Ni == Nj && Nj == Nk) || (Ni==Nj && Nj>Nk))
    {
      int j = (blockDim.x * blockIdx.x + threadIdx.x) * 4; 
      int i = blockDim.y*blockIdx.y*4+threadIdx.y;
      int tx = threadIdx.x; 
      int ty = threadIdx.y; 
      
      __shared__ float As1[BLOCK_SIZE][BLOCK_SIZE], As2[BLOCK_SIZE][BLOCK_SIZE], As3[BLOCK_SIZE][BLOCK_SIZE], As4[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs1[BLOCK_SIZE][BLOCK_SIZE], Bs2[BLOCK_SIZE][BLOCK_SIZE], Bs3[BLOCK_SIZE][BLOCK_SIZE], Bs4[BLOCK_SIZE][BLOCK_SIZE];
      if ((i < Ni) && (j < Nj)){

      float sum1 = 0, sum2 = 0 ,sum3 = 0, sum4 = 0;
      float sum5 = 0, sum6 = 0 ,sum7 = 0, sum8 = 0;
      float sum9 = 0, sum10 = 0 ,sum11 = 0, sum12 = 0;
      float sum13 = 0, sum14 = 0 ,sum15 = 0, sum16 = 0;

      for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
      {
        As1[ty][tx] = A[i*Nk+(ks+tx)];
        As2[ty][tx] = A[(i+16)*Nk+(ks+tx)]; 
        As3[ty][tx] = A[(i+32)*Nk+(ks+tx)]; 
        As4[ty][tx] = A[(i+48)*Nk+(ks+tx)]; 

        Bs1[tx][ty] = B[j*Nk+(ks+ty)];
        Bs2[tx][ty] = B[(j+1)*Nk+(ks+ty)];
        Bs3[tx][ty] = B[(j+2)*Nk+(ks+ty)];
        Bs4[tx][ty] = B[(j+3)*Nk+(ks+ty)];
        __syncthreads(); 
        for (int k = ks; k < ks+BLOCK_SIZE; k++)
        {
          sum1 += As1[ty][k-ks] * Bs1[tx][k-ks]; 
          sum2 += As1[ty][k-ks] * Bs2[tx][k-ks]; 
          sum3 += As1[ty][k-ks] * Bs3[tx][k-ks]; 
          sum4 += As1[ty][k-ks] * Bs4[tx][k-ks]; 

          sum5 += As2[ty][k-ks] * Bs1[tx][k-ks]; 
          sum6 += As2[ty][k-ks] * Bs2[tx][k-ks]; 
          sum7 += As2[ty][k-ks] * Bs3[tx][k-ks]; 
          sum8 += As2[ty][k-ks] * Bs4[tx][k-ks]; 

          sum9 += As3[ty][k-ks] * Bs1[tx][k-ks]; 
          sum10 += As3[ty][k-ks] * Bs2[tx][k-ks]; 
          sum11 += As3[ty][k-ks] * Bs3[tx][k-ks]; 
          sum12 += As3[ty][k-ks] * Bs4[tx][k-ks]; 

          sum13 += As4[ty][k-ks] * Bs1[tx][k-ks]; 
          sum14 += As4[ty][k-ks] * Bs2[tx][k-ks]; 
          sum15 += As4[ty][k-ks] * Bs3[tx][k-ks]; 
          sum16 += As4[ty][k-ks] * Bs4[tx][k-ks];  
        }  
        __syncthreads();
      }
      C[i*Nj+j] = sum1;
      C[i*Nj+j+1] = sum2;
      C[i*Nj+j+2] = sum3;
      C[i*Nj+j+3] = sum4;

      C[(i+16)*Nj+j] = sum5;
      C[(i+16)*Nj+j+1] = sum6;
      C[(i+16)*Nj+j+2] = sum7;
      C[(i+16)*Nj+j+3] = sum8;

      C[(i+32)*Nj+j] = sum9;
      C[(i+32)*Nj+j+1] = sum10;
      C[(i+32)*Nj+j+2] = sum11;
      C[(i+32)*Nj+j+3] = sum12;

      C[(i+48)*Nj+j] = sum13;
      C[(i+48)*Nj+j+1] = sum14;
      C[(i+48)*Nj+j+2] = sum15;
      C[(i+48)*Nj+j+3] = sum16;
    }
    }
    else
    {
      int i = blockDim.x*blockIdx.x+threadIdx.x; 
      int j = blockDim.y*blockIdx.y+threadIdx.y;
      int tx = threadIdx.x; 
      int ty = threadIdx.y; 

      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE];
      if ((i < Ni) && (j < Nj)){
      float sum = 0;
      for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
      {
        As[tx][ty] = A[i*Nk+(ks+ty)]; 
        Bs[ty][tx] = B[j*Nk+(ks+tx)];
        __syncthreads(); 
        for (int k = ks; k < ks+BLOCK_SIZE; k++)
          sum += As[tx][k-ks] * Bs[ty][k-ks];   
        __syncthreads();
      }
      C[i*Nj+j] = sum;
      }
    }
}
else
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if(i<Ni && j<Nj)
  {
    float sum = 0;
    for(int k = 0; k < Nk; k++)
    {
      sum += A[i*Nk+k]*B[j*Nk+k];
    }
    C[i*Nj + j] = sum;
  }
}
}

__global__ void aTb_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  if(Ni%BLOCK_SIZE == 0 && Nj%BLOCK_SIZE==0 && Nk%BLOCK_SIZE==0)
  {
    if((Ni == Nj && Nj==Nk)|| (Ni==Nj && Nj>Nk))
    {
      // int i = blockDim.x*blockIdx.x*4+threadIdx.x; 
      // int j = (blockDim.y*blockIdx.y+threadIdx.y)*4;
      // int tx = threadIdx.x; 
      // int ty = threadIdx.y; 
      int j = (blockDim.x * blockIdx.x + threadIdx.x) * 4; 
      int i = blockDim.y*blockIdx.y*4+threadIdx.y;
      int tx = threadIdx.x; 
      int ty = threadIdx.y; 

      __shared__ float As1[BLOCK_SIZE][BLOCK_SIZE], As2[BLOCK_SIZE][BLOCK_SIZE], As3[BLOCK_SIZE][BLOCK_SIZE], As4[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs1[BLOCK_SIZE][BLOCK_SIZE], Bs2[BLOCK_SIZE][BLOCK_SIZE], Bs3[BLOCK_SIZE][BLOCK_SIZE], Bs4[BLOCK_SIZE][BLOCK_SIZE];
      if ((i < Ni) && (j < Nj)){
        float sum1 = 0, sum2=0, sum3=0, sum4=0;
        float sum5 = 0,sum6 = 0, sum7 = 0, sum8 = 0;
        float sum9 = 0,sum10 = 0, sum11 = 0, sum12 = 0;
        float sum13 = 0,sum14 = 0, sum15 = 0, sum16 = 0;

        for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
            {
              // As1[ty][tx] = A[(ks+ty)*Ni+i]; 
              // As2[ty][tx] = A[(ks+ty)*Ni+(i+16)]; 
              // As3[ty][tx] = A[(ks+ty)*Ni+(i+32)]; 
              // As4[ty][tx] = A[(ks+ty)*Ni+(i+48)]; 

              // Bs1[tx][ty] = B[(ks+tx)*Nj+j];
              // Bs2[tx][ty] = B[(ks+tx)*Nj+j+1];
              // Bs3[tx][ty] = B[(ks+tx)*Nj+j+2];
              // Bs4[tx][ty] = B[(ks+tx)*Nj+j+3];
              As1[tx][ty] = A[(ks+tx)*Ni+i];
              As2[tx][ty] = A[(ks+tx)*Ni+(i+16)]; 
              As3[tx][ty] = A[(ks+tx)*Ni+(i+32)]; 
              As4[tx][ty] = A[(ks+tx)*Ni+(i+48)]; 

              Bs1[ty][tx] = B[(ks+ty)*Nj+j];
              Bs2[ty][tx] = B[(ks+ty)*Nj+j+1];
              Bs3[ty][tx] = B[(ks+ty)*Nj+j+2];
              Bs4[ty][tx] = B[(ks+ty)*Nj+j+3];

              __syncthreads(); 
              for (int k = ks; k < ks+BLOCK_SIZE; k++)
              {
                sum1 += As1[k-ks][ty] * Bs1[k-ks][tx];   
                sum2 += As1[k-ks][ty] * Bs2[k-ks][tx];   
                sum3 += As1[k-ks][ty] * Bs3[k-ks][tx];   
                sum4 += As1[k-ks][ty] * Bs4[k-ks][tx]; 

                sum5 += As2[k-ks][ty] * Bs1[k-ks][tx];   
                sum6 += As2[k-ks][ty] * Bs2[k-ks][tx];   
                sum7 += As2[k-ks][ty] * Bs3[k-ks][tx];   
                sum8 += As2[k-ks][ty] * Bs4[k-ks][tx];

                sum9 += As3[k-ks][ty] * Bs1[k-ks][tx];   
                sum10 += As3[k-ks][ty] * Bs2[k-ks][tx];   
                sum11 += As3[k-ks][ty] * Bs3[k-ks][tx];   
                sum12 += As3[k-ks][ty] * Bs4[k-ks][tx];

                sum13 += As4[k-ks][ty] * Bs1[k-ks][tx];   
                sum14 += As4[k-ks][ty] * Bs2[k-ks][tx];   
                sum15 += As4[k-ks][ty] * Bs3[k-ks][tx];   
                sum16 += As4[k-ks][ty] * Bs4[k-ks][tx];  
              }
              __syncthreads();
            }
            C[i*Nj+j] = sum1;
            C[i*Nj+j+1] = sum2;
            C[i*Nj+j+2] = sum3;
            C[i*Nj+j+3] = sum4;

            C[(i+16)*Nj+j] = sum5;
            C[(i+16)*Nj+j+1] = sum6;
            C[(i+16)*Nj+j+2] = sum7;
            C[(i+16)*Nj+j+3] = sum8;

            C[(i+32)*Nj+j] = sum9;
            C[(i+32)*Nj+j+1] = sum10;
            C[(i+32)*Nj+j+2] = sum11;
            C[(i+32)*Nj+j+3] = sum12;

            C[(i+48)*Nj+j] = sum13;
            C[(i+48)*Nj+j+1] = sum14;
            C[(i+48)*Nj+j+2] = sum15;
            C[(i+48)*Nj+j+3] = sum16;
        }
    }
    else
    {
      int i = blockDim.x*blockIdx.x+threadIdx.x; 
      int j = blockDim.y*blockIdx.y+threadIdx.y;
      int tx = threadIdx.x; 
      int ty = threadIdx.y; 

      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
      if ((i < Ni) && (j < Nj)){
        float sum = 0;
        for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
            {
              As[ty][tx] = A[(ks+ty)*Ni+i]; 
              Bs[tx][ty] = B[(ks+tx)*Nj+j];
              __syncthreads(); 
              for (int k = ks; k < ks+BLOCK_SIZE; k++)
              {
                sum += As[k-ks][tx] * Bs[k-ks][ty];     
              }
              __syncthreads();
            }
            C[i*Nj+j] = sum;
        }
    }
  }
else
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if(i<Ni && j<Nj)
  {
    float sum = 0;
    for(int k = 0; k < Nk; k++)
    {
      sum += A[k*Ni+i]*B[k*Nj+j];
    }
    C[i*Nj + j] = sum;
  }
}
}

__global__ void aTbT_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{

  if(Ni%BLOCK_SIZE == 0 && Nj%BLOCK_SIZE==0 && Nk%BLOCK_SIZE==0)
  {
    if((Ni==Nj && Nj==Nk)|| (Ni==Nj && Nj>Nk))
    {
      int i = blockDim.x * blockIdx.x * 4 + threadIdx.x; 
      int j = (blockDim.y * blockIdx.y + threadIdx.y) * 4;
      int tx = threadIdx.x; 
      int ty = threadIdx.y; 

      __shared__ float As1[BLOCK_SIZE][BLOCK_SIZE], As2[BLOCK_SIZE][BLOCK_SIZE], As3[BLOCK_SIZE][BLOCK_SIZE], As4[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs1[BLOCK_SIZE][BLOCK_SIZE], Bs2[BLOCK_SIZE][BLOCK_SIZE], Bs3[BLOCK_SIZE][BLOCK_SIZE], Bs4[BLOCK_SIZE][BLOCK_SIZE];

      if ((i < Ni) && (j < Nj)){

          float sum1 = 0,sum2 = 0, sum3 = 0, sum4 = 0;
          float sum5 = 0,sum6 = 0, sum7 = 0, sum8 = 0;
          float sum9 = 0,sum10 = 0, sum11 = 0, sum12 = 0;
          float sum13 = 0,sum14 = 0, sum15 = 0, sum16 = 0;

          for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
          {
            As1[ty][tx] = A[(ks+ty)*Ni+i]; 
            As2[ty][tx] = A[(ks+ty)*Ni+i+16]; 
            As3[ty][tx] = A[(ks+ty)*Ni+i+32]; 
            As4[ty][tx] = A[(ks+ty)*Ni+i+48]; 

            Bs1[ty][tx] = B[j*Nk+(ks+tx)];
            Bs2[ty][tx] = B[(j+1)*Nk+(ks+tx)];
            Bs3[ty][tx] = B[(j+2)*Nk+(ks+tx)];
            Bs4[ty][tx] = B[(j+3)*Nk+(ks+tx)];

            __syncthreads(); 

            for (int k = ks; k < ks+BLOCK_SIZE; k++)
            {
              sum1 += As1[k-ks][tx] * Bs1[ty][k-ks]; 
              sum2 += As1[k-ks][tx] * Bs2[ty][k-ks]; 
              sum3 += As1[k-ks][tx] * Bs3[ty][k-ks]; 
              sum4 += As1[k-ks][tx] * Bs4[ty][k-ks]; 

              sum5 += As2[k-ks][tx] * Bs1[ty][k-ks]; 
              sum6 += As2[k-ks][tx] * Bs2[ty][k-ks]; 
              sum7 += As2[k-ks][tx] * Bs3[ty][k-ks]; 
              sum8 += As2[k-ks][tx] * Bs4[ty][k-ks]; 

              sum9 += As3[k-ks][tx] * Bs1[ty][k-ks]; 
              sum10 += As3[k-ks][tx] * Bs2[ty][k-ks]; 
              sum11 += As3[k-ks][tx] * Bs3[ty][k-ks]; 
              sum12 += As3[k-ks][tx] * Bs4[ty][k-ks]; 

              sum13 += As4[k-ks][tx] * Bs1[ty][k-ks]; 
              sum14 += As4[k-ks][tx] * Bs2[ty][k-ks]; 
              sum15 += As4[k-ks][tx] * Bs3[ty][k-ks]; 
              sum16 += As4[k-ks][tx] * Bs4[ty][k-ks]; 
            }
            __syncthreads();
          }
          C[i*Nj+j] = sum1;
          C[i*Nj+j+1] = sum2;
          C[i*Nj+j+2] = sum3;
          C[i*Nj+j+3] = sum4;

          C[(i+16)*Nj+j] = sum5;
          C[(i+16)*Nj+j+1] = sum6;
          C[(i+16)*Nj+j+2] = sum7;
          C[(i+16)*Nj+j+3] = sum8;

          C[(i+32)*Nj+j] = sum9;
          C[(i+32)*Nj+j+1] = sum10;
          C[(i+32)*Nj+j+2] = sum11;
          C[(i+32)*Nj+j+3] = sum12;

          C[(i+48)*Nj+j] = sum13;
          C[(i+48)*Nj+j+1] = sum14;
          C[(i+48)*Nj+j+2] = sum15;
          C[(i+48)*Nj+j+3] = sum16;
      }
    }
    else
    {
      int i = blockDim.x*blockIdx.x+threadIdx.x; 
      int j = blockDim.y*blockIdx.y+threadIdx.y;
      int tx = threadIdx.x; 
      int ty = threadIdx.y; 

      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE];
      if ((i < Ni) && (j < Nj)){
        float sum = 0;
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
    }
  else
  {
    int i = blockDim.x*blockIdx.x+threadIdx.x; 
    int j = blockDim.y*blockIdx.y+threadIdx.y;
    if(i<Ni && j<Nj)
    {
    float sum = 0;
    for(int k = 0; k < Nk; k++)
    {
      sum += A[k*Ni + i] * B[ j*Nk+k];
    }
    C[i*Nj + j] = sum;
    }
  }
}