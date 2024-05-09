#include <stdio.h>
#include <math.h>
#include <omp.h>

//input sizes : 8192 8192 16
//input sizes : 1024 1024 1024
//input sizes : 16 16 4194304

#define min(a, b) (((a) < (b)) ? (a) : (b))

void ab_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
  int tile_size = 16;
  int it, jt, kt;

if(Ni == Nj && Nk < Ni)
{
   #pragma omp parallel for private(i,j,k) shared(A, B, C)
   for (i = 0; i < Ni; i++)
   {
      int rem1 = Nk%4;
         for (k = 0; k < rem1; k++)
            for (j = 0; j < Nj; j++)
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
         for (k = rem1; k < Nk; k+=4)
         {
            for (j = 0; j < Nj; j++)
            {
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+(k+1)]*B[(k+1)*Nj+j];
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+(k+2)]*B[(k+2)*Nj+j];
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+(k+3)]*B[(k+3)*Nj+j];
            }
         }
      }
   // #pragma omp parallel for schedule(dynamic)
   // for (i = 0; i < Ni; i++)
   //    for (k = 0; k < Nk; k++)
   //       for (j = 0; j < Nj; j++)
   //          C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
}
else if(Ni == Nj && Nk == Nj)
{
   // #pragma omp parallel for private(kt,i,j,k)
   // for (kt = 0; kt < Nk; kt+=tile_size)
   //    for (i = 0; i < Ni; i++)
   //       for (k = kt; k < min(kt+tile_size, Nk); k++)
   //          for (j = 0; j < Nj; j++)
   //             C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];

   #pragma omp parallel for private(i,j,k) shared(A, B, C)
   for (i = 0; i < Ni; i++)
   {
      int rem1 = Nk%4;
         for (k = 0; k < rem1; k++)
            for (j = 0; j < Nj; j++)
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
         for (k = rem1; k < Nk; k+=4)
         {
            for (j = 0; j < Nj; j++)
            {
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+(k+1)]*B[(k+1)*Nj+j];
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+(k+2)]*B[(k+2)*Nj+j];
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+(k+3)]*B[(k+3)*Nj+j];
            }
         }
   }
   
}
else if(Ni == Nj && Nk > Ni)
{
   #pragma omp parallel for private(i,j,k) shared(A, B, C)
   for (i = 0; i < Ni; i++)
   {
      int rem1 = Nk%4;
         for (k = 0; k < rem1; k++)
            for (j = 0; j < Nj; j++)
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
         for (k = rem1; k < Nk; k+=4)
         {
            for (j = 0; j < Nj; j++)
            {
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+(k+1)]*B[(k+1)*Nj+j];
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+(k+2)]*B[(k+2)*Nj+j];
               C[i*Nj+j] = C[i*Nj+j]+A[i*Nk+(k+3)]*B[(k+3)*Nj+j];
            }
         }
      }
   }
}

void abT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
  int tile_size = 16;
  int it, jt, kt;

//   #pragma omp parallel for schedule(dynamic)
//   for (it = 0; it < Ni; it+=tile_size)
//      for (jt = 0; jt < Nj; jt+=tile_size)
//         for (kt = 0; kt < Nk; kt+=tile_size)
//            for (i = it; i < min(it+tile_size, Ni); i++)
//               for (j = jt; j < min(jt+tile_size, Nj); j++)
//                  for (k = kt; k < min(kt+tile_size, Nk); k++)
//                     // C[i][j] = C[i][j] + A[i][k]*B[j][k];
//                     C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];


if(Ni == Nj && Nk < Ni)
{

        #pragma omp parallel for
        for (it = 0; it < Ni; it+=tile_size)
         for (jt = 0; jt < Nj; jt+=tile_size)
          for (kt = 0; kt < Nk; kt+=tile_size)
           for (i = it; i < min(it+tile_size, Ni); i++)
            for (j = jt; j < min(jt+tile_size, Nj); j++)
             for (k = kt; k < min(kt+tile_size, Nk); k++)
              // C[i][j] = C[i][j] + A[i][k]*B[j][k];
              C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];

   // #pragma omp parallel for private(i,j,k)
   // for (i = 0; i < Ni; i++)
   // {
   //    int rem1 = Nk%4;
   //       for (k = 0; k < rem1; k++)
   //          for (j = 0; j < Nj; j++)
   //             C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
   //       for (k = rem1; k < Nk; k+=4)
   //       {
   //          for (j = 0; j < Nj; j++)
   //          {
   //             C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
   //             C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+1)]*B[j*Nk+(k+1)];
   //             C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+2)]*B[j*Nk+(k+2)];
   //             C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+3)]*B[j*Nk+(k+3)];
   //          }
   //       }
   //    }

}
else if(Ni == Nj && Nk == Ni)
{
   // int rem = Ni%4;
   // for(i=0;i<rem;i++)
   // {
   //    for (j = 0; j < Nj; j++)
   //       for (k = 0; k < Nk; k++)
   //       {
   //          C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
   //       }
   // }
   // #pragma omp parallel for private(i,j,k)
   // for (i = rem; i < Ni; i+=4)
   //    for (j = 0; j < Nj; j++)
   //       for (k = 0; k < Nk; k++)
   //       {
   //          C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
   //          C[(i+1)*Nj+j]=C[(i+1)*Nj+j]+A[(i+1)*Nk+k]*B[j*Nk+k];
   //          C[(i+2)*Nj+j]=C[(i+2)*Nj+j]+A[(i+2)*Nk+k]*B[j*Nk+k];
   //          C[(i+3)*Nj+j]=C[(i+3)*Nj+j]+A[(i+3)*Nk+k]*B[j*Nk+k];
   //       }
   //   #pragma omp parallel for
   //    for (it = 0; it < Ni; it+=tile_size)
   //       for (jt = 0; jt < Nj; jt+=tile_size)
   //          for (kt = 0; kt < Nk; kt+=tile_size)
   //             for (i = it; i < min(it+tile_size, Ni); i++)
   //                for (j = jt; j < min(jt+tile_size, Nj); j++)
   //                   for (k = kt; k < min(kt+tile_size, Nk); k++)
   //                      // C[i][j] = C[i][j] + A[i][k]*B[j][k];
   //                      C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];

      #pragma omp parallel for
      for (it = 0; it < Ni; it+=tile_size)
         for (i = it; i < min(it+tile_size, Ni); i++)
         {
            int rem1 = Nj%4;
            for (j = 0; j < rem1; j++)
               for (k = 0; k < Nk; k++)
                  C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
            
            for (j = 0; j < Nj; j+=4)
               for (k = 0; k < Nk; k++)
               {
                  // C[i][j] = C[i][j] + A[i][k]*B[j][k];
                  C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
                  C[i*Nj+j+1]=C[i*Nj+j+1]+A[i*Nk+k]*B[(j+1)*Nk+k];
                  C[i*Nj+j+2]=C[i*Nj+j+2]+A[i*Nk+k]*B[(j+2)*Nk+k];
                  C[i*Nj+j+3]=C[i*Nj+j+3]+A[i*Nk+k]*B[(j+3)*Nk+k];
               }
         }
}
else if(Ni == Nj && Nk > Ni)
{  
   // #pragma omp parallel for schedule(dynamic)
   // for (i = 0; i < Ni; i++)
   //    for (j = 0; j < Nj; j++)
   //       for (k = 0; k < Nk; k++)
   //          C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];

   #pragma omp parallel for private(i,j,k)
   for (i = 0; i < Ni; i++)
   {
      int rem1 = Nk%4;
         for (k = 0; k < rem1; k++)
            for (j = 0; j < Nj; j++)
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
         for (k = rem1; k < Nk; k+=4)
         {
            for (j = 0; j < Nj; j++)
            {
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+1)]*B[j*Nk+(k+1)];
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+2)]*B[j*Nk+(k+2)];
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+3)]*B[j*Nk+(k+3)];
            }
         }
      }
}
else
{
   #pragma omp parallel for schedule(dynamic)
   for (i = 0; i < Ni; i++)
      for (j = 0; j < Nj; j++)
         for (k = 0; k < Nk; k++)
            C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
}
}

void aTb_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
  int tile_size = 16;
  int it, jt, kt;

//   #pragma omp parallel for schedule(dynamic)
//   for (it = 0; it < Ni; it+=tile_size)
//      for (jt = 0; jt < Nj; jt+=tile_size)
//         for (kt = 0; kt < Nk; kt+=tile_size)
//            for (i = it; i < min(it+tile_size, Ni); i++)
//               for (j = jt; j < min(jt+tile_size, Nj); j++)
//                  for (k = kt; k < min(kt+tile_size, Nk); k++)
//                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];

   // #pragma omp parallel for schedule(dynamic)
   //    for (k = 0; k < Nk; k++)
   //       for (i = 0; i < Ni; i++)
   //          for (j = 0; j < Nj; j++)
   //             C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];

if(Ni == Nj && Nk < Ni)
{
   #pragma omp parallel for private(i,j,k) shared(A, B, C)
   for (i = 0; i < Ni; i++)
      for (k = 0; k < Nk; k++)
         for (j = 0; j < Nj; j++)
            C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
}
else if(Ni == Nj && Nk == Ni) 
{ 
   // #pragma omp parallel for schedule(dynamic) private(i,j,k) shared(A,B,C)
   // for (i = 0; i < Ni; i++)
   //    for (k = 0; k < Nk; k++)
   //       for (j = 0; j < Nj; j++)
   //          C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];

//   #pragma omp parallel for schedule(dynamic)
//   for (it = 0; it < Ni; it+=tile_size)
//      for (jt = 0; jt < Nj; jt+=tile_size)
//         for (kt = 0; kt < Nk; kt+=tile_size)
//            for (i = it; i < min(it+tile_size, Ni); i++)
//               for (j = jt; j < min(jt+tile_size, Nj); j++)
//                  for (k = kt; k < min(kt+tile_size, Nk); k++)
//                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];

// #pragma omp parallel for private(i,j,k) shared(A, B, C)
//    for (i = 0; i < Ni; i++)
//    {
//       int rem1 = k%4;
//          for (k = 0; k < rem1; k++)
//             for (j = 0; j < Nj; j++)
//                C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
//          for (k = rem1; k < Nk; k+=4)
//          {
//             for (j = 0; j < Nj; j++)
//             {
//                C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
//                C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
//                C[i*Nj+j]=C[i*Nj+j]+A[(k+2)*Ni+i]*B[(k+2)*Nj+j];
//                C[i*Nj+j]=C[i*Nj+j]+A[(k+3)*Ni+i]*B[(k+3)*Nj+j];
//             }
//          }
//       }
 #pragma omp parallel for private(i,j,k) shared(A, B, C)
   for (i = 0; i < Ni; i++)
      for (k = 0; k < Nk; k++)
         for (j = 0; j < Nj; j++)
            C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
}
else if(Ni == Nj && Nk > Ni)
{  
   // #pragma omp parallel for schedule(dynamic) private(i,j,k) shared(A,B,C)
   // for (i = 0; i < Ni; i++)
   //    for (k = 0; k < Nk; k++)
   //       for (j = 0; j < Nj; j++)
   //          C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];

   #pragma omp parallel for private(i,j,k) shared(A, B, C)
   for (i = 0; i < Ni; i++)
   {
      int rem1 = k%4;
         for (k = 0; k < rem1; k++)
            for (j = 0; j < Nj; j++)
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
         for (k = rem1; k < Nk; k+=4)
         {
            for (j = 0; j < Nj; j++)
            {
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+2)*Ni+i]*B[(k+2)*Nj+j];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+3)*Ni+i]*B[(k+3)*Nj+j];
            }
         }
      }
}
else
{
   #pragma omp parallel for private(i,j,k) shared(A, B, C)
   for (i = 0; i < Ni; i++)
      for (k = 0; k < Nk; k++)
         for (j = 0; j < Nj; j++)
            C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
}
}

void aTbT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
  int tile_size = 16;
  int it, jt, kt;

if(Ni == Nj && Nk < Ni)
{

      //   #pragma omp parallel for schedule(dynamic)
      //   for (it = 0; it < Ni; it+=tile_size)
      //    for (jt = 0; jt < Nj; jt+=tile_size)
      //     for (kt = 0; kt < Nk; kt+=tile_size)
      //      for (i = it; i < min(it+tile_size, Ni); i++)
      //       for (j = jt; j < min(jt+tile_size, Nj); j++)
      //        for (k = kt; k < min(kt+tile_size, Nk); k++)
      //         // C[i][j] = C[i][j] + A[k][i]*B[j][k];
      //         C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
      #pragma omp parallel for private(i,j,k)
      for (i = 0; i < Ni; i++)
      {
      int rem1 = Nj%4;
         for (j = 0; j < rem1; j++)
            for (k = 0; k < Nk; k++)
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];

         for (j = rem1; j < Nj; j+=4)
            for (k = 0; k < Nk; k++)
            {
                  C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
                  C[i*Nj+j+1]=C[i*Nj+j+1]+A[k*Ni+i]*B[(j+1)*Nk+k];
                  C[i*Nj+j+2]=C[i*Nj+j+2]+A[k*Ni+i]*B[(j+2)*Nk+k];
                  C[i*Nj+j+3]=C[i*Nj+j+3]+A[k*Ni+i]*B[(j+3)*Nk+k];
            }
      }
}
else if(Ni == Nj && Nk == Ni)
{
   // #pragma omp parallel for schedule(dynamic) private(i,j,k) shared(A,B,C)
   // for (k = 0; k < Nk; k++)
   // for (j = 0; j < Nj; j++)
   // for (i = 0; i < Ni; i++)
   //          C[k*Nj+j]=C[k*Nj+j]+A[k*Ni+i]*B[j*Nk+k];

// #pragma omp parallel for schedule(dynamic) private(i,j,k) shared(A,B,C)
// for (int k = 0; k < Nk; ++k) {
//     for (int j = 0; j < Nj; ++j) {
//         C[k*Nj+j] = 0.0;
//         for (int i = 0; i < Ni; ++i) {
//             C[k*Nj+j] += A[i*Nk+k] * B[j*Ni+i];
//         }
//     }
// }
// #pragma omp parallel for schedule(dynamic) private(i,j,k) shared(A,B,C)
// for (int j = 0; j < Nj; ++j) {
//     for (int k = 0; k < Nk; ++k) {
//         C[j*Nk+k] = 0.0;
//         for (int i = 0; i < Ni; ++i) {
//             C[j*Nk+k] += A[i*Nj+j] * B[k*Ni+i];
//         }
//     }
// }
// #pragma omp parallel for schedule(dynamic) private(i,j,k) shared(A,B,C)
// for (int j = 0; j < Nj; ++j) {
//    for (int i = 0; i < Ni; ++i) {
//       C[j*Ni+i] = 0.0;
//     for (int k = 0; k < Nk; ++k) {
//       C[j*Ni+i] += A[k*Nj+j] * B[i*Nk+k];
//       }
//     }
// }
// #pragma omp parallel for schedule(dynamic) private(i,j,k) shared(A,B,C)
// for (i = 0; i < Ni; i++)
//    for (j = 0; j < Nj; j++)
//       for (k = 0; k < Nk; k++)
//          C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];

      //   #pragma omp parallel for schedule(dynamic)
      //   for (it = 0; it < Ni; it+=tile_size)
      //    for (jt = 0; jt < Nj; jt+=tile_size)
      //     for (kt = 0; kt < Nk; kt+=tile_size)
      //      for (i = it; i < min(it+tile_size, Ni); i++)
      //       for (j = jt; j < min(jt+tile_size, Nj); j++)
      //        for (k = kt; k < min(kt+tile_size, Nk); k++)
      //         // C[i][j] = C[i][j] + A[k][i]*B[j][k];
      //         C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];

   #pragma omp parallel for private(i,j,k)
   for (i = 0; i < Ni; i++)
   {
      int rem1 = Nk%4;
         for (k = 0; k < rem1; k++)
            for (j = 0; j < Nj; j++)
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
         for (k = rem1; k < Nk; k+=4)
         {
            for (j = 0; j < Nj; j++)
            {
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[j*Nk+(k+1)];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+2)*Ni+i]*B[j*Nk+(k+2)];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+3)*Ni+i]*B[j*Nk+(k+3)];
            }
         }
      }
}
else if(Ni == Nj && Nk > Ni)
{  
   // #pragma omp parallel for schedule(dynamic)
   //      for (it = 0; it < Ni; it+=tile_size)
   //       for (jt = 0; jt < Nj; jt+=tile_size)
   //        for (kt = 0; kt < Nk; kt+=tile_size)
   //         for (i = it; i < min(it+tile_size, Ni); i++)
   //          for (j = jt; j < min(jt+tile_size, Nj); j++)
   //           for (k = kt; k < min(kt+tile_size, Nk); k++)
   //            // C[i][j] = C[i][j] + A[k][i]*B[j][k];
   //            C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];

   #pragma omp parallel for private(i,j,k)
   for (i = 0; i < Ni; i++)
   {
      int rem1 = Nk%4;
         for (k = 0; k < rem1; k++)
            for (j = 0; j < Nj; j++)
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
         for (k = rem1; k < Nk; k+=4)
         {
            for (j = 0; j < Nj; j++)
            {
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[j*Nk+(k+1)];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+2)*Ni+i]*B[j*Nk+(k+2)];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+3)*Ni+i]*B[j*Nk+(k+3)];
            }
         }
      }

   // #pragma omp parallel for private(i,j,k)
   // for (i = 0; i < Ni; i++)
   // {
   //    int rem1 = Nj%4;
   //       for (j = 0; j < rem1; j+=4)
   //          for (k = 0; k < Nk; k++)
   //             C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];

   //       for (j = rem1; j < Nj; j+=4)
   //          for (k = 0; k < Nk; k++)
   //          {
   //                C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
   //                C[i*Nj+j+1]=C[i*Nj+j+1]+A[k*Ni+i]*B[(j+1)*Nk+k];
   //                C[i*Nj+j+2]=C[i*Nj+j+2]+A[k*Ni+i]*B[(j+2)*Nk+k];
   //                C[i*Nj+j+3]=C[i*Nj+j+3]+A[k*Ni+i]*B[(j+3)*Nk+k];
   //          }
   //    }
}
else
{
   #pragma omp parallel for schedule(dynamic)
        for (it = 0; it < Ni; it+=tile_size)
         for (jt = 0; jt < Nj; jt+=tile_size)
          for (kt = 0; kt < Nk; kt+=tile_size)
           for (i = it; i < min(it+tile_size, Ni); i++)
            for (j = jt; j < min(jt+tile_size, Nj); j++)
             for (k = kt; k < min(kt+tile_size, Nk); k++)
              // C[i][j] = C[i][j] + A[k][i]*B[j][k];
              C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
}
}
