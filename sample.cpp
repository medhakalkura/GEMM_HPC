#include <iostream>
using namespace std;
int main()
{
    int ni,nj,nk;
    cout<<"Enter matrix sizes ";
    cin>>ni>>nj>>nk;
    int A[ni][nk], B[nk][nj], C[ni][nj];
    int x=1;
    for(int i=0;i<ni;i++)
        for(int k=0;k<nk;k++)
        A[i][k]=x++;
    x=1;
    for(int k=0;k<nk;k++)
        for(int j=0;j<nj;j++)
        B[k][j]=x++;
    for(int i=0;i<ni;i++)
        for(int j=0;j<nj;j++)
        C[i][j]=0;
    for(int i=0;i<ni;i++)
        for(int j=0;j<nj;j++)
            for(int k=0;k<nk;k++)
                C[i][j]=C[i][j]+A[k][i]*B[j][k];

    cout<<"\nA:\n";
    for(int i=0;i<ni;i++)
    {
        cout<<"\n";
        for(int k=0;k<nk;k++)
            cout<<A[i][k]<<" ";
    }
    cout<<"\n\nB:\n";
    for(int k=0;k<nk;k++)
    {
        cout<<"\n";
        for(int j=0;j<nj;j++)
            cout<<B[k][j]<<" ";
    }
    cout<<"\n\nC:\n";
    for(int i=0;i<ni;i++)
    {
        cout<<"\n";
        for(int j=0;j<nj;j++)
            cout<<C[i][j]<<" ";
    }
    return 0;
} 
