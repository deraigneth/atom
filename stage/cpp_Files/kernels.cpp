#include "../header_Files/kernels.h"



double cuda_atomicAdd(double *address, double val){
  double assumed, old = *address ;
  do {
    assumed = old ;
    old = __longlong_as_double(atomicCAS((unsigned long long int *)address, __double_as_longlong(assumed), __double_as_longlong(val + assumed)));
  } while (assumed != old);
  return old;
}


void CPU_getCellEnergy(CPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez) {
    unsigned int i = blockIdx.x;
    unsigned int l = blockIdx.y;
    unsigned int k = blockIdx.z;
    Cell *c0 = cells[0], nc;
    double t, ex, ey, ez;

    int n = c0->getGlobalCellNumber(i, l, k);

    ex = d_Ex[n];
    ey = d_Ey[n];
    ez = d_Ez[n];

    t = ex * ex + ey * ey + ez * ez;

    cuda_atomicAdd(d_ee, t);
}
