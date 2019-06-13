#include "kernel_cpu.h"

double atomicADD_CPU(double *address, double val) {
  double assumed, old = *address;
  do {
      assumed = old;
      old = __longlong_as_double(atomicCAS((unsigned long long int *)address, __double_as_longlong(assumed), __double_as_longlong(val + assumed)));
  } while (assumed != old);
  return old;
}

void getCellEnergy_CPU(GPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez){
  // à voir plus tard
}

void writeAllCurrents_CPU(GPUCell **cells, int n0, double *jx, double *jy, double *jz, double *rho){
  //à voir plus tard
}

void setCellDoubleToZero_CPU(CellDouble *dst, unsigned int n){
  if (n < CellExtent * CellExtent * CellExtent) {
      double *d_dst;

      d_dst = (double *) (dst->M);
      d_dst[n] = 0.0;
  }//a regarder
}

void set_cell_double_arrays_to_zero_CPU(CellDouble *, CellDouble *, CellDouble *, int, int, int){
  //à completer

}

void copyFromSharedMemoryToCell_CPU(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        Cell *c,
        int index
){
  //à completer
}

void emh2_Element_CPU(Cell *c, int i, int l, int k, double *Q, double *H){
  // à compléter
}

void emh1_CPU(GPUCell **cells, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2){
  //à compléter
}

void periodicCurrentElement_CPU(Cell *c, int i, int k, double *E, int dir, int dirE, int N){
  //à compléter
}
