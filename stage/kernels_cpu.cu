#include "kernel_cpu.h"

double atomicADD_CPU(double *address, double val) {
  double assumed, old = *address;
  do {
      assumed = old;
      old = __longlong_as_double(atomicCAS((unsigned long long int *)address, __double_as_longlong(assumed), __double_as_longlong(val + assumed)));
  } while (assumed != old);
  return old;
}

void SetFieldsToCells_CPU(GPUCell **cells, double *Ex, double *Ey, double *Ez, double *Hx, double *Hy, double *Hz){

}

void MakeDepartureLists_CPU(GPUCell **cells, int *d_stage){

}

void copyCellDouble_CPU(CellDouble *dst, CellDouble *src, unsigned int n) {

}
void copyFieldsToSharedMemory_CPU(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        CellDouble *c_ex,
        CellDouble *c_ey,
        CellDouble *c_ez,
        CellDouble *c_hx,
        CellDouble *c_hy,
        CellDouble *c_hz,
        Cell *c,
        int index,
        dim3 blockId,
        int blockDimX
){

}
void AccumulateCurrentWithParticlesInCell_CPU(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index, int blockDimX){

}

void CurrentsAllCells_CPU(GPUCell **cells){

}

void emh1_Element_CPU(Cell *c, int3 i, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2) {

}

void periodicElement_CPU(Cell *c, int i, int k, double *E, int dir, int to, int from){

}
 void CurrentPeriodic_CPU(GPUCell **cells, double *E, int dirE, int dir, int i_s, int k_s, int N){

 }
