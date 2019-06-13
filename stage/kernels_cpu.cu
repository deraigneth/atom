#include "kernel_cpu.h"

double atomicADD(double *address, double val) {
  double assumed, old = *address;
  do {
      assumed = old;
      old = __longlong_as_double(atomicCAS((unsigned long long int *)address, __double_as_longlong(assumed), __double_as_longlong(val + assumed)));
  } while (assumed != old);
  return old;
}

void SetAllCurrentsToZero_CPU(GPUCell **cells) {

}

void WriteControlSystem_CPU(Cell **cells) {

}

void writeCurrentComponent_CPU(CellDouble *J, CurrentTensorComponent *t1, CurrentTensorComponent *t2, int pqr2) {

}

void assignSharedWithLocal_CPU(
        CellDouble **c_jx,
        CellDouble **c_jy,
        CellDouble **c_jz,
        CellDouble **c_ex,
        CellDouble **c_ey,
        CellDouble **c_ez,
        CellDouble **c_hx,
        CellDouble **c_hy,
        CellDouble **c_hz,
        CellDouble *fd) {

}

void MoveParticlesInCell_CPU(Cell *c, int index, int blockDimX) {

}

void StepAllCells_CPU(GPUCell **cells) {

}


void emh2_CPU(GPUCell **cells, int i_s, int l_s, int k_s, double *Q, double *H) {

}


void emeElement_CPU(Cell *c, int3 i, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2) {

}

void periodicCurrentElement_CPU(Cell *c, int i, int k, double *E, int dir, int dirE, int N) {

}
