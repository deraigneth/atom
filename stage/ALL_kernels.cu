#ifdef __CUDACC__
double cuda_atomicAdd(double *address, double val) {
  atomicAdd_GPU(double *address, double val) ;
}
#else
double cuda_atomicAdd(double *address, double val) {
  atomicADD_CPU(double *address, double val) ;
}
#endif

#ifdef __CUDACC__
void GPU_SetFieldsToCells(GPUCell **cells, double *Ex, double *Ey, double *Ez, double *Hx, double *Hy, double *Hz){
  SetFieldsToCells_GPU(GPUCell **cells, double *Ex, double *Ey, double *Ez, double *Hx, double *Hy, double *Hz) ;
}
#else
void GPU_SetFieldsToCells(GPUCell **cells, double *Ex, double *Ey, double *Ez, double *Hx, double *Hy, double *Hz){
  SetFieldsToCells_CPU(GPUCell **cells, double *Ex, double *Ey, double *Ez, double *Hx, double *Hy, double *Hz) ;
}
#endif

#ifdef __CUDACC__
void GPU_MakeDepartureLists(GPUCell **cells, int *d_stage){
  MakeDepartureLists_GPU(GPUCell **cells, int *d_stage) ;
}
#else
void GPU_MakeDepartureLists(GPUCell **cells, int *d_stage){
  MakeDepartureLists_CPU(GPUCell **cells, int *d_stage);
}
#endif

#ifdef __CUDACC__
void copyCellDouble(CellDouble *dst, CellDouble *src, unsigned int n){
  copyCellDouble_GPU(CellDouble *dst, CellDouble *src, unsigned int n);
}
#else
void copyCellDouble(CellDouble *dst, CellDouble *src, unsigned int n){
  copyCellDouble_CPU(CellDouble *dst, CellDouble *src, unsigned int n) ;
}
#endif

#ifdef __CUDACC__
void copyFieldsToSharedMemory(
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
  copyFieldsToSharedMemory_GPU(
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
  );
}
#else
void copyFieldsToSharedMemory(
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
  copyFieldsToSharedMemory_CPU(
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
  );
}
#endif

#ifdef __CUDACC__
void AccumulateCurrentWithParticlesInCell(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index, int blockDimX){
  AccumulateCurrentWithParticlesInCell_GPU(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index, int blockDimX);
}
#else
void AccumulateCurrentWithParticlesInCell(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index, int blockDimX){
  AccumulateCurrentWithParticlesInCell_CPU(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index, int blockDimX);
}
#endif

#ifdef __CUDACC__
void GPU_CurrentsAllCells(GPUCell **cells){
  void CurrentsAllCells_GPU(GPUCell **cells);
}
#else
void GPU_CurrentsAllCells(GPUCell **cells){
  void CurrentsAllCells_CPU(GPUCell **cells);
}
#endif

#ifdef __CUDACC__
void emh1_Element(Cell *, int3, double *, double *, double *, double *, double, double, int3, int3){
  emh1_Element_GPU(Cell *c, int3 i, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2) ;
}
#else
void emh1_Element(Cell *, int3, double *, double *, double *, double *, double, double, int3, int3){
  emh1_Element_CPU(Cell *c, int3 i, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2) ;
}
#endif

#ifdef __CUDACC__
void periodicElement(Cell *c, int i, int k, double *E, int dir, int to, int from){
  periodicElement_GPU(Cell *c, int i, int k, double *E, int dir, int to, int from) ;
}
#else
void periodicElement(Cell *c, int i, int k, double *E, int dir, int to, int from){
  periodicElement_CPU(Cell *c, int i, int k, double *E, int dir, int to, int from);
}

#ifdef __CUDACC__
void GPU_CurrentPeriodic(GPUCell **cells, double *E, int dirE, int dir, int i_s, int k_s, int N){
  CurrentPeriodic_GPU(GPUCell **cells, double *E, int dirE, int dir, int i_s, int k_s, int N);
}
#else
void GPU_CurrentPeriodic(GPUCell **cells, double *E, int dirE, int dir, int i_s, int k_s, int N){
  CurrentPeriodic_CPU(GPUCell **cells, double *E, int dirE, int dir, int i_s, int k_s, int N);
}
