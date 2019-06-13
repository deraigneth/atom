

#ifdef __CUDACC__

void GPU_SetAllCurrentsToZero(GPUCell **cells){
  void SetAllCurrentsToZero_GPU(GPUCell **cells);
}
#else
void GPU_SetAllCurrentsToZero(GPUCell **){
  void SetAllCurrentsToZero_CPU(GPUCell **cells);
}
#endif

#ifdef __CUDACC__
void GPU_WriteControlSystem(Cell **cells){
  WriteControlSystem_GPU(Cell **cells);
}
#else
void GPU_WriteControlSystem(Cell **cells){
  WriteControlSystem_CPU(Cell **cells);
}
#endif

#ifdef __CUDACC__
void writeCurrentComponent(CellDouble *J, CurrentTensorComponent *t1, CurrentTensorComponent *t2, int pqr2){
  void writeCurrentComponent_GPU(CellDouble *J, CurrentTensorComponent *t1, CurrentTensorComponent *t2, int pqr2);
}
#else
void writeCurrentComponent(CellDouble *J, CurrentTensorComponent *t1, CurrentTensorComponent *t2, int pqr2){
  void writeCurrentComponent_CPU(CellDouble *J, CurrentTensorComponent *t1, CurrentTensorComponent *t2, int pqr2);
}
#endif

#ifdef __CUDACC__
void assignSharedWithLocal(
        CellDouble **c_jx,
        CellDouble **c_jy,
        CellDouble **c_jz,
        CellDouble **c_ex,
        CellDouble **c_ey,
        CellDouble **c_ez,
        CellDouble **c_hx,
        CellDouble **c_hy,
        CellDouble **c_hz,
        CellDouble *fd){
          void assignSharedWithLocal_GPU(
                  CellDouble **c_jx,
                  CellDouble **c_jy,
                  CellDouble **c_jz,
                  CellDouble **c_ex,
                  CellDouble **c_ey,
                  CellDouble **c_ez,
                  CellDouble **c_hx,
                  CellDouble **c_hy,
                  CellDouble **c_hz,
                  CellDouble *fd);
        }
#else
void assignSharedWithLocal(
        CellDouble **c_jx,
        CellDouble **c_jy,
        CellDouble **c_jz,
        CellDouble **c_ex,
        CellDouble **c_ey,
        CellDouble **c_ez,
        CellDouble **c_hx,
        CellDouble **c_hy,
        CellDouble **c_hz,
        CellDouble *fd){
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
                  CellDouble *fd);
        }
#endif

#ifdef __CUDACC__
void MoveParticlesInCell(Cell *c, int index, int blockDimX){
  void MoveParticlesInCell_GPU(Cell *c, int index, int blockDimX);
}
#else
void MoveParticlesInCell(Cell *c, int index, int blockDimX){
  void MoveParticlesInCell_CPU(Cell *c, int index, int blockDimX);
}
#endif


#ifdef __CUDACC__
void GPU_StepAllCells(GPUCell **cells){
  void StepAllCells_GPU(GPUCell **cells);
}
#else
void GPU_StepAllCells(GPUCell **cells){
  void StepAllCells_CPU(GPUCell **cells);
}
#endif

#ifdef __CUDACC__
void GPU_emh2(GPUCell **cells, int i_s, int l_s, int k_s, double *Q, double *H){
  void emh2_GPU(GPUCell **cells, int i_s, int l_s, int k_s, double *Q, double *H);
}
#else
void GPU_emh2(GPUCell **cells, int i_s, int l_s, int k_s, double *Q, double *H){
  void emh2_CPU(GPUCell **cells, int i_s, int l_s, int k_s, double *Q, double *H);
}
#endif

#ifdef __CUDACC__
void emeElement(Cell *c, int3 i, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2){
  void emeElement_GPU(Cell *c, int3 i, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2);
}
#else
void emeElement(Cell *c, int3 i, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2){
  void emeElement_CPU(Cell *c, int3 i, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2);
}
#endif

#ifdef __CUDACC__
void periodicCurrentElement(Cell *c, int i, int k, double *E, int dir, int dirE, int N){
  void periodicCurrentElement_GPU(Cell *c, int i, int k, double *E, int dir, int dirE, int N);
}
#else
void periodicCurrentElement(Cell *c, int i, int k, double *E, int dir, int dirE, int N){
  void periodicCurrentElement_CPU(Cell *c, int i, int k, double *E, int dir, int dirE, int N);

double cuda_atomicAdd(double *address, double val) {
  atomicAdd_GPU(double *address, double val) ;
}
#else
double cuda_atomicAdd(double *address, double val) {
  atomicADD_CPU(double *address, double val) ;
}
#endif


#ifdef __CUDACC__
void GPU_getCellEnergy(GPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez){
  getCellEnergy_GPU(GPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez);
}
#else
void GPU_getCellEnergy(GPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez){
  getCellEnergy_CPU(GPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez);
}
#endif


#ifdef __CUDACC__
void GPU_WriteAllCurrents(GPUCell **cells, int n0, double *jx, double *jy, double *jz, double *rho){
  writeAllCurrents_GPU(GPUCell **cells, int n0, double *jx, double *jy, double *jz, double *rho);
}
#else
void GPU_WriteAllCurrents(GPUCell **cells, int n0, double *jx, double *jy, double *jz, double *rho){
  writeAllCurrents_CPU(GPUCell **cells, int n0, double *jx, double *jy, double *jz, double *rho);
}
#endif


#ifdef __CUDACC__
void GPU_ArrangeFlights(GPUCell **cells, int *d_stage){
  arrangeFlights_GPU(GPUCell **cells, int *d_stage);
}
#else
void GPU_ArrangeFlights(GPUCell **cells, int *d_stage){
  arrangeFlights_CPU(GPUCell **cells, int *d_stage);
}
#endif


#ifdef __CUDACC__
void setCellDoubleToZero(CellDouble *dst, unsigned int n){
  setCellDoubleToZero_GPU(CellDouble *dst, unsigned int n);
}
#else
void setCellDoubleToZero(CellDouble *dst, unsigned int n){
  setCellDoubleToZero_CPU(CellDouble *dst, unsigned int n);
}

#ifdef __CUDACC__
void set_cell_double_arrays_to_zero(CellDouble *, CellDouble *, CellDouble *, int, int, int){
  set_cell_double_arrays_to_zero_GPU(CellDouble *, CellDouble *, CellDouble *, int, int, int);
}
#else
void set_cell_double_arrays_to_zero(CellDouble *, CellDouble *, CellDouble *, int, int, int){
  set_cell_double_arrays_to_zero_CPU(CellDouble *, CellDouble *, CellDouble *, int, int, int);
}
#endif


#ifdef __CUDACC__
void copyFromSharedMemoryToCell(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        Cell *c,
        int index
){
  copyFromSharedMemoryToCell_GPU(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index);
}
#else
void copyFromSharedMemoryToCell(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        Cell *c,
        int index
){
  copyFromSharedMemoryToCell_CPU(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index);
}
#endif

#ifdef __CUDACC__
void emh2_Element(Cell *c, int i, int l, int k, double *Q, double *H){
  emh2_Element_GPU(Cell *c, int i, int l, int k, double *Q, double *H);
}
#else
void emh2_Element(Cell *c, int i, int l, int k, double *Q, double *H){
  emh2_Element_CPU(Cell *c, int i, int l, int k, double *Q, double *H);
}
#endif


#ifdef __CUDACC__
void GPU_emh1(GPUCell **cells, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2){
  emh1_GPU(GPUCell **cells, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2);
}
#else
void GPU_emh1(GPUCell **cells, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2){
  emh1_CPU(GPUCell **cells, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2);
}
#endif

#ifdef __CUDACC__
void periodicCurrentElement(Cell *c, int i, int k, double *E, int dir, int dirE, int N){
  periodicCurrentElement_GPU(Cell *c, int i, int k, double *E, int dir, int dirE, int N);
}
#else
void periodicCurrentElement(Cell *c, int i, int k, double *E, int dir, int dirE, int N){
  periodicCurrentElement_CPU(Cell *c, int i, int k, double *E, int dir, int dirE, int N);
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
