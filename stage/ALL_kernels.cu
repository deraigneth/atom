

#ifdef __CUDACC__

void GPU_SetAllCurrentsToZero(GPUCell **cells, dim3 dimGrid, dim3 dimBlockExt){
  SetAllCurrentsToZero_GPU( **cells);
}
#else
void GPU_SetAllCurrentsToZero(GPUCell **cells, dim3 dimGrid, dim3 dimBlockExt){
   SetAllCurrentsToZero_CPU( **cells, dimGrid, dimBlockExt);
}
#endif

#ifdef __CUDACC__
void GPU_WriteControlSystem(Cell **cells){
  WriteControlSystem_GPU( **cells);
}
#else
void GPU_WriteControlSystem(Cell **cells){
  WriteControlSystem_CPU( **cells);
}
#endif

#ifdef __CUDACC__
void writeCurrentComponent(CellDouble *J, CurrentTensorComponent *t1, CurrentTensorComponent *t2, int pqr2){
   writeCurrentComponent_GPU( *J,  *t1,  *t2,  pqr2);
}
#else
void writeCurrentComponent(CellDouble *J, CurrentTensorComponent *t1, CurrentTensorComponent *t2, int pqr2){
   writeCurrentComponent_CPU( *J,  *t1,  *t2,  pqr2);
}
#endif


#ifdef __CUDACC__
void assignSharedWithLocal(CellDouble **c_jx, CellDouble **c_jy, CellDouble **c_jz, CellDouble **c_ex, CellDouble **c_ey, CellDouble **c_ez, CellDouble **c_hx, CellDouble **c_hy, CellDouble **c_hz, CellDouble *fd){
           assignSharedWithLocal_GPU(**c_jx, **c_jy, **c_jz, **c_ex, **c_ey, **c_ez, **c_hx, **c_hy, **c_hz, *fd);
        }
#else
void assignSharedWithLocal(CellDouble **c_jx,CellDouble **c_jy,CellDouble **c_jz,CellDouble **c_ex,CellDouble **c_ey,CellDouble **c_ez,CellDouble **c_hx,CellDouble **c_hy,CellDouble **c_hz,CellDouble *fd){
           assignSharedWithLocal_CPU( **c_jx, **c_jy, **c_jz, **c_ex, **c_ey, **c_ez, **c_hx, **c_hy, **c_hz, *fd);
        }
#endif

#ifdef __CUDACC__
void MoveParticlesInCell(Cell *c, int index, int blockDimX){
   MoveParticlesInCell_GPU( *c,  index,  blockDimX);
}
#else
void MoveParticlesInCell(Cell *c, int index, int blockDimX){
   MoveParticlesInCell_CPU( *c,  index,  blockDimX);
}
#endif


#ifdef __CUDACC__
void GPU_StepAllCells(GPUCell **cells){
   StepAllCells_GPU( **cells);
}
#else
void GPU_StepAllCells(GPUCell **cells){
   StepAllCells_CPU( **cells);
}
#endif

#ifdef __CUDACC__
void GPU_emh2(GPUCell **cells, int i_s, int l_s, int k_s, double *Q, double *H){
   emh2_GPU( **cells,  i_s,  l_s,  k_s,  *Q,  *H);
}
#else
void GPU_emh2(GPUCell **cells, int i_s, int l_s, int k_s, double *Q, double *H){
   emh2_CPU( **cells,  i_s,  l_s,  k_s,  *Q,  *H);
}
#endif

#ifdef __CUDACC__
void emeElement(Cell *c, int3 i, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2){
   emeElement_GPU( *c,  i,  *E,  *H1,  *H2,  *J,  c1,  c2,  tau,  d1,  d2);
}
#else
void emeElement(Cell *c, int3 i, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2){
   emeElement_CPU( *c,  i,  *E,  *H1,  *H2,  *J,  c1,  c2,  tau,  d1,  d2);
}
#endif

#ifdef __CUDACC__
void periodicCurrentElement(Cell *c, int i, int k, double *E, int dir, int dirE, int N){
   periodicCurrentElement_GPU( *c,  i,  k,  *E,  dir,  dirE,  N);
}
#else
void periodicCurrentElement(Cell *c, int i, int k, double *E, int dir, int dirE, int N){
   periodicCurrentElement_CPU( *c,  i,  k,  *E,  dir,  dirE,  N);
}
#endif


#ifdef __CUDACC__
double cuda_atomicAdd(double *address, double val) {
  atomicAdd_GPU( *address,  val) ;
}
#else
double cuda_atomicAdd(double *address, double val) {
  atomicADD_CPU( *address,  val) ;
}
#endif


#ifdef __CUDACC__
void GPU_getCellEnergy(GPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez, dim3 dimGrid, dim3 dimBlockOne){
  getCellEnergy_GPU( **cells,  *d_ee,  *d_Ex,  *d_Ey,  *d_Ez);
}
#else
void GPU_getCellEnergy(GPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez, dim3 dimGrid, dim3 dimBlockOne){
  getCellEnergy_CPU( **cells,  *d_ee,  *d_Ex,  *d_Ey,  *d_Ez, dimGrid, dimBlockOne);
}
#endif


#ifdef __CUDACC__
void GPU_WriteAllCurrents(GPUCell **cells, int n0, double *jx, double *jy, double *jz, double *rho){
  writeAllCurrents_GPU( **cells,  n0,  *jx,  *jy,  *jz,  *rho);
}
#else
void GPU_WriteAllCurrents(GPUCell **cells, int n0, double *jx, double *jy, double *jz, double *rho){
  writeAllCurrents_CPU( **cells,  n0,  *jx,  *jy,  *jz,  *rho);
}
#endif


#ifdef __CUDACC__
void GPU_ArrangeFlights(GPUCell **cells, int *d_stage){
  arrangeFlights_GPU( **cells,  *d_stage);
}
#else
void GPU_ArrangeFlights(GPUCell **cells, int *d_stage){
  arrangeFlights_CPU( **cells,  *d_stage);
}
#endif


#ifdef __CUDACC__
void setCellDoubleToZero(CellDouble *dst, unsigned int n){
  setCellDoubleToZero_GPU( *dst,   n);
}
#else
void setCellDoubleToZero(CellDouble *dst, unsigned int n){
  setCellDoubleToZero_CPU( *dst,   n);
}
#endif

#ifdef __CUDACC__
void set_cell_double_arrays_to_zero(CellDouble *m_c_jx, CellDouble *m_c_jy, CellDouble *m_c_jz, int size, int index, int blockDimX){
  set_cell_double_arrays_to_zero_GPU( *m_c_jx,  *m_c_jy,  *m_c_jz, size ,index , blockDimX);
}
#else
void set_cell_double_arrays_to_zero(CellDouble *m_c_jx, CellDouble *m_c_jy, CellDouble *m_c_jz, int size, int index, int blockDimX){
  set_cell_double_arrays_to_zero_CPU( *m_c_jx,  *m_c_jy,  *m_c_jz, size, index , blockDimX);
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
  copyFromSharedMemoryToCell_GPU( *c_jx,  *c_jy,  *c_jz,  *c,  index);
}
#else
void copyFromSharedMemoryToCell(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        Cell *c,
        int index
){
  copyFromSharedMemoryToCell_CPU( *c_jx,  *c_jy,  *c_jz,  *c,  index);
}
#endif

#ifdef __CUDACC__
void emh2_Element(Cell *c, int i, int l, int k, double *Q, double *H){
  emh2_Element_GPU( *c,  i,  l,  k,  *Q,  *H);
}
#else
void emh2_Element(Cell *c, int i, int l, int k, double *Q, double *H){
  emh2_Element_CPU( *c,  i,  l,  k,  *Q,  *H);
}
#endif


#ifdef __CUDACC__
void GPU_emh1(GPUCell **cells, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2){
  emh1_GPU( **cells,  *Q,  *H,  *E1,  *E2,  c1,  c2,  d1,  d2);
}
#else
void GPU_emh1(GPUCell **cells, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2){
  emh1_CPU( **cells,  *Q,  *H,  *E1,  *E2,  c1,  c2,  d1,  d2);
}
#endif


#ifdef __CUDACC__
void GPU_SetFieldsToCells(GPUCell **cells, double *Ex, double *Ey, double *Ez, double *Hx, double *Hy, double *Hz){
  SetFieldsToCells_GPU( **cells,  *Ex,  *Ey,  *Ez,  *Hx,  *Hy,  *Hz) ;
}
#else
void GPU_SetFieldsToCells(GPUCell **cells, double *Ex, double *Ey, double *Ez, double *Hx, double *Hy, double *Hz){
  SetFieldsToCells_CPU( **cells,  *Ex,  *Ey,  *Ez,  *Hx,  *Hy,  *Hz) ;
}
#endif

#ifdef __CUDACC__
void GPU_MakeDepartureLists(GPUCell **cells, int *d_stage, dim3 dimGrid, dim3 dimBlockOne){
  MakeDepartureLists_GPU( **cells,  *d_stage) ;
}
#else
void GPU_MakeDepartureLists(GPUCell **cells, int *d_stage, dim3 dimGrid, dim dimBlockOne){
  MakeDepartureLists_CPU( **cells,  *d_stage, dimGrid, dimBlockOne);
}
#endif

#ifdef __CUDACC__
void copyCellDouble(CellDouble *dst, CellDouble *src, unsigned int n){
  copyCellDouble_GPU( *dst,  *src,   n);
}
#else
void copyCellDouble(CellDouble *dst, CellDouble *src, unsigned int n){
  copyCellDouble_CPU( *dst,  *src,   n) ;
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
           *c_jx,
           *c_jy,
           *c_jz,
           *c_ex,
           *c_ey,
           *c_ez,
           *c_hx,
           *c_hy,
           *c_hz,
           *c,
           index,
           blockId,
           blockDimX
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
           *c_jx,
           *c_jy,
           *c_jz,
           *c_ex,
           *c_ey,
           *c_ez,
           *c_hx,
           *c_hy,
           *c_hz,
           *c,
           index,
           blockId,
           blockDimX
  );
}
#endif

#ifdef __CUDACC__
void AccumulateCurrentWithParticlesInCell(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index, int blockDimX){
  AccumulateCurrentWithParticlesInCell_GPU( *c_jx,  *c_jy,  *c_jz,  *c,  index,  blockDimX);
}
#else
void AccumulateCurrentWithParticlesInCell(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index, int blockDimX){
  AccumulateCurrentWithParticlesInCell_CPU( *c_jx,  *c_jy,  *c_jz,  *c,  index,  blockDimX);
}
#endif

#ifdef __CUDACC__
void GPU_CurrentsAllCells(GPUCell **cells){
   CurrentsAllCells_GPU( **cells);
}
#else
void GPU_CurrentsAllCells(GPUCell **cells){
   CurrentsAllCells_CPU( **cells);
}
#endif

#ifdef __CUDACC__
void emh1_Element(Cell *c, int3 i, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2){
  emh1_Element_GPU( *c,  i,  *Q,  *H,  *E1,  *E2,  c1,  c2,  d1,  d2) ;
}
#else
void emh1_Element(Cell *c, int3 i, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2){
  emh1_Element_CPU( *c,  i,  *Q,  *H,  *E1,  *E2,  c1,  c2,  d1,  d2) ;
}
#endif

#ifdef __CUDACC__
void periodicElement(Cell *c, int i, int k, double *E, int dir, int to, int from){
  periodicElement_GPU( *c,  i,  k,  *E,  dir,  to,  from) ;
}
#else
void periodicElement(Cell *c, int i, int k, double *E, int dir, int to, int from){
  periodicElement_CPU( *c,  i,  k,  *E,  dir,  to,  from);
}
#endif

#ifdef __CUDACC__
void GPU_CurrentPeriodic(GPUCell **cells, double *E, int dirE, int dir, int i_s, int k_s, int N){
  CurrentPeriodic_GPU( **cells,  *E,  dirE,  dir,  i_s,  k_s,  N);
}
#else
void GPU_CurrentPeriodic(GPUCell **cells, double *E, int dirE, int dir, int i_s, int k_s, int N){
  CurrentPeriodic_CPU( **cells,  *E,  dirE,  dir,  i_s,  k_s,  N);
}
#endif

#ifdef __CUDACC__
void  GPU_eme(GPUCell **cells, int3 s, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2){
  eme_GPU( **cells,  s,  *E,  *H1,  *H2,  *J,  c1,  c2,  tau,  d1,  d2);
}
#else
void  GPU_eme(GPUCell **cells, int3 s, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2){
  eme_CPU(**cells,  s,  *E,  *H1,  *H2,  *J,  c1,  c2,  tau,  d1,  d2);
}
#endif

#ifdef __CUDACC__
void GPU_periodic(GPUCell **cells, int i_s, int k_s, double *E, int dir, int to, int from, dim3 dimGrid, dim3 dimBlock){
  periodic_GPU( **cells,  i_s,  k_s,  *E,  dir,  to,from);
}
#else
void GPU_periodic(GPUCell **cells, int i_s, int k_s, double *E, int dir, int to, int from, dim3 dimGrid, dim3 dimBlock){
  periodic_CPU( **cells,  i_s,  k_s,  *E,  dir,  to, from, dimGrid, dimBlock);
}
#endif
