

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
}
#endif
