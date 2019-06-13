


void GPU_getCellEnergy(GPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez);

void GPU_SetAllCurrentsToZero(GPUCell **);

void GPU_WriteControlSystem(Cell **);

void writeCurrentComponent(CellDouble *, CurrentTensorComponent *, CurrentTensorComponent *, int );

void assignSharedWithLocal(CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble *);

void MoveParticlesInCell(Cell *, int , int );

void GPU_StepAllCells(GPUCell **);

void GPU_emh2(GPUCell **, int , int , int , double *, double *);

void emeElement(Cell *, int3, double *, double *, double *, double *, double, double, double, int3, int3);

void periodicCurrentElement(Cell *, int, int, double *, int, int, int);
