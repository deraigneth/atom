
double atomicADD(double *address, double val) ;

void SetAllCurrentsToZero_CPU(GPUCell **);

void WriteControlSystem_CPU(Cell **);

void writeCurrentComponent_CPU(CellDouble *, CurrentTensorComponent *, CurrentTensorComponent *, int );

void assignSharedWithLocal_CPU(CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble *);

void MoveParticlesInCell_CPU(Cell *, int , int );

void StepAllCells_CPU(GPUCell **);

void emh2_CPU(GPUCell **, int , int , int , double *, double *);

void emeElement_CPU(Cell *, int3 , double *, double *, double *, double *, double , double , double , int3 , int3 );

void periodicCurrentElement_CPU(Cell *, int, int, double *, int, int, int);
