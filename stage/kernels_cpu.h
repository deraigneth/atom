

void SetAllCurrentsToZero_CPU(GPUCell **, dim3, dim3);

void WriteControlSystem_CPU(Cell **, dim3, dim3);

void writeCurrentComponent_CPU(CellDouble *, CurrentTensorComponent *, CurrentTensorComponent *, int );

void assignSharedWithLocal_CPU(CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble **,CellDouble *);

void MoveParticlesInCell_CPU(Cell *, int , int );

void StepAllCells_CPU(GPUCell **, dim3, dim3);

void emh2_CPU(GPUCell **, int , int , int , double *, double *, dim3 , dim3);

void emeElement_CPU(Cell *, int3 , double *, double *, double *, double *, double , double , double , int3 , int3 );

void periodicCurrentElement_CPU(Cell *, int, int, double *, int, int, int);

double atomicADD_CPU(double *, double ) ;



void SetFieldsToCells_CPU(GPUCell **, double *, double *, double *, double *, double *, double *, dim3, dim3) ;

void MakeDepartureLists_CPU(GPUCell **, int *, dim3, dim3) ;

void copyCellDouble_CPU(CellDouble *, CellDouble *, unsigned int) ;

void copyFieldsToSharedMemory_CPU(CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, Cell *, int, dim3, int);

void AccumulateCurrentWithParticlesInCell_CPU(CellDouble *, CellDouble *, CellDouble *, Cell *, int , int ) ;

void CurrentsAllCells_CPU(GPUCell **, dim3, dim3);

void emh1_Element_CPU(Cell *, int3 , double *, double *, double *, double *, double , double , int3 , int3 ) ;

void periodicElement_CPU(Cell *, int, int, double *, int, int, int) ;

void   CurrentPeriodic_CPU(GPUCell **, double *, int , int , int , int , int , dim3 , dim3);

void getCellEnergy_CPU(GPUCell **, double *, double *, double *, double *, dim3, dim3);

void writeAllCurrents_CPU(GPUCell **, int , double *, double *, double *, double *,dim3, dim3);

void setCellDoubleToZero_CPU(CellDouble *, unsigned int );

void set_cell_double_arrays_to_zero_CPU(CellDouble *, CellDouble *, CellDouble *, int, int, int);

void copyFromSharedMemoryToCell_CPU(
        CellDouble *,
        CellDouble *,
        CellDouble *,
        Cell *,
        int
)

void emh2_Element_CPU(Cell *, int , int , int , double *, double *);

void emh1_CPU(GPUCell **, double *, double *, double *, double *, double , double , int3 , int3 , dim3 , dim3);


void periodic_CPU(GPUCell**, int , int , double *, int , int , int, dim3, dim3 );


void arrangeFlights_GPU(GPUCell **, int *, dim3, dim3);

void  eme_CPU(GPUCell **, int3 , double *, double *, double *, double *, double , double , double , int3 , int3 , dim3 , dim3 );
