
double atomicADD_CPU(double *, double ) ;


void SetFieldsToCells_CPU(GPUCell **, double *, double *, double *, double *, double *, double *) ;

void MakeDepartureLists_CPU(GPUCell **, int *) ;

void copyCellDouble_CPU(CellDouble *, CellDouble *, unsigned int) ;

void copyFieldsToSharedMemory_CPU(CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, Cell *, int, dim3, int);

void AccumulateCurrentWithParticlesInCell_CPU(CellDouble *, CellDouble *, CellDouble *, Cell *, int , int ) ;

void CurrentsAllCells_CPU(GPUCell **);

void emh1_Element_CPU(Cell *, int3 , double *, double *, double *, double *, double , double , int3 , int3 ) ;

void periodicElement_CPU(Cell *, int, int, double *, int, int, int) ;

void   CurrentPeriodic_CPU(GPUCell **, double *, int , int , int , int , int );

void getCellEnergy_CPU(GPUCell **, double *, double *, double *, double *);

void writeAllCurrents_CPU(GPUCell **, int , double *, double *, double *, double *);

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

void emh1_CPU(GPUCell **, double *, double *, double *, double *, double , double , int3 , int3 );

void periodicCurrentElement_CPU(Cell *, int , int , double *, int , int , int );
