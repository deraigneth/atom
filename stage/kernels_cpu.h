
double atomicADD_CPU(double *, double ) ;

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
