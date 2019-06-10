
#ifndef KERNELS_H_
#define KERNELS_H_

#include "CPUcell.h"
#include "archAPI.h"

double cuda_atomicAdd(double *, double);

void CPU_getCellEnergy(CPUCell **, double *, double *, double *, double *);

void CPU_SetAllCurrentsToZero(CPUCell **);

void CPU_SetFieldsToCells(CPUCell **, double *, double *, double *, double *, double *, double *);

void CPU_WriteAllCurrents(CPUCell **, int, double *, double *, double *, double *);

void CPU_WriteControlSystem(Cell **);

void CPU_MakeDepartureLists(CPUCell **, int *);

void CPU_ArrangeFlights(CPUCell **, int *);

void writeCurrentComponent(CellDouble *, CurrentTensorComponent *, CurrentTensorComponent *, int);

void copyCellDouble(CellDouble *, CellDouble *, unsigned int);

void setCellDoubleToZero(CellDouble *, unsigned int);

void assignSharedWithLocal(CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble *);

void copyFieldsToSharedMemory(CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, Cell *, int, dim3, int);

void set_cell_double_arrays_to_zero(CellDouble *, CellDouble *, CellDouble *, int, int, int);

void MoveParticlesInCell(Cell *, int, int);

void AccumulateCurrentWithParticlesInCell(CellDouble *, CellDouble *, CellDouble *, Cell *, int, int);

void copyFromSharedMemoryToCell(CellDouble *, CellDouble *, CellDouble *, Cell *, int);

void CPU_StepAllCells(CPUCell **);

void CPU_CurrentsAllCells(CPUCell **);

void emh2_Element(Cell *, int, int, int, double *, double *);

void CPU_emh2(CPUCell **, int, int, int, double *, double *);

void emh1_Element(Cell *, int3, double *, double *, double *, double *, double, double, int3, int3);

void CPU_emh1(CPUCell **, double *, double *, double *, double *, double, double, int3, int3);

void emeElement(Cell *, int3, double *, double *, double *, double *, double, double, double, int3, int3);

void periodicElement(Cell *, int, int, double *, int, int, int);

void CPU_periodic(CPUCell **, int, int, double *, int, int, int);

void periodicCurrentElement(Cell *, int, int, double *, int, int, int);

void CPU_CurrentPeriodic(CPUCell **, double *, int, int, int, int, int);

void CPU_eme(CPUCell **, int3, double *, double *, double *, double *, double, double, double, int3, int3);

#endif /* KERNELS_H_ */
