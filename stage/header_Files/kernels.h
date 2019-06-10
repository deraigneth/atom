
#ifndef KERNELS_H_
#define KERNELS_H_

#include "gpucell.h"
#include "archAPI.h"

double cuda_atomicAdd(double *, double);

void GPU_getCellEnergy(GPUCell **, double *, double *, double *, double *);

void GPU_SetAllCurrentsToZero(GPUCell **);

void GPU_SetFieldsToCells(GPUCell **, double *, double *, double *, double *, double *, double *);

void GPU_WriteAllCurrents(GPUCell **, int, double *, double *, double *, double *);

void GPU_WriteControlSystem(Cell **);

void GPU_MakeDepartureLists(GPUCell **, int *);

void GPU_ArrangeFlights(GPUCell **, int *);

void writeCurrentComponent(CellDouble *, CurrentTensorComponent *, CurrentTensorComponent *, int);

void copyCellDouble(CellDouble *, CellDouble *, unsigned int);

void setCellDoubleToZero(CellDouble *, unsigned int);

void assignSharedWithLocal(CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble *);

void copyFieldsToSharedMemory(CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, Cell *, int, dim3, int);

void set_cell_double_arrays_to_zero(CellDouble *, CellDouble *, CellDouble *, int, int, int);

void MoveParticlesInCell(Cell *, int, int);

void AccumulateCurrentWithParticlesInCell(CellDouble *, CellDouble *, CellDouble *, Cell *, int, int);

void copyFromSharedMemoryToCell(CellDouble *, CellDouble *, CellDouble *, Cell *, int);

void GPU_StepAllCells(GPUCell **);

void GPU_CurrentsAllCells(GPUCell **);

void emh2_Element(Cell *, int, int, int, double *, double *);

void GPU_emh2(GPUCell **, int, int, int, double *, double *);

void emh1_Element(Cell *, int3, double *, double *, double *, double *, double, double, int3, int3);

void GPU_emh1(GPUCell **, double *, double *, double *, double *, double, double, int3, int3);

void emeElement(Cell *, int3, double *, double *, double *, double *, double, double, double, int3, int3);

void periodicElement(Cell *, int, int, double *, int, int, int);

void GPU_periodic(GPUCell **, int, int, double *, int, int, int);

void periodicCurrentElement(Cell *, int, int, double *, int, int, int);

void GPU_CurrentPeriodic(GPUCell **, double *, int, int, int, int, int);

void GPU_eme(GPUCell **, int3, double *, double *, double *, double *, double, double, double, int3, int3);

#endif /* KERNELS_H_ */
