#ifndef KERNELS_H_
#define KERNELS_H_

#include "../include/gpucell.h"
#include "../include/archAPI.h"



__global__ void GPU_getCellEnergy(GPUCell **, double *, double *, double *, double *);

__global__ void SetAllCurrentsToZero_GPU(GPUCell **);

__global__ void GPU_SetFieldsToCells(GPUCell **, double *, double *, double *, double *, double *, double *);

__global__ void GPU_WriteAllCurrents(GPUCell **, int, double *, double *, double *, double *);

__global__ void WriteControlSystem_GPU(Cell **);

__global__ void GPU_MakeDepartureLists(GPUCell **, int *);

__global__ void GPU_ArrangeFlights(GPUCell **, int *);

__device__ void writeCurrentComponent_GPU(CellDouble *, CurrentTensorComponent *, CurrentTensorComponent *, int);

__device__ void copyCellDouble(CellDouble *, CellDouble *, unsigned int);

__device__ void setCellDoubleToZero(CellDouble *, unsigned int);

__device__ void assignSharedWithLocal_GPU(CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble *);

__device__ void copyFieldsToSharedMemory(CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, Cell *, int, dim3, int);

__device__ void set_cell_double_arrays_to_zero(CellDouble *, CellDouble *, CellDouble *, int, int, int);

__device__ void MoveParticlesInCell_GPU(Cell *, int, int);

__device__ void AccumulateCurrentWithParticlesInCell(CellDouble *, CellDouble *, CellDouble *, Cell *, int, int);

__device__ void copyFromSharedMemoryToCell(CellDouble *, CellDouble *, CellDouble *, Cell *, int);

__global__ void StepAllCells_GPU(GPUCell **);

__global__ void GPU_CurrentsAllCells(GPUCell **);

__host__ __device__ void emh2_Element(Cell *, int, int, int, double *, double *);

__global__ void emh2_GPU(GPUCell **, int, int, int, double *, double *);

__host__ __device__ void emh1_Element(Cell *, int3, double *, double *, double *, double *, double, double, int3, int3);

__global__ void GPU_emh1(GPUCell **, double *, double *, double *, double *, double, double, int3, int3);

__host__ __device__ void emeElement_GPU(Cell *, int3, double *, double *, double *, double *, double, double, double, int3, int3);

__host__ __device__ void periodicElement(Cell *, int, int, double *, int, int, int);

__global__ void GPU_periodic(GPUCell **, int, int, double *, int, int, int);

__host__ __device__ void periodicCurrentElement_GPU(Cell *, int, int, double *, int, int, int);

__global__ void GPU_CurrentPeriodic(GPUCell **, double *, int, int, int, int, int);

__global__ void GPU_eme(GPUCell **, int3, double *, double *, double *, double *, double, double, double, int3, int3);

#endif /* KERNELS_H_ */
