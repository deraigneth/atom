#ifndef KERNELS_H_
#define KERNELS_H_

#include "../include/gpucell.h"
#include "../include/archAPI.h"

__device__ double atomicAdd_GPU(double *, double);

__global__ void getCellEnergy_GPU(GPUCell **, double *, double *, double *, double *);

__global__ void GPU_SetAllCurrentsToZero(GPUCell **);

__global__ void SetFieldsToCells_GPU(GPUCell **, double *, double *, double *, double *, double *, double *);

__global__ void writeAllCurrents_GPU(GPUCell **, int, double *, double *, double *, double *);

__global__ void GPU_WriteControlSystem(Cell **);

__global__ void MakeDepartureLists_GPU(GPUCell **, int *);

__global__ void arrangeFlights_GPU(GPUCell **, int *);

__device__ void writeCurrentComponent(CellDouble *, CurrentTensorComponent *, CurrentTensorComponent *, int);

__device__ void copyCellDouble_GPU(CellDouble *, CellDouble *, unsigned int);

__device__ void setCellDoubleToZero_GPU(CellDouble *, unsigned int);

__device__ void assignSharedWithLocal(CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble **, CellDouble *);

__device__ void copyFieldsToSharedMemory_GPU(CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, Cell *, int, dim3, int);

__device__ void set_cell_double_arrays_to_zero_GPU(CellDouble *, CellDouble *, CellDouble *, int, int, int);

__device__ void MoveParticlesInCell(Cell *, int, int);

__device__ void AccumulateCurrentWithParticlesInCell_GPU(CellDouble *, CellDouble *, CellDouble *, Cell *, int, int);

__device__ void copyFromSharedMemoryToCell_GPU(CellDouble *, CellDouble *, CellDouble *, Cell *, int);

__global__ void GPU_StepAllCells(GPUCell **);

__global__ void CurrentsAllCells_GPU(GPUCell **);

__host__ __device__ void emh2_Element_GPU(Cell *, int, int, int, double *, double *);

__global__ void GPU_emh2(GPUCell **, int, int, int, double *, double *);

__host__ __device__ void emh1_Element_GPU(Cell *, int3, double *, double *, double *, double *, double, double, int3, int3);

__global__ void emh1_GPU(GPUCell **, double *, double *, double *, double *, double, double, int3, int3);

__host__ __device__ void emeElement(Cell *, int3, double *, double *, double *, double *, double, double, double, int3, int3);

__host__ __device__ void periodicElement_GPU(Cell *, int, int, double *, int, int, int);

__global__ void GPU_periodic(GPUCell **, int, int, double *, int, int, int);

__host__ __device__ void periodicCurrentElement_GPU(Cell *, int, int, double *, int, int, int);

__global__ void CurrentPeriodic_GPU(GPUCell **, double *, int, int, int, int, int);

__global__ void GPU_eme(GPUCell **, int3, double *, double *, double *, double *, double, double, double, int3, int3);

#endif /* KERNELS_H_ */
