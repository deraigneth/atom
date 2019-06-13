#ifdef __CUDACC__
double cuda_atomicAdd(double *address, double val) {
  atomicAdd_cuda(double *address, double val) ;
}
#else
double cuda_atomicAdd(double *address, double val) {
  // atomicADD (...)
  // autre fonction CPU dans le fichier kernels_cpu
}
