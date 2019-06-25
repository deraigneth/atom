#include "kernel_cpu.h"

double atomicADD_CPU(double *address, double val) {
  double assumed, old = *address;
  do {
      assumed = old;
      old = __longlong_as_double(atomicCAS((unsigned long long int *)address, __double_as_longlong(assumed), __double_as_longlong(val + assumed)));
  } while (assumed != old);
  return old;
}


void SetAllCurrentsToZero_CPU(GPUCell **cells, dim3 dimGrid, dim3 dimBlockExt) {

}

void WriteControlSystem_CPU(Cell **cells) {

}

void writeCurrentComponent_CPU(CellDouble *J, CurrentTensorComponent *t1, CurrentTensorComponent *t2, int pqr2) {

}

void assignSharedWithLocal_CPU(
        CellDouble **c_jx,
        CellDouble **c_jy,
        CellDouble **c_jz,
        CellDouble **c_ex,
        CellDouble **c_ey,
        CellDouble **c_ez,
        CellDouble **c_hx,
        CellDouble **c_hy,
        CellDouble **c_hz,
        CellDouble *fd) {

}

void MoveParticlesInCell_CPU(Cell *c, int index, int blockDimX) {

}

void StepAllCells_CPU(GPUCell **cells, dim3 dimGrid, dim3 dimBlock) {
  Cell *c, *c0 = cells[0];
  //à regarder ce que signifie shared
  CellDouble fd[9];
  CellDouble *c_jx, *c_jy, *c_jz, *c_ex, *c_ey, *c_ez, *c_hx, *c_hy, *c_hz;
  Particle p;
  for (int i=0; i<dimGrid.x; i++){
    for (int j=0; j<dimGrid.y; j++){
      for (int k=0; k<dimGrid.z; k++){
        c = cells[c0->getGlobalCellNumber(i, j, k)];
        dim3 val(i,j,k);
      }
    }
  }
  //utilise :
  //assignSharedWithLocal_CPU(&c_jx, &c_jy, &c_jz, &c_ex, &c_ey, &c_ez, &c_hx, &c_hy, &c_hz, fd);
  for (int l=0; l<dimGrid.x; l++){
    for (int m=0; m<dimBlock.x; m++){
      // penser à revoir les index et BlockIdx
      //copyFieldsToSharedMemory_CPU(c_jx, c_jy, c_jz, c_ex, c_ey, c_ez, c_hx, c_hy, c_hz, c,m, val, dimGrid.x);


      //MoveParticlesInCell_CPU(c, m, dimGrid.x);

      //copyFromSharedMemoryToCell_CPU(c_jx, c_jy, c_jz, c, m);
    }
  }
}


void emh2_CPU(GPUCell **cells, int i_s, int l_s, int k_s, double *Q, double *H, dim3 dimGrid, dim3 dimBlock) {
  Cell *c0 = cells[0] ;
  for (int i=0 ; i< dimGrid.x; i++){
    for (int j=0 ; j < dimGrid.y ; j++){
      for (int z=0 ; z < dimGrid.z ; z++){
              emh2_Element_CPU(c0, i_s + i, l_s + j, k_s + z, Q, H);
      }
    }
  }
}


void emeElement_CPU(Cell *c, int3 i, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2) {

}

void periodicCurrentElement_CPU(Cell *c, int i, int k, double *E, int dir, int dirE, int N) {


}
void SetFieldsToCells_CPU(GPUCell **cells, double *Ex, double *Ey, double *Ez, double *Hx, double *Hy, double *Hz, dim3 dimGrid, dim3 dimBlockExt){
  Cell *c, *c0 = cells[0] ;
  for (int i=0 ; i< dimGrid.x; i++){
    for (int j=0 ; j < dimGrid.y ; j++){
      for (int z=0 ; z < dimGrid.z ; z++){
              c = cells[c0->getGlobalCellNumber(i,j,z)] ;
      }
    }
  }
  for (int i=0 ; i< dimGrid.x; i++){
    for (int j=0 ; j < dimGrid.y ; j++){
      for (int z=0 ; z < dimGrid.z ; z++){
        for (int g=0 ; g < dimBlockExt.x ; g++){
          for (int h=0 ; h < dimBlockExt.y ; h++){
            for (int k=0 ; k < dimBlockExt.z ; k++){
              uint3 val(g,h,k) ;
              c->readFieldsFromArrays(Ex,Ey,Ez,Hx,Hy,Hz,val) ;
            }
          }
        }
      }
    }
  }
}


void MakeDepartureLists_CPU(GPUCell **cells, int *d_stage, dim3 dimGrid, dim3 dimBlockOne){




}

void copyCellDouble_CPU(CellDouble *dst, CellDouble *src, unsigned int n) {

}
void copyFieldsToSharedMemory_CPU(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        CellDouble *c_ex,
        CellDouble *c_ey,
        CellDouble *c_ez,
        CellDouble *c_hx,
        CellDouble *c_hy,
        CellDouble *c_hz,
        Cell *c,
        int index,
        dim3 blockId,
        int blockDimX
){

}
void AccumulateCurrentWithParticlesInCell_CPU(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index, int blockDimX){

}

void CurrentsAllCells_CPU(GPUCell **cells, dim3 dimGrid, dim3 dimBlock){
  Cell *c, *c0 = cells[0];
  //à regarder les shared
  CellDouble fd[9];
  CellDouble *c_jx, *c_jy, *c_jz, *c_ex, *c_ey, *c_ez, *c_hx, *c_hy, *c_hz;
  CellDouble m_c_jx[CURRENT_SUM_BUFFER_LENGTH];
  CellDouble m_c_jy[CURRENT_SUM_BUFFER_LENGTH];
  CellDouble m_c_jz[CURRENT_SUM_BUFFER_LENGTH];

  for (int i=0; i<dimGrid.x; i++){
    for (int j=0; j<dimGrid.y; j++){
      for (int k=0; k<dimGrid.z; k++){
        c = cells[c0->getGlobalCellNumber(i, j, k)];
        dim3 val(i,j,k);
      }
    }
  }
  //assignSharedWithLocal_CPU(&c_jx, &c_jy, &c_jz, &c_ex, &c_ey, &c_ez, &c_hx, &c_hy, &c_hz, fd);

  for (int l=0; l<dimGrid.x; l++){
    for (int m=0; m<dimBlock.x; m++){
      // penser à revoir les index et BlockIdx
      //copyFieldsToSharedMemory_CPU(c_jx, c_jy, c_jz, c_ex, c_ey, c_ez, c_hx, c_hy, c_hz, c, m, val, dimGrid.x);

      //set_cell_double_arrays_to_zero_CPU(m_c_jx, m_c_jy, m_c_jz, CURRENT_SUM_BUFFER_LENGTH, m, dimGrid.x);

      //AccumulateCurrentWithParticlesInCell_CPU(c_jx, c_jy, c_jz, c, m, dimGrid.x);

      //copyFromSharedMemoryToCell_CPU(c_jx, c_jy, c_jz, c, m);
    }
  }
}

void emh1_Element_CPU(Cell *c, int3 i, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2) {

}

void periodicElement_CPU(Cell *c, int i, int k, double *E, int dir, int to, int from){

}
 void CurrentPeriodic_CPU(GPUCell **cells, double *E, int dirE, int dir, int i_s, int k_s, int N, dim3 dimGrid, dim3 dimBlock){
   for (int i=0; i<dimGrid.x; i++){
     for (int j=0; j<dimGrid.z; j++){
       Cell *c0 = cells[0];
       //utilise la fonction periodicCurrentElement_CPU
       //periodicCurrentElement_CPU(c0, i + i_s, j + k_s, E, dir, dirE, N);
     }
   }
 }

void getCellEnergy_CPU(GPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez, dim3 dimGrid, dim3 dimBlockOne){
  Cell *c0 = cells[0], nc;
  double t, ex, ey, ez;

  for (i=0; i<dimGrid; i++){
    for (j=0; j<dimGrid; j++){
      for (h=0; h<dimGrid; h++){
        uint3 val(i,j,h);
        int n = c0-> getGlobalCellNumber(val.x, val.y, val.z);
        ex = d_Ex[n];
        ey = d_Ey[n];
        ez = d_Ez[n];

        t = ex * ex + ey * ey + ez * ez;

        atomicADD_CPU(d_ee, t);

      }
    }
  }
}

void writeAllCurrents_CPU(GPUCell **cells, int n0, double *jx, double *jy, double *jz, double *rho,dim3 dimGrid, dim3 dimBlock){
  Cell *c, *c0 = cells[0];
  for (int i=0; i<dimGrid.x;i++){
    for (int j=0; j<dimGrid.y;j++){
      for (int k=0; k<dimGrid.z;k++){
        c = cells[c0->getGlobalCellNumber(i,j,k)] ;
      }
    }
  }
  for (int i=0 ; i< dimGrid.x; i++){
    for (int j=0 ; j < dimGrid.y ; j++){
      for (int z=0 ; z < dimGrid.z ; z++){
        for (int g=0 ; g < dimBlockExt.x ; g++){
          for (int h=0 ; h < dimBlockExt.y ; h++){
            for (int p=0 ; p < dimBlockExt.z ; p++){
              int n = c->getFortranCellNumber(c->i + g - 1, c->l + h -1 , c->k + p -1);
              if (n < 0) n = -n;
              double t, t_x, t_y;
              t_x = c->Jx->M[g][h][p];
              int3 i3 = c->getCellTripletNumber(n);

              atomicADD_CPU(&(jx[n]), t_x);
              t_y = c->Jy->M[g][h][p];
              atomicADD_CPU(&(jy[n]), t_y);
              t = c->Jz->M[g][h][p];
              atomicADD_CPU(&(jz[n]), t);
            }
          }
        }
      }
    }
  }

}

void setCellDoubleToZero_CPU(CellDouble *dst, unsigned int n){
  if (n < CellExtent * CellExtent * CellExtent) {
      double *d_dst;

      d_dst = (double *) (dst->M);
      d_dst[n] = 0.0;
  }//a regarder
}

void set_cell_double_arrays_to_zero_CPU(CellDouble *, CellDouble *, CellDouble *, int, int, int){
  //à completer

}

void copyFromSharedMemoryToCell_CPU(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        Cell *c,
        int index
){
  //à completer
}

void emh2_Element_CPU(Cell *c, int i, int l, int k, double *Q, double *H){
  // à compléter
}

void emh1_CPU(GPUCell **cells, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2, dim3 dimGrid, dim3 dimBlock){
  for (int i=0; i< dimGrid.x; i++ ){
    for (int j=0; j< dimGrid.y; j++ ){
      for (int k=0; k< dimGrid.z; k++ ){
        int3 i3 = make_int3(i, j, k);
        Cell *c0 = cells[0];
        //utilise la fonction:
        //emh1_Element_CPU(c0, i3, Q, H, E1, E2, c1, c2, d1, d2);
      }
    }
  }
}



void periodic_CPU(GPUCell **cells, int i_s, int k_s, double *E, int dir, int to, int from, dim3 dimGrid, dim3 dimBlock) {

}

void eme_CPU(GPUCell **cells, int3 s, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2, dim3 dimGrid, dim3 dimBlock){
  Cell *c0 = cells[0];
  // for nx
  for (int i=0; i<dimGrid.x;i++){
    for (int j = 0 ; j < dimBlock.x ; j++){
      s.x += j ;
    }
    s.x += i ;
  }
  // ou alors s.x = i * dimGrid.x + j ;

  // for ny
  for (int i=0; i<dimGrid.y;i++){
    for (int j = 0 ; j < dimBlock.y ; j++){
      s.y += j ;
    }
    s.y += i ;
  }

  // for nz
  for (int i=0; i<dimGrid.z;i++){
    for (int j = 0 ; j < dimBlock.z ; j++){
      s.z += j ;
    }
    s.z += i ;
  }

  // à changer car fonction GPU
  // emeElement_GPU(c0, s, E, H1, H2, J, c1, c2, tau, d1, d2);


}

void arrangeFlights_CPU(GPUCell **cells, int *d_stage, dim3 dimGridBulk, dim3 dimBlockOne){
  for (int nx=0; nx<dimGridBulk.x; nx++){
    for (int ny=0; ny<dimGridBulk.y; ny++){
      for (int nz=0; nz<dimGridBulk.z; nz++){
        int ix, iy, iz, snd_ix, snd_iy, snd_iz, num, n;
        Particle p;

        Cell *c, *c0 = cells[0], nc, *snd_c;

        c = cells[n = c0->getGlobalCellNumber(nx, ny, nz)];

        for (ix = 0; ix < 3; ix++){
            for (iy = 0; iy < 3; iy++){
                for (iz = 0; iz < 3; iz++) {
                    int index = ix * 9 + iy * 3 + iz;
                    n = c0->getWrapCellNumber(nx + ix - 1, ny + iy - 1, nz + iz - 1);

                    snd_c = cells[n];
                    if (nx == 24 && ny == 2 && nz == 2) {
                        d_stage[index * 4] = snd_c->i;
                        d_stage[index * 4 + 1] = snd_c->l;
                        d_stage[index * 4 + 2] = snd_c->k;
                        d_stage[index * 4 + 3] = snd_c->departureListLength;
                    }

                    snd_ix = ix;
                    snd_iy = iy;
                    snd_iz = iz;
                    c->inverseDirection(&snd_ix, &snd_iy, &snd_iz);

                    num = snd_c->departure[snd_ix][snd_iy][snd_iz];

                    for (int i = 0; i < num; i++) {
                        p = snd_c->departureList[snd_ix][snd_iy][snd_iz][i];
                        if (nx == 24 && ny == 2 && nz == 2) {}
                        c->Insert(p);
                    }
                }
           }
        }
      }
    }
  }
}
