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
  for (int i=0;i<dimGrid.x;i++){
    for (int j=0;j<dimGrid.y;j++){
      for(int k=0;k<dimGrid.z;k++){
        Cell *c, *c0 = cells[0], nc;
        c = cells[c0->getGlobalCellNumber(nx, ny, nz)];
        nc = *c;
        for (int g=0 ; g < dimBlockExt.x ; g++){
          for (int h=0 ; h < dimBlockExt.y ; h++){
            for (int z=0 ; z < dimBlockExt.z ; z++){
              uint3 thread1(g,h,z);
              nc.SetAllCurrentsToZero(thread1);
            }
          }
        }
      }
    }
  }
}

void WriteControlSystem_CPU(Cell **cells, dim3 dimGrid, dim3 dimBlock) {
  for (int i=0;i<dimGrid.x;i++){
    for(int j=0;j<dimGrid.y;j++){
      for(int k=0;k<dimGrid.z;k++){
        Cell *c, *c0 = cells[0], nc;
        c = cells[c0->getGlobalCellNumber(i,j,k)];
        nc = *c;
        nc.SetControlSystemToParticles();
      }
    }
  }
}

void writeCurrentComponent_CPU(CellDouble *J, CurrentTensorComponent *t1, CurrentTensorComponent *t2, int pqr2) {
  atomicADD_CPU(&(J->M[t1->i11][t1->i12][t1->i13]), t1->t[0]);
  atomicADD_CPU(&(J->M[t1->i21][t1->i22][t1->i23]), t1->t[1]);
  atomicADD_CPU(&(J->M[t1->i31][t1->i32][t1->i33]), t1->t[2]);
  atomicADD_CPU(&(J->M[t1->i41][t1->i42][t1->i43]), t1->t[3]);

  if (pqr2 == 2) {
      atomicADD_CPU(&(J->M[t2->i11][t2->i12][t2->i13]), t2->t[0]);
      atomicADD_CPU(&(J->M[t2->i21][t2->i22][t2->i23]), t2->t[1]);
      atomicADD_CPU(&(J->M[t2->i31][t2->i32][t2->i33]), t2->t[2]);
      atomicADD_CPU(&(J->M[t2->i41][t2->i42][t2->i43]), t2->t[3]);
  }
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
          *c_ex = &(fd[0]);
          *c_ey = &(fd[1]);
          *c_ez = &(fd[2]);

          *c_hx = &(fd[3]);
          *c_hy = &(fd[4]);
          *c_hz = &(fd[5]);

          *c_jx = &(fd[6]);
          *c_jy = &(fd[7]);
          *c_jz = &(fd[8]);

}

void MoveParticlesInCell_CPU(Cell *c, int index, int blockDimX) {
  CellTotalField cf;

  while (index < c->number_of_particles) {
      cf.Ex = c->Ex;
      cf.Ey = c->Ey;
      cf.Ez = c->Ez;
      cf.Hx = c->Hx;
      cf.Hy = c->Hy;
      cf.Hz = c->Hz;

      c->MoveSingleParticle(index, cf);

      index += blockDimX;
  }
}

void StepAllCells_CPU(GPUCell **cells, dim3 dimGrid, dim3 dimBlock) {
  Cell *c, *c0 = cells[0];
  CellDouble fd[9];
  CellDouble *c_jx, *c_jy, *c_jz, *c_ex, *c_ey, *c_ez, *c_hx, *c_hy, *c_hz;
  Particle p;
  assignSharedWithLocal_CPU(&c_jx, &c_jy, &c_jz, &c_ex, &c_ey, &c_ez, &c_hx, &c_hy, &c_hz, fd);
  for (int i=0; i<dimGrid.x; i++){
    for (int a=0;a<dimGrid.y;a++){
      for(int b=0;b<dimGrid.z;b++){
        for (int j=0; j<dimBlock.x; j++){
          c = cells[c0->getGlobalCellNumber(i, a, b)];
          dim3 block(i,a,b);
          copyFieldsToSharedMemory_CPU(c_jx, c_jy, c_jz, c_ex, c_ey, c_ez, c_hx, c_hy, c_hz, c,j, block,i);
          MoveParticlesInCell_CPU(c, j, i);
          copyFromSharedMemoryToCell_CPU(c_jx, c_jy, c_jz, c,j,dimBlock);
        }
      }
    }
  }
}


void emh2_CPU(GPUCell **cells, int i_s, int l_s, int k_s, double *Q, double *H, dim3 dimGrid, dim3 dimBlock) {
  for (int i=0 ; i< dimGrid.x; i++){
    for (int j=0 ; j < dimGrid.y ; j++){
      for (int z=0 ; z < dimGrid.z ; z++){
              Cell *c0 = cells[0] ;
              emh2_Element_CPU(c0, i_s + i, l_s + j, k_s + z, Q, H);
      }
    }
  }
}


void emeElement_CPU(Cell *c, int3 i, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2) {
  int n = c->getGlobalCellNumber(i.x, i.y, i.z);
  int n1 = c->getGlobalCellNumber(i.x + d1.x, i.y + d1.y, i.z + d1.z);
  int n2 = c->getGlobalCellNumber(i.x + d2.x, i.y + d2.y, i.z + d2.z);

  E[n] += c1 * (H1[n] - H1[n1]) - c2 * (H2[n] - H2[n2]) - tau * J[n];
}

void periodicCurrentElement_CPU(Cell *c, int i, int k, double *E, int dir, int dirE, int N) {
  int n1 = c->getGlobalBoundaryCellNumber(i, k, dir, 1);
  int n_Nm1 = c->getGlobalBoundaryCellNumber(i, k, dir, N - 1);

  if (dir != dirE) {
      E[n1] += E[n_Nm1];
  }

  if (dir != 1 || dirE != 1) {
      E[n_Nm1] = E[n1];
  }

  int n_Nm2 = c->getGlobalBoundaryCellNumber(i, k, dir, N - 2);
  int n0 = c->getGlobalBoundaryCellNumber(i, k, dir, 0);

  E[n0] += E[n_Nm2];
  E[n_Nm2] = E[n0];
}


void SetFieldsToCells_CPU(GPUCell **cells, double *Ex, double *Ey, double *Ez, double *Hx, double *Hy, double *Hz, dim3 dimGrid, dim3 dimBlockExt){
  for (int i=0 ; i< dimGrid.x; i++){
    for (int j=0 ; j < dimGrid.y ; j++){
      for (int z=0 ; z < dimGrid.z ; z++){
              Cell *c, *c0 = cells[0] ;
              c = cells[c0->getGlobalCellNumber(i,j,z)] ;
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
  for (int i=;i<dimGrid.x;i++){
    for (int j=;j<dimGrid.y;j++){
      for (int k=;k<dimGrid.z;k++){
        int ix, iy, iz ;
        Particle p;
        Cell *c, *c0 = cells[0], nc;
        c = cells[c0->getGlobalCellNumber(i, j, k)];
        c->departureListLength = 0;
        for (ix = 0; ix < 3; ix++) {
            for (iy = 0; iy < 3; iy++) {
                for (iz = 0; iz < 3; iz++) {
                    c->departure[ix][iy][iz] = 0;
                }
            }
        }
        c->departureListLength = 0;
        for (int num = 0; num < c->number_of_particles; num++) {
            p = c->readParticleFromSurfaceDevice(num);

            if (!c->isPointInCell(p.GetX())) { //check Paricle = operator !!!!!!!!!!!!!!!!!!!!!!!!!!!
                c->removeParticleFromSurfaceDevice(num, &p, &(c->number_of_particles));
                c->flyDirection(&p, &ix, &iy, &iz);
                if (p.fortran_number == 325041 && p.sort == 2) {
                    d_stage[0] = ix;
                    d_stage[1] = iy;
                    d_stage[2] = iz;
                }


                if (c->departureListLength == PARTICLES_FLYING_ONE_DIRECTION) {
                    d_stage[0] = TOO_MANY_PARTICLES;
                    d_stage[1] = c->i;
                    d_stage[2] = c->l;
                    d_stage[3] = c->k;
                    d_stage[1] = ix;
                    d_stage[2] = iy;
                    d_stage[3] = iz;
                    return;
                }
                c->departureListLength++;
                int num1 = c->departure[ix][iy][iz];

                c->departureList[ix][iy][iz][num1] = p;
                if (p.fortran_number == 325041 && p.sort == 2) {
                    d_stage[4] = num1;
                    d_stage[5] = c->departureList[ix][iy][iz][num1].fortran_number;

                }

                c->departure[ix][iy][iz] += 1;
                num--;
            }
        }
      }
    }
  }
}

void copyCellDouble_CPU(CellDouble *dst, CellDouble *src, unsigned int n) {
  if (n < CellExtent * CellExtent * CellExtent) {
      double *d_dst, *d_src;//,t;

      d_dst = (double *) (dst->M);
      d_src = (double *) (src->M);

      d_dst[n] = d_src[n];
  }
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
  while (index < CellExtent * CellExtent * CellExtent) {
      copyCellDouble_CPU(c->Jx, c_jx, index);
      copyCellDouble_CPU(c->Jy, c_jy, index);
      copyCellDouble_CPU(c->Jz, c_jz, index);
      for (int i=0;i<blockId.x;i++){
        index += i ;
      }
    }
  c->busyParticleArray = 0;
}


void AccumulateCurrentWithParticlesInCell_CPU(CellDouble *c_jx, CellDouble *c_jy, CellDouble *c_jz, Cell *c, int index, int blockDimX){
  DoubleCurrentTensor dt;
  int pqr2;

  while (index < c->number_of_particles) {
      c->AccumulateCurrentSingleParticle(index, &pqr2, &dt);

      writeCurrentComponent_CPU(c_jx, &(dt.t1.Jx), &(dt.t2.Jx), pqr2);

      writeCurrentComponent_CPU(c_jy, &(dt.t1.Jy), &(dt.t2.Jy), pqr2);
      writeCurrentComponent_CPU(c_jz, &(dt.t1.Jz), &(dt.t2.Jz), pqr2);

      index += blockDimX;
  }
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
        for (int z=0; z<dimBlock.x;z++){
          c = cells[c0->getGlobalCellNumber(i,j,k)];
          assignSharedWithLocal_CPU(&c_jx, &c_jy, &c_jz, &c_ex, &c_ey, &c_ez, &c_hx, &c_hy, &c_hz, fd);
          dim3 block(x,y,z);
          copyFieldsToSharedMemory_CPU(c_jx, c_jy, c_jz, c_ex, c_ey, c_ez, c_hx, c_hy, c_hz, c,z, block,i);
          set_cell_double_arrays_to_zero_CPU(m_c_jx, m_c_jy, m_c_jz, CURRENT_SUM_BUFFER_LENGTH,z ,i );
          AccumulateCurrentWithParticlesInCell_CPU(c_jx, c_jy, c_jz, c, z,i);
          copyFromSharedMemoryToCell_CPU(c_jx, c_jy, c_jz, c,z,dimBlock);
        }
      }
    }
  }
}

void emh1_Element_CPU(Cell *c, int3 i, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2) {
  int n = c->getGlobalCellNumber(i.x, i.y, i.z);
  int n1 = c->getGlobalCellNumber(i.x + d1.x, i.y + d1.y, i.z + d1.z);
  int n2 = c->getGlobalCellNumber(i.x + d2.x, i.y + d2.y, i.z + d2.z);

  double e1_n1 = E1[n1];
  double e1_n = E1[n];
  double e2_n2 = E2[n2];
  double e2_n = E2[n];

  double t = 0.5 * (c1 * (e1_n1 - e1_n) - c2 * (e2_n2 - e2_n));
  Q[n] = t;
  H[n] += Q[n];
}

void periodicElement_CPU(Cell *c, int i, int k, double *E, int dir, int to, int from){
  int n = c->getGlobalBoundaryCellNumber(i, k, dir, to);
  int n1 = c->getGlobalBoundaryCellNumber(i, k, dir, from);
  E[n] = E[n1];
}


 void CurrentPeriodic_CPU(GPUCell **cells, double *E, int dirE, int dir, int i_s, int k_s, int N, dim3 dimGrid, dim3 dimBlock){
   for (int i=0; i<dimGrid.x; i++){
     for (int j=0; j<dimGrid.z; j++){
       Cell *c0 = cells[0];
       periodicCurrentElement_CPU(c0, i + i_s, j + k_s, E, dir, dirE, N);
     }
   }
 }

void getCellEnergy_CPU(GPUCell **cells, double *d_ee, double *d_Ex, double *d_Ey, double *d_Ez, dim3 dimGrid, dim3 dimBlockOne){
  double t, ex, ey, ez;
  for (int i=0; i<dimGrid; i++){
    for (int j=0; j<dimGrid; j++){
      for (int h=0; h<dimGrid; h++){
        Cell *c0 = cells[0], nc;
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
  for (int i=0; i<dimGrid.x;i++){
    for (int j=0; j<dimGrid.y;j++){
      for (int k=0; k<dimGrid.z;k++){
        Cell *c, *c0 = cells[0];
        c = cells[c0->getGlobalCellNumber(i,j,k)] ;
      }
    }
  }
  for (int i=0 ; i< dimGrid.x; i++){
    for (int j=0 ; j < dimGrid.y ; j++){
      for (int z=0 ; z < dimGrid.z ; z++){
        for (int g=0 ; g < dimBlock.x ; g++){
          for (int h=0 ; h < dimBlock.y ; h++){
            for (int p=0 ; p < dimBlock.z ; p++){
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

void set_cell_double_arrays_to_zero_CPU(CellDouble *m_c_jx,CellDouble *m_c_jy,CellDouble *m_c_jz,int size,int index,int blockDimX){
  for (int i = 0; i < size; i++) {
      setCellDoubleToZero_CPU(&(m_c_jx[i]), index);
      setCellDoubleToZero_CPU(&(m_c_jy[i]), index);
      setCellDoubleToZero_CPU(&(m_c_jz[i]), index);
  }

  while (index < CellExtent * CellExtent * CellExtent) {
      for (int i = 0; i < size; i++) {}
      index += blockDimX;
  }

  // __syncthreads();

}

void copyFromSharedMemoryToCell_CPU(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        Cell *c,
        int index ,
        dim3 dimBlock
){
  while (index < CellExtent * CellExtent * CellExtent)
   {
      copyCellDouble_CPU(c->Jx, c_jx, index);
      copyCellDouble_CPU(c->Jy, c_jy, index);
      copyCellDouble_CPU(c->Jz, c_jz, index);

      index += dimBlock.x
    }
      c->busyParticleArray = 0;
}

void emh2_Element_CPU(Cell *c, int i, int l, int k, double *Q, double *H){
  int n = c->getGlobalCellNumber(i, l, k);
  H[n] += Q[n];
}

void emh1_CPU(GPUCell **cells, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2, dim3 dimGrid, dim3 dimBlock){
  for (int i=0; i< dimGrid.x; i++ ){
    for (int j=0; j< dimGrid.y; j++ ){
      for (int k=0; k< dimGrid.z; k++ ){
        int3 i3 = make_int3(i, j, k);
        Cell *c0 = cells[0];
        emh1_Element_CPU(c0, i3, Q, H, E1, E2, c1, c2, d1, d2);
      }
    }
  }
}



void periodic_CPU(GPUCell **cells, int i_s, int k_s, double *E, int dir, int to, int from, dim3 dimGrid, dim3 dimBlock) {
  for (int i=0; i<dimGrid.x;i++){
    for (int j=0; j<dimGrid.z;j++){
      Cell *c0 = cells[0] ;
      periodicElement_CPU(c0, i + i_s, j + k_s, E, dir, to, from);
    }
  }
}

void eme_CPU(GPUCell **cells, int3 s, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2, dim3 dimGrid, dim3 dimBlock){
  for (int i=0; i<dimGrid.x * dimBlock.x ;i++){
      for (int j=0; j<dimGrid.y * dimBlock.y ;j++){
          for (int k=0; k<dimGrid.z * dimBlock.z ;k++){
            Cell *c0 = cells[0];
            s.x += i ;
            s.y += j ;
            s.z += k ;
          }
        }
      }
  emeElement_CPU(c0, s, E, H1, H2, J, c1, c2, tau, d1, d2);
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
