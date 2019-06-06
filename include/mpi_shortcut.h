/*
 * mpi_shortcut.h
 *
 *  Created on: Nov 5, 2014
 *      Author: snytav
 */

#ifndef MPI_SHORTCUT_H_
#define MPI_SHORTCUT_H_

#include <mpi.h>

#include "archAPI.h"


int InitMPI(int argc,char *argv[]);
// Initialize the MPI execution environment


int sumMPI(int size,double *jx,double *jy,double *jz);


int sumMPIenergy(double *e);

int CloseMPI();
// Close the MPI

int getRank();

int getSize();

#endif /* MPI_SHORTCUT_H_ */
