#ifndef GPUCELL_H_
#define GPUCELL_H_

#include "cell.h"
#include "archAPI.h"

class CPUCell : public Cell {
public:

    CPUCell() {}

    ~CPUCell() {}

    CPUCell(int i1, int l1, int k1, double Lx, double Ly, double Lz, int Nx1, int Ny1, int Nz1, double tau1) : Cell(i1, l1, k1, Lx, Ly, Lz, Nx1, Ny1, Nz1, tau1) {}

    CPUCell *copyCellToDevice();

    void copyCellFromDevice(CPUCell *d_src, CPUCell *h_dst);

    CPUCell *allocateCopyCellFromDevice();

    double compareToCell(Cell &d_src);
};

#endif /* GPUCELL_H_ */
