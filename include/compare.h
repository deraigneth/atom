/*
 * compare.h
 *
 *  Created on: Jun 2, 2018
 *      Author: snytav
 */

#ifndef COMPARE_H_
#define COMPARE_H_

#include <string>
#include <string>
#include <math.h>

#define TOLERANCE 1e-15

double compare(double *a,double *b,int num,std::string legend,double tol);
/* fonction qui compare tous les éléments de la liste et qui vérifie que chaque différence en valeur absolue entre deux éléments de deux listes est inférieure à la tolérance fixée, elle affiche ensuite un message.
 * Si de plus t=t/num, elle affiche un autre message*/

int comd(double a,double b);
/*fonction qui vérifie si deux flottant sont différent ou pas*/

#endif /* COMPARE_H_ */
