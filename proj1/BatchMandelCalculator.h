/**
 * @file BatchMandelCalculator.h
  * @author Simon Sestak <xsesta06@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 7.11.2021
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int * calculateMandelbrot();

private:
	int blockSize = 64;
   	int *data;
    double *tmpReal;
    double *tmpImag;
    double *tmpR2;
    double *tmpI2;
};

#endif