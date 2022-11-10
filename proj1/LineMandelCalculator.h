/**
 * @file LineMandelCalculator.h
 * @author Simon Sestak <xsesta06@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 7.11.2021
 */

#ifndef LINEMANDELCALCULATOR_H
#define LINEMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    int *data;
    float *tmpReal;
    float *tmpImag;
    float *tmpR2;
    float *tmpI2;
    bool *tmpSkip;
};
#endif
