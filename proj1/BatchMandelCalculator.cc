/**
 * @file BatchMandelCalculator.cc
 * @author Simon Sestak <xsesta06@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 7.11.2021
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
    tmpReal = (double *)(_mm_malloc(blockSize * sizeof(double), 64));
    tmpImag = (double *)(_mm_malloc(blockSize * sizeof(double), 64));
    tmpR2 = (double *)(_mm_malloc(blockSize * sizeof(double), 64));
    tmpI2 = (double *)(_mm_malloc(blockSize * sizeof(double), 64));
}

BatchMandelCalculator::~BatchMandelCalculator() {
	_mm_free(data);
    _mm_free(tmpReal);
    _mm_free(tmpImag);
    _mm_free(tmpI2);
    _mm_free(tmpR2);
    data = NULL;
    tmpReal = NULL;
    tmpImag = NULL;
    tmpR2 = NULL;
    tmpI2 = NULL;
}


int * BatchMandelCalculator::calculateMandelbrot () {
    int *pdata = data;
    double *pReal = tmpReal;
    double *pImag = tmpImag;
    
    double *r2 = tmpR2;
    double *i2 = tmpI2;

    double real[blockSize];
    int values[blockSize];
    double col[blockSize];

    for(int row = 0; row < height; row++) {

        double imag = y_start + row * dy;

        for(int colTi = 0; colTi < width / blockSize; colTi++) { 
            
            int i = 0;
            #pragma omp simd linear(i)
            for(; i < blockSize; i++)  {
                col[i] = colTi * blockSize + i;
                real[i] = x_start + col[i] * dx;
                values[i] = limit;
            }

            int counter = 0;
            for(int iter = 0; iter < limit; iter++) {
                #pragma omp simd aligned(pdata, pReal, pImag, r2, i2: 64) reduction(+: counter) simdlen(80)
                for(int colLi = 0; colLi < blockSize; colLi++) { 
                    if(iter == 0) { 
                        pReal[colLi] = real[colLi];
                        pImag[colLi] = imag;
                    }

                    r2[colLi] = pReal[colLi] * pReal[colLi];
                    i2[colLi] = pImag[colLi] * pImag[colLi];

                    if(r2[colLi] + i2[colLi] > 4.0 && values[colLi] == limit ) {
                        values[colLi] = iter;
                        counter += 1;
                    } else { 
                        pImag[colLi] = 2.0 * pReal[colLi] * pImag[colLi] + imag;
                        pReal[colLi] = r2[colLi] - i2[colLi] + real[colLi];
                    }
                }
                if (counter == blockSize)
                    iter = limit;
            }
            #pragma omp simd aligned(pdata: 64)
            for (int k = 0; k < blockSize; k++)
                *(pdata++) = values[k];
        }
    }
    return data;
}