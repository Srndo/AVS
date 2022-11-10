/**
 * @file LineMandelCalculator.cc
 * @author Simon Sestak <xsesta06@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 7.11.2021
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>

#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) : BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
    data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
    tmpReal = (float *)(_mm_malloc(width * sizeof(float), 64));
    tmpImag = (float *)(_mm_malloc(width * sizeof(float), 64));
    tmpR2 = (float *)(_mm_malloc(width * sizeof(float), 64));
    tmpI2 = (float *)(_mm_malloc(width * sizeof(float), 64));
}

LineMandelCalculator::~LineMandelCalculator() {
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

int *LineMandelCalculator::calculateMandelbrot () {
    int *pdata = data;
    float *pReal = tmpReal;
    float *pImag = tmpImag;
    
    float *r2 = tmpR2;
    float *i2 = tmpI2;

    float real[width];
    float imag[width];

    int values[width];

    for (int i = 0; i < height; i++) // i == "selected" row
    {
        imag[i] = y_start + i * dy;
        int j = 0;
        #pragma omp simd linear(j)
        for (; j < width; j++) { 
            real[j] = x_start + j * dx;
            values[j] = limit;
        }

        int iterace = 0;
        register int counter = 0;
        #pragma omp simd linear(iterace) reduction(+: counter)
        for (; iterace < limit; ++iterace)
        {
            int j = 0;
            #pragma omp simd aligned(pdata, pReal, pImag, r2, i2: 64) linear(j) simdlen(80)
            for (;  j < width; j++) {
                if (iterace == 0) {
                    pReal[j] = real[j];
                    pImag[j] = imag[i];
                }

                r2[j] = pReal[j] * pReal[j];
                i2[j] = pImag[j] * pImag[j];

                if ((r2[j] + i2[j]) > 4.0f && values[j] == limit) {
                    values[j] = iterace;
                    counter += 1;
                } else {
                    pImag[j] = 2.0f * pReal[j] * pImag[j] + imag[i];
                    pReal[j] = r2[j] - i2[j] + real[j];
                }
            }
            if (counter == width)
                iterace = limit;
    }
        #pragma omp simd
        for (int k = 0; k < width; k++)
            *(pdata++) = values[k];
    }
    return data;
}

