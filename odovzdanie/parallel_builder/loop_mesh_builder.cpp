/**
 * @file    loop_mesh_builder.cpp
 *
 * @author  Simon Sestak <xsesta06@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP loops
 *
 * @date    28.11.2021
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "loop_mesh_builder.h"
using namespace std;

LoopMeshBuilder::LoopMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "OpenMP Loop")
{

}

unsigned LoopMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    const size_t cubesCount = mGridSize * mGridSize * mGridSize;
    unsigned triangles = 0;

    #pragma omp parallel for default(none) shared(cubesCount, field) reduction(+:triangles) schedule(dynamic, 32)
    for (size_t i = 0; i < cubesCount; i++)
    {
        const Vec3_t<float> cubeOffset(i % mGridSize,
                                      (i / mGridSize) % mGridSize,
                                       i / (mGridSize * mGridSize));
        triangles += buildCube(cubeOffset, field);
    }

    return triangles;
}

float LoopMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());
    float value = numeric_limits<float>::max();

    for(unsigned i = 0; i < count; ++i) {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);
        value = min(value, distanceSquared);
    }

    return sqrt(value);
}

void LoopMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    tempTriangles.push_back(triangle);
}
