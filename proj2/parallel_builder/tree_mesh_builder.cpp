/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Simon Sestak <xsesta06@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

using namespace std;

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    unsigned totalTriangles = 0;

    if(cubeSize > 1) {
        Vec3_t<float> middlePoint((position.x + cubeSize / 2) * mGridResolution,
                                  (position.y + cubeSize / 2) * mGridResolution,
                                  (position.z + cubeSize / 2) * mGridResolution);

        if(evaluateFieldAt(middlePoint, field) > (field.getIsoLevel() + (sqrt(3)/2) * (cubeSize * mGridResolution)))
            return 0;
        
        for(int x = 0; x < 2; x++) {
            for(int y = 0; y < 2; y++) {
                for(int z = 0; z < 2; z++) {
                    
                    #pragma omp task shared(totalTriangles)
                    {
                        Vec3_t<float> cube(position.x + x * (cubeSize / 2),
                                           position.y + y * (cubeSize / 2),
                                           position.z + z * (cubeSize / 2));
                        
                        unsigned tmp = evalTree(cube, field, cubeSize / 2);

                        #pragma omp critical
                        totalTriangles += tmp;
                    }
                }
            }
        }
    } else {
        unsigned tmp = buildCube(position, field);

        #pragma omp critical
        totalTriangles += tmp;
    }

    #pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned totalTriangles = 0;
    Vec3_t<float> offset(0, 0, 0);

    #pragma omp parallel
    #pragma omp single
    totalTriangles = evalTree(offset, field, mGridSize);

    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float return_value = numeric_limits<float>::max();

    for(unsigned i = 0; i < count; ++i)
    {
        float distance  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distance       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distance       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        return_value = min(value, distanceSquared);
    }

    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    mTriangles.push_back(triangle);
}
