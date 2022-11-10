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
    unsigned triangles = 0;
    Vec3_t<float> offset(0, 0, 0);

    #pragma omp parallel
    #pragma omp single nowait
    triangles = evalTree(offset, field, mGridSize);

    return triangles;
}

unsigned TreeMeshBuilder::evalTree(const Vec3_t<float> &position, const ParametricScalarField &field, size_t cubeSize) {
    unsigned triangles = 0;
    if(cubeSize > 1) { 
        const float condition = field.getIsoLevel() + (sqrt(3) / 2) * (cubeSize * mGridResolution);
        const Vec3_t<float> point(position.x + cubeSize/2, position.y + cubeSize/2, position.z + cubeSize/2);
        const Vec3_t<float> pointNormal(point.x * mGridResolution, point.y * mGridResolution, point.z * mGridResolution);

        if ( evaluateFieldAt(pointNormal, field) > condition) {
            return 0;
        }

        for(int x = 0; x < 2; x++) {
            for(int y = 0; y < 2; y++) {
                for(int z = 0; z < 2; z++) {
                    
                    #pragma omp task shared(triangles)
                    {
                        Vec3_t<float> cube(position.x + x * (cubeSize / 2),
                                           position.y + y * (cubeSize / 2),
                                           position.z + z * (cubeSize / 2));
                        
                        unsigned tmp = evalTree(cube, field, cubeSize / 2);

                        #pragma omp critical
                        triangles += tmp;
                    }
                }
            }
        }
    } else { 
        triangles += buildCube(position, field);
    }

    #pragma omp taskwait
    return triangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{

    float returnVal = numeric_limits<float>::max();

    for (const Vec3_t<float> point : field.getPoints())
    {
        float distanceSquared = (pos.x - point.x) * (pos.x - point.x) 
                              + (pos.y - point.y) * (pos.y - point.y)
                              + (pos.z - point.z) * (pos.z - point.z);
        returnVal = std::min(returnVal, distanceSquared);
    }

    return sqrt(returnVal);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    tempTriangles.push_back(triangle);
}
