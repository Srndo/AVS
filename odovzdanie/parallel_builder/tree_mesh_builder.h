/**
 * @file    tree_mesh_builder.h
 *
 * @author  Simon Sestak <xsesta06@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"
using namespace std;

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const { return tempTriangles.data(); }
    unsigned evalTree(const Vec3_t<float> &position, const ParametricScalarField &field, size_t cubeSize);
    vector<Triangle_t> tempTriangles;
};

#endif // TREE_MESH_BUILDER_H
