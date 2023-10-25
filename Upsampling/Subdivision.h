#pragma once

#include "../CWFTypes.h"

namespace ComplexWrinkleField {
    // Standard Loop subdivision
    void Subdivide(const Mesh& mesh,                    // input mesh
                   Mesh& upmesh,                        // upsampled mesh
                   int level,                           // upsampling level
                   bool fixedBnd = false,               // fix boundary points
                   SparseMatrixX* upS0 = nullptr        // upsampling matrix
                   );

    // CWF Loop subdivision
    void Subdivide(const CWF& cwf,								    // input CWF
                   CWF& upcwf,									    // output CWF
                   int level,                                       // upsampling level
                   bool fixedBnd = false,                           // fix boundary points
                   SparseMatrixX* upS0 = nullptr,					// upsampled matrix
                   SparseMatrixX* upS1 = nullptr,					// upsampled matrix
                   ComplexSparseMatrixX* upComplexS0 = nullptr);	// upsampled matrix
}