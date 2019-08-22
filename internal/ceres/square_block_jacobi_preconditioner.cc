// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: tristan.konolige@gmail.com (Tristan Konolige)

#include "ceres/square_block_jacobi_preconditioner.h"

#include "ceres/block_random_access_sparse_matrix.h"
#include "ceres/block_random_access_sparse_matrix.h"

namespace ceres {
namespace internal {

SquareBlockJacobiPreconditioner::SquareBlockJacobiPreconditioner(
    const std::vector<int>& blocks) : blocks_(blocks) {
  m_.reset(new BlockRandomAccessDiagonalMatrix(blocks_));
}

bool SquareBlockJacobiPreconditioner::Update(const LinearOperator& A, const double* D) {
  LOG(FATAL) << "Unimplemented. Call UpdateImpl instead";
  return false;
}

bool SquareBlockJacobiPreconditioner::UpdateImpl(const BlockRandomAccessSparseMatrix& A, const double* D) {
  BlockRandomAccessSparseMatrix* sc = const_cast<BlockRandomAccessSparseMatrix*>(&A);
  for (int i = 0; i < blocks_.size(); ++i) {
    const int block_size = blocks_[i];

    int sc_r, sc_c, sc_row_stride, sc_col_stride;
    CellInfo* sc_cell_info =
        sc->GetCell(i, i, &sc_r, &sc_c, &sc_row_stride, &sc_col_stride);
    CHECK(sc_cell_info != nullptr);
    MatrixRef sc_m(sc_cell_info->values, sc_row_stride, sc_col_stride);

    int pre_r, pre_c, pre_row_stride, pre_col_stride;
    CellInfo* pre_cell_info = m_->GetCell(
        i, i, &pre_r, &pre_c, &pre_row_stride, &pre_col_stride);
    CHECK(pre_cell_info != nullptr);
    MatrixRef pre_m(pre_cell_info->values, pre_row_stride, pre_col_stride);

    pre_m.block(pre_r, pre_c, block_size, block_size) =
        sc_m.block(sc_r, sc_c, block_size, block_size);
  }
  m_->Invert();

  return true;
}

SquareBlockJacobiPreconditioner::~SquareBlockJacobiPreconditioner() {}

void SquareBlockJacobiPreconditioner::RightMultiply(const double* x,
                                                    double* y) const {
  m_->RightMultiply(x, y);
}

}  // namespace internal
}  // namespace ceres
