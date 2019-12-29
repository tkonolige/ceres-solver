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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/schur_complement_solver.h"

#include <algorithm>
#include <ctime>
#include <memory>
#include <set>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "ceres/block_random_access_dense_matrix.h"
#include "ceres/block_random_access_matrix.h"
#include "ceres/block_random_access_sparse_matrix.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/conjugate_gradients_solver.h"
#include "ceres/detect_structure.h"
#include "ceres/internal/eigen.h"
#include "ceres/lapack.h"
#include "ceres/linear_solver.h"
#include "ceres/multigrid.h"
#include "ceres/sparse_cholesky.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "ceres/wall_time.h"

namespace ceres {
namespace internal {

using std::make_pair;
using std::pair;
using std::set;
using std::vector;

namespace {

class BlockRandomAccessSparseMatrixAdapter : public LinearOperator {
 public:
  explicit BlockRandomAccessSparseMatrixAdapter(
      const BlockRandomAccessSparseMatrix& m)
      : m_(m) {}

  virtual ~BlockRandomAccessSparseMatrixAdapter() {}

  // y = y + Ax;
  void RightMultiply(const double* x, double* y) const final {
    m_.SymmetricRightMultiply(x, y);
  }

  // y = y + A'x;
  void LeftMultiply(const double* x, double* y) const final {
    m_.SymmetricRightMultiply(x, y);
  }

  int num_rows() const final { return m_.num_rows(); }
  int num_cols() const final { return m_.num_rows(); }
  int64_t num_nonzeros() const final { return m_.num_nonzeros(); }

 private:
  const BlockRandomAccessSparseMatrix& m_;
};

class BlockRandomAccessDiagonalMatrixAdapter : public LinearOperator {
 public:
  explicit BlockRandomAccessDiagonalMatrixAdapter(
      const BlockRandomAccessDiagonalMatrix& m)
      : m_(m) {}

  virtual ~BlockRandomAccessDiagonalMatrixAdapter() {}

  // y = y + Ax;
  void RightMultiply(const double* x, double* y) const final {
    m_.RightMultiply(x, y);
  }

  // y = y + A'x;
  void LeftMultiply(const double* x, double* y) const final {
    m_.RightMultiply(x, y);
  }

  int num_rows() const final { return m_.num_rows(); }
  int num_cols() const final { return m_.num_rows(); }
  int64_t num_nonzeros() const final { return m_.num_nonzeros(); }

 private:
  const BlockRandomAccessDiagonalMatrix& m_;
};

class BlockDiagonalPreconditionerAdapter :
  public BlockSparseMatrixPreconditioner {
  public:
    explicit BlockDiagonalPreconditionerAdapter(
        const std::vector<int>& blocks)
      : m_(blocks), blocks_(blocks) {}

    void RightMultiply(const double* x, double* y) const final {
      m_.RightMultiply(x, y);
    }

    int num_rows() const final { return m_.num_rows(); }
    int num_cols() const final { return m_.num_rows(); }
    int64_t num_nonzeros() const final { return m_.num_nonzeros(); }

    bool Update_(const BlockRandomAccessSparseMatrix& sc_,
        const double* D, const TrustRegionMinimizer* minimizer) {
      BlockRandomAccessSparseMatrix& sc = const_cast<BlockRandomAccessSparseMatrix&>(sc_);
      // Extract block diagonal from the Schur complement to construct the
      // schur_jacobi preconditioner.
      for (int i = 0; i < blocks_.size(); ++i) {
        const int block_size = blocks_[i];

        int sc_r, sc_c, sc_row_stride, sc_col_stride;
        CellInfo* sc_cell_info =
          sc.GetCell(i, i, &sc_r, &sc_c, &sc_row_stride, &sc_col_stride);
        CHECK(sc_cell_info != nullptr);
        MatrixRef sc_m(sc_cell_info->values, sc_row_stride, sc_col_stride);

        int pre_r, pre_c, pre_row_stride, pre_col_stride;
        CellInfo* pre_cell_info = m_.GetCell(
            i, i, &pre_r, &pre_c, &pre_row_stride, &pre_col_stride);
        CHECK(pre_cell_info != nullptr);
        MatrixRef pre_m(pre_cell_info->values, pre_row_stride, pre_col_stride);

        pre_m.block(pre_r, pre_c, block_size, block_size) =
          sc_m.block(sc_r, sc_c, block_size, block_size);
      }
      m_.Invert();

      return true;
    }

  private:
    bool UpdateImpl(const BlockSparseMatrix& sc_,
        const double* D, const TrustRegionMinimizer* minimizer) {
      return true;
    }

    BlockRandomAccessDiagonalMatrix m_;
    const std::vector<int>& blocks_;
};

}  // namespace

LinearSolver::Summary SchurComplementSolver::SolveImpl(
    BlockSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x,
    const TrustRegionMinimizer* minimizer) {
  EventLogger event_logger("SchurComplementSolver::Solve");
  const double setup_start_time = WallTimeInSeconds();

  const CompressedRowBlockStructure* bs = A->block_structure();
  if (eliminator_.get() == NULL) {
    const int num_eliminate_blocks = options_.elimination_groups[0];
    const int num_f_blocks = bs->cols.size() - num_eliminate_blocks;

    DetectStructure(*bs,
                    num_eliminate_blocks,
                    &options_.row_block_size,
                    &options_.e_block_size,
                    &options_.f_block_size);
    InitStorage(bs);

    // For the special case of the static structure <2,3,6> with
    // exactly one f block use the SchurEliminatorForOneFBlock.
    //
    // TODO(sameeragarwal): A more scalable template specialization
    // mechanism that does not cause binary bloat.
    if (options_.row_block_size == 2 &&
        options_.e_block_size == 3 &&
        options_.f_block_size == 6 &&
        num_f_blocks == 1) {
      eliminator_.reset(new SchurEliminatorForOneFBlock<2, 3, 6>);
    } else {
      eliminator_.reset(SchurEliminatorBase::Create(options_));
    }

    CHECK(eliminator_);
    const bool kFullRankETE = true;
    eliminator_->Init(num_eliminate_blocks, kFullRankETE, bs);
  }


  std::fill(x, x + A->num_cols(), 0.0);
  event_logger.AddEvent("Setup");
  const double setup_end_time = WallTimeInSeconds();

  const double schur_start_time = WallTimeInSeconds();
  eliminator_->Eliminate(BlockSparseMatrixData(*A),
                         b,
                         per_solve_options.D,
                         lhs_.get(),
                         rhs_.get());
  event_logger.AddEvent("Eliminate");
  const double schur_end_time = WallTimeInSeconds();

  const double setup_start_time2 = WallTimeInSeconds();
  UpdatePreconditioner(minimizer);
  const double setup_end_time2 = WallTimeInSeconds();


  const double solve_start_time = WallTimeInSeconds();
  double* reduced_solution = x + A->num_cols() - lhs_->num_cols();
  LinearSolver::Summary summary =
      SolveReducedLinearSystem(per_solve_options, reduced_solution, minimizer);
  event_logger.AddEvent("ReducedSolve");
  const double solve_end_time = WallTimeInSeconds();

  const double schur_start_time2 = WallTimeInSeconds();
  if (summary.termination_type == LINEAR_SOLVER_SUCCESS) {
    eliminator_->BackSubstitute(
        BlockSparseMatrixData(*A), b, per_solve_options.D, reduced_solution, x);
    event_logger.AddEvent("BackSubstitute");
  }
  const double schur_end_time2 = WallTimeInSeconds();

  summary.setup_time = setup_end_time - setup_start_time + setup_end_time2 - setup_start_time2;
  summary.solve_time = solve_end_time - solve_start_time;
  summary.schur_time = schur_end_time - schur_start_time + schur_end_time2 - schur_start_time2;

  return summary;
}

// Initialize a BlockRandomAccessDenseMatrix to store the Schur
// complement.
void DenseSchurComplementSolver::InitStorage(
    const CompressedRowBlockStructure* bs) {
  const int num_eliminate_blocks = options().elimination_groups[0];
  const int num_col_blocks = bs->cols.size();

  vector<int> blocks(num_col_blocks - num_eliminate_blocks, 0);
  for (int i = num_eliminate_blocks, j = 0; i < num_col_blocks; ++i, ++j) {
    blocks[j] = bs->cols[i].size;
  }

  set_lhs(new BlockRandomAccessDenseMatrix(blocks));
  set_rhs(new double[lhs()->num_rows()]);
}

// Solve the system Sx = r, assuming that the matrix S is stored in a
// BlockRandomAccessDenseMatrix. The linear system is solved using
// Eigen's Cholesky factorization.
LinearSolver::Summary DenseSchurComplementSolver::SolveReducedLinearSystem(
    const LinearSolver::PerSolveOptions& per_solve_options, double* solution,
    const TrustRegionMinimizer* minimizer) {
  LinearSolver::Summary summary;
  summary.num_iterations = 0;
  summary.termination_type = LINEAR_SOLVER_SUCCESS;
  summary.message = "Success.";

  const BlockRandomAccessDenseMatrix* m =
      down_cast<const BlockRandomAccessDenseMatrix*>(lhs());
  const int num_rows = m->num_rows();

  // The case where there are no f blocks, and the system is block
  // diagonal.
  if (num_rows == 0) {
    return summary;
  }

  summary.num_iterations = 1;

  if (options().dense_linear_algebra_library_type == EIGEN) {
    Eigen::LLT<Matrix, Eigen::Upper> llt =
        ConstMatrixRef(m->values(), num_rows, num_rows)
            .selfadjointView<Eigen::Upper>()
            .llt();
    if (llt.info() != Eigen::Success) {
      summary.termination_type = LINEAR_SOLVER_FAILURE;
      summary.message =
          "Eigen failure. Unable to perform dense Cholesky factorization.";
      return summary;
    }

    VectorRef(solution, num_rows) = llt.solve(ConstVectorRef(rhs(), num_rows));
  } else {
    VectorRef(solution, num_rows) = ConstVectorRef(rhs(), num_rows);
    summary.termination_type = LAPACK::SolveInPlaceUsingCholesky(
        num_rows, m->values(), solution, &summary.message);
  }

  return summary;
}

void DenseSchurComplementSolver::UpdatePreconditioner(const TrustRegionMinimizer* minimizer) {
}

SparseSchurComplementSolver::SparseSchurComplementSolver(
    const LinearSolver::Options& options)
    : SchurComplementSolver(options) {
  if (options.type != ITERATIVE_SCHUR) {
    sparse_cholesky_ = SparseCholesky::Create(options);
  }
}

SparseSchurComplementSolver::~SparseSchurComplementSolver() {}

// Determine the non-zero blocks in the Schur Complement matrix, and
// initialize a BlockRandomAccessSparseMatrix object.
void SparseSchurComplementSolver::InitStorage(
    const CompressedRowBlockStructure* bs) {
  if (preconditioner_.get() == NULL && options().preconditioner_type == JULIA_MULTIGRID) {
    Preconditioner::Options preconditioner_options;
    preconditioner_options.type = options().preconditioner_type;
    preconditioner_options.visibility_clustering_type =
      options().visibility_clustering_type;
    preconditioner_options.sparse_linear_algebra_library_type =
      options().sparse_linear_algebra_library_type;
    preconditioner_options.num_threads = options().num_threads;
    preconditioner_options.row_block_size = options().row_block_size;
    // preconditioner_options.e_block_size = options().e_block_size;
    // preconditioner_options.f_block_size = options().f_block_size;
    // e_block_size and f_block_size are not set unless we are using a direct solver
    preconditioner_options.e_block_size = 3;
    preconditioner_options.f_block_size = 9;
    preconditioner_options.elimination_groups = options().elimination_groups;
    preconditioner_.reset(new MultigridPreconditioner(*bs, preconditioner_options, NULL));
  }


  const int num_eliminate_blocks = options().elimination_groups[0];
  const int num_col_blocks = bs->cols.size();
  const int num_row_blocks = bs->rows.size();

  blocks_.resize(num_col_blocks - num_eliminate_blocks, 0);
  for (int i = num_eliminate_blocks; i < num_col_blocks; ++i) {
    blocks_[i - num_eliminate_blocks] = bs->cols[i].size;
  }

  set<pair<int, int>> block_pairs;
  for (int i = 0; i < blocks_.size(); ++i) {
    block_pairs.insert(make_pair(i, i));
  }

  int r = 0;
  while (r < num_row_blocks) {
    int e_block_id = bs->rows[r].cells.front().block_id;
    if (e_block_id >= num_eliminate_blocks) {
      break;
    }
    vector<int> f_blocks;

    // Add to the chunk until the first block in the row is
    // different than the one in the first row for the chunk.
    for (; r < num_row_blocks; ++r) {
      const CompressedRow& row = bs->rows[r];
      if (row.cells.front().block_id != e_block_id) {
        break;
      }

      // Iterate over the blocks in the row, ignoring the first
      // block since it is the one to be eliminated.
      for (int c = 1; c < row.cells.size(); ++c) {
        const Cell& cell = row.cells[c];
        f_blocks.push_back(cell.block_id - num_eliminate_blocks);
      }
    }

    sort(f_blocks.begin(), f_blocks.end());
    f_blocks.erase(unique(f_blocks.begin(), f_blocks.end()), f_blocks.end());
    for (int i = 0; i < f_blocks.size(); ++i) {
      for (int j = i + 1; j < f_blocks.size(); ++j) {
        block_pairs.insert(make_pair(f_blocks[i], f_blocks[j]));
      }
    }
  }

  // Remaining rows do not contribute to the chunks and directly go
  // into the schur complement via an outer product.
  for (; r < num_row_blocks; ++r) {
    const CompressedRow& row = bs->rows[r];
    CHECK_GE(row.cells.front().block_id, num_eliminate_blocks);
    for (int i = 0; i < row.cells.size(); ++i) {
      int r_block1_id = row.cells[i].block_id - num_eliminate_blocks;
      for (int j = 0; j < row.cells.size(); ++j) {
        int r_block2_id = row.cells[j].block_id - num_eliminate_blocks;
        if (r_block1_id <= r_block2_id) {
          block_pairs.insert(make_pair(r_block1_id, r_block2_id));
        }
      }
    }
  }

  set_lhs(new BlockRandomAccessSparseMatrix(blocks_, block_pairs));
  set_rhs(new double[lhs()->num_rows()]);
}

LinearSolver::Summary SparseSchurComplementSolver::SolveReducedLinearSystem(
    const LinearSolver::PerSolveOptions& per_solve_options, double* solution,
    const TrustRegionMinimizer* minimizer) {
  if (options().type == ITERATIVE_SCHUR) {
    return SolveReducedLinearSystemUsingConjugateGradients(per_solve_options,
                                                           solution, minimizer);
  }

  LinearSolver::Summary summary;
  summary.num_iterations = 0;
  summary.termination_type = LINEAR_SOLVER_SUCCESS;
  summary.message = "Success.";

  const TripletSparseMatrix* tsm =
      down_cast<const BlockRandomAccessSparseMatrix*>(lhs())->matrix();
  if (tsm->num_rows() == 0) {
    return summary;
  }

  std::unique_ptr<CompressedRowSparseMatrix> lhs;
  const CompressedRowSparseMatrix::StorageType storage_type =
      sparse_cholesky_->StorageType();
  if (storage_type == CompressedRowSparseMatrix::UPPER_TRIANGULAR) {
    lhs.reset(CompressedRowSparseMatrix::FromTripletSparseMatrix(*tsm));
    lhs->set_storage_type(CompressedRowSparseMatrix::UPPER_TRIANGULAR);
  } else {
    lhs.reset(
        CompressedRowSparseMatrix::FromTripletSparseMatrixTransposed(*tsm));
    lhs->set_storage_type(CompressedRowSparseMatrix::LOWER_TRIANGULAR);
  }

  *lhs->mutable_col_blocks() = blocks_;
  *lhs->mutable_row_blocks() = blocks_;

  summary.num_iterations = 1;
  summary.termination_type = sparse_cholesky_->FactorAndSolve(
      lhs.get(), rhs(), solution, &summary.message);
  return summary;
}

LinearSolver::Summary
SparseSchurComplementSolver::SolveReducedLinearSystemUsingConjugateGradients(
    const LinearSolver::PerSolveOptions& per_solve_options, double* solution, const TrustRegionMinimizer* minimizer) {
  CHECK(options().use_explicit_schur_complement);
  const int num_rows = lhs()->num_rows();
  // The case where there are no f blocks, and the system is block
  // diagonal.
  if (num_rows == 0) {
    LinearSolver::Summary summary;
    summary.num_iterations = 0;
    summary.termination_type = LINEAR_SOLVER_SUCCESS;
    summary.message = "Success.";
    return summary;
  }

  BlockRandomAccessSparseMatrix* sc = down_cast<BlockRandomAccessSparseMatrix*>(
      const_cast<BlockRandomAccessMatrix*>(lhs()));
  std::unique_ptr<LinearOperator> lhs_adapter(
      new BlockRandomAccessSparseMatrixAdapter(*sc));

  LinearSolver::Options cg_options;
  cg_options.min_num_iterations = options().min_num_iterations;
  cg_options.max_num_iterations = options().max_num_iterations;
  ConjugateGradientsSolver cg_solver(cg_options);

  LinearSolver::PerSolveOptions cg_per_solve_options;
  cg_per_solve_options.r_tolerance = per_solve_options.r_tolerance;
  cg_per_solve_options.q_tolerance = per_solve_options.q_tolerance;
  cg_per_solve_options.preconditioner = preconditioner_.get();

  return cg_solver.Solve(
      lhs_adapter.get(), rhs(), cg_per_solve_options, solution, minimizer);
}

void SparseSchurComplementSolver::UpdatePreconditioner(const TrustRegionMinimizer* minimizer) {
  BlockRandomAccessSparseMatrix* sc = down_cast<BlockRandomAccessSparseMatrix*>(
      const_cast<BlockRandomAccessMatrix*>(lhs()));

  if(options().preconditioner_type == JULIA_MULTIGRID) {
    const TripletSparseMatrix* tsm = sc->matrix();

    std::unique_ptr<CompressedRowSparseMatrix> l;
    l.reset(CompressedRowSparseMatrix::FromTripletSparseMatrix(*tsm));
    l->set_storage_type(CompressedRowSparseMatrix::UNSYMMETRIC);

    *l->mutable_col_blocks() = blocks_;
    *l->mutable_row_blocks() = blocks_;

    ((MultigridPreconditioner*)preconditioner_.get())->UpdateExplicit(*l.get(), NULL, minimizer);
  } else if(options().preconditioner_type == SCHUR_JACOBI) {
    if (preconditioner_.get() == NULL) {
      preconditioner_.reset(new BlockDiagonalPreconditionerAdapter(blocks_));
    }
    BlockDiagonalPreconditionerAdapter* prec =
      down_cast<BlockDiagonalPreconditionerAdapter*>(preconditioner_.get());
    prec->Update_(*sc, NULL, minimizer);
  } else {
    LOG(FATAL) << "Invalid preconditioner " << options().preconditioner_type;
  }
}

}  // namespace internal
}  // namespace ceres
