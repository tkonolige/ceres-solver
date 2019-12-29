#pragma once

#include "ceres/linear_solver.h"
#include "ceres/preconditioner.h"
#include "ceres/schur_eliminator.h"
#include "ceres/implicit_schur_complement.h"

#include "julia.h"

namespace ceres {
namespace internal {

  class MultigridPreconditioner : public BlockSparseMatrixPreconditioner {
    public:
      MultigridPreconditioner(const CompressedRowBlockStructure& bs, const Preconditioner::Options& options, ImplicitSchurComplement* complement);
      MultigridPreconditioner(const MultigridPreconditioner&) = delete;
      void operator=(const MultigridPreconditioner&) = delete;

      void RightMultiply(const double* x, double* y) const final;
      int num_rows() const final;
      int64_t num_nonzeros() const final;
      bool UpdateExplicit(const CompressedRowSparseMatrix& A, const double* D, const TrustRegionMinimizer* minimizer);

    private:
      bool UpdateImpl(const BlockSparseMatrix& A, const double* D, const TrustRegionMinimizer* minimizer) final;
      void InitStorage(const CompressedRowBlockStructure*);

      jl_value_t* mg_;
      int num_rows_;
      Preconditioner::Options options_;
      bool use_implicit_;
      std::unique_ptr<SchurEliminatorBase> eliminator_; // eliminator for constructing explicit schur complement
      ImplicitSchurComplement* schur_complement_; // implicit schur complement
      std::vector<int> blocks_;
      std::unique_ptr<BlockRandomAccessMatrix> lhs_;
      std::unique_ptr<double[]> dummy_rhs;
      std::unique_ptr<double[]> dummy_b;
  };
}
}
