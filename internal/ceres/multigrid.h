#pragma once

#include "ceres/linear_solver.h"
#include "ceres/preconditioner.h"

#include "julia.h"

namespace ceres {
namespace internal {

  class MultigridPreconditioner : public CompressedRowSparseMatrixPreconditioner {
    public:
      MultigridPreconditioner(const CompressedRowBlockStructure& bs, const Preconditioner::Options& options);
      MultigridPreconditioner(const MultigridPreconditioner&) = delete;
      void operator=(const MultigridPreconditioner&) = delete;

      void RightMultiply(const double* x, double* y) const final;
      int num_rows() const final;
      int64_t num_nonzeros() const final;

    private:
      bool UpdateImpl(const CompressedRowSparseMatrix& A, const double* D, const TrustRegionMinimizer* minimizer) final;

      jl_value_t* mg_;
      int num_rows_;
      Preconditioner::Options options_;
  };
}
}
