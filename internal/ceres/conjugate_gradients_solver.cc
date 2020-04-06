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
//
// A preconditioned conjugate gradients solver
// (ConjugateGradientsSolver) for positive semidefinite linear
// systems.
//
// We have also augmented the termination criterion used by this
// solver to support not just residual based termination but also
// termination based on decrease in the value of the quadratic model
// that CG optimizes.

#include "ceres/conjugate_gradients_solver.h"

#include <iostream>
#include <cmath>
#include <cstddef>
#include "ceres/internal/eigen.h"
#include "ceres/linear_operator.h"
#include "ceres/trust_region_minimizer.h"
#include "ceres/stringprintf.h"
#include "ceres/types.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
#include <gflags/gflags.h>
#include <array>
#include <Eigen/QR>
#include <Eigen/Geometry>

DEFINE_bool(fcg, false, "File to dump multigrid hierarchy to");
DEFINE_bool(random_rhs, false, "Use random right hand side");
DEFINE_double(rtol, -1, "Relative solve tolerance, <= 0 disables");


namespace ceres {
namespace internal {
namespace {

double csc(double x) {
  return 1.0/sin(x);
}

// Linearized rotation or R around axis
void linearized_rotation(const double* R, const std::array<double, 3> axis, double* out) {
  double squared_norm = R[0] * R[0] + R[1] * R[1] + R[2] * R[2];
  if (squared_norm < 1e-8) {
    out[0] = axis[0];
    out[1] = axis[1];
    out[2] = axis[2];
  } else {
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> RR(R);
    double alpha = RR.norm();
    Eigen::Matrix<double, 3, 1> r = RR / alpha;
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> a(axis.data());

    Eigen::Map<Eigen::Matrix<double, 3, 1>> o(out);
    o = 0.5 * abs(csc(alpha/2))*(r*(-alpha*cos(alpha/2)+sqrt(2-2*cos(alpha)))*r.dot(-a)+alpha*(-a*cos(alpha/2)+r.cross(-a)*sin(alpha/2)));
  }
}

Eigen::MatrixXd nullspace(const Eigen::VectorXd& params) {
    Eigen::MatrixXd ns(params.size(), 7);
    for(int i = 0; i < params.size()/9; i++) {
      int off = i * 9;
      // camera translations
      std::array<double, 3> trans_x = {1, 0, 0};
      AngleAxisRotatePoint(params.data() + off, trans_x.data(), ns.col(0).data()+off+3);
      std::array<double, 3> trans_y = {0, 1, 0};
      AngleAxisRotatePoint(params.data() + off, trans_y.data(), ns.col(1).data()+off+3);
      std::array<double, 3> trans_z = {0, 0, 1};
      AngleAxisRotatePoint(params.data() + off, trans_z.data(), ns.col(2).data()+off+3);

      // scaling
      ns(off+3,3) = params[off+3];
      ns(off+4,3) = params[off+4];
      ns(off+5,3) = params[off+5];

      // rotations
      linearized_rotation(params.data() + off, {1, 0, 0}, ns.col(4).data()+off);
      linearized_rotation(params.data() + off, {0, 1, 0}, ns.col(5).data()+off);
      linearized_rotation(params.data() + off, {0, 0, 1}, ns.col(6).data()+off);
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(ns);
    Eigen::MatrixXd thinQ;
    thinQ.setIdentity(params.size(), 7);
    qr.householderQ().applyThisOnTheLeft(thinQ);
    return thinQ;
}

template<class T>
void orthogonalize(const Eigen::MatrixXd& ns, T rhs) {
  for(int i = 0; i < ns.cols(); i++) {
    double d = ns.col(i).dot(rhs);
    rhs = rhs - d * ns.col(i);
  }
}

Eigen::VectorXd compatible_random_rhs(const Eigen::MatrixXd& ns) {
  Eigen::VectorXd rhs = Eigen::VectorXd::Random(ns.rows());
  orthogonalize(ns, rhs);
  return rhs;
}

bool IsZeroOrInfinity(double x) {
  return ((x == 0.0) || std::isinf(x));
}

}  // namespace

ConjugateGradientsSolver::ConjugateGradientsSolver(
    const LinearSolver::Options& options)
    : options_(options) {
}

LinearSolver::Summary ConjugateGradientsSolver::Solve(
    LinearOperator* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x,
    const TrustRegionMinimizer* minimizer) {
  CHECK(A != nullptr);
  CHECK(x != nullptr);
  CHECK(b != nullptr);
  CHECK_EQ(A->num_rows(), A->num_cols());

  LinearSolver::Summary summary;
  summary.termination_type = LINEAR_SOLVER_NO_CONVERGENCE;
  summary.message = "Maximum number of iterations reached.";
  summary.num_iterations = 0;
  summary.flops = 0;

  const int num_cols = A->num_cols();
  VectorRef xref(x, num_cols);
  Vector b_;
  Eigen::MatrixXd ns;
  if(FLAGS_random_rhs) {
    ns = nullspace(minimizer->x_);
    b_ = compatible_random_rhs(ns);
  }
  ConstVectorRef bref(FLAGS_random_rhs ? b_.data() : b, num_cols);

  const double norm_b = bref.norm();
  if (norm_b == 0.0) {
    xref.setZero();
    summary.termination_type = LINEAR_SOLVER_SUCCESS;
    summary.message = "Convergence. |b| = 0.";
    return summary;
  }

  Vector r(num_cols);
  Vector p(num_cols);
  Vector z(num_cols);
  Vector tmp(num_cols);

  bool use_rtol = FLAGS_rtol > 0;
  const double tol_r = (use_rtol ? FLAGS_rtol : per_solve_options.r_tolerance) * norm_b;
  Vector r_prev;
  if(FLAGS_fcg) {
    r_prev = Vector(num_cols);
  }


  tmp.setZero();
  A->RightMultiply(x, tmp.data());
  summary.flops += A->num_nonzeros();
  r = bref - tmp;
  if(FLAGS_random_rhs) {
    orthogonalize(ns, VectorRef(r.data(), r.size()));
  }
  double norm_r = r.norm();
  if (options_.min_num_iterations == 0 && norm_r <= tol_r) {
    summary.termination_type = LINEAR_SOLVER_SUCCESS;
    summary.message =
        StringPrintf("Convergence. |r| = %e <= %e.", norm_r, tol_r);
    return summary;
  }

  double rho = 1.0;

  // Initial value of the quadratic model Q = x'Ax - 2 * b'x.
  double Q0 = -1.0 * xref.dot(bref + r);

  for (summary.num_iterations = 1;; ++summary.num_iterations) {
    // Apply preconditioner
    if (per_solve_options.preconditioner != NULL) {
      z.setZero();
      per_solve_options.preconditioner->RightMultiply(r.data(), z.data());
      summary.flops += per_solve_options.preconditioner->num_nonzeros();
    } else {
      z = r;
    }
    if(FLAGS_random_rhs) {
      orthogonalize(ns, VectorRef(z.data(), z.size()));
    }

    double last_rho = rho;
    if(FLAGS_fcg) {
      rho = z.dot(r - r_prev);
    } else {
      rho = r.dot(z);
    }
    if (IsZeroOrInfinity(rho)) {
      summary.termination_type = LINEAR_SOLVER_FAILURE;
      summary.message = StringPrintf("Numerical failure. rho = r'z = %e.", rho);
      break;
    }

    if (summary.num_iterations == 1) {
      p = z;
    } else {
      double beta = rho / last_rho;
      if (IsZeroOrInfinity(beta)) {
        summary.termination_type = LINEAR_SOLVER_FAILURE;
        summary.message = StringPrintf(
            "Numerical failure. beta = rho_n / rho_{n-1} = %e, "
            "rho_n = %e, rho_{n-1} = %e", beta, rho, last_rho);
        break;
      }
      p = z + beta * p;
    }

    Vector& q = z;
    q.setZero();
    A->RightMultiply(p.data(), q.data());
    summary.flops += A->num_nonzeros();
    const double pq = p.dot(q);
    if ((pq <= 0) || std::isinf(pq)) {
      summary.termination_type = LINEAR_SOLVER_NO_CONVERGENCE;
      summary.message = StringPrintf(
          "Matrix is indefinite, no more progress can be made. "
          "p'q = %e. |p| = %e, |q| = %e",
          pq, p.norm(), q.norm());
      break;
    }

    const double alpha = rho / pq;
    if (std::isinf(alpha)) {
      summary.termination_type = LINEAR_SOLVER_FAILURE;
      summary.message =
          StringPrintf("Numerical failure. alpha = rho / pq = %e, "
                       "rho = %e, pq = %e.", alpha, rho, pq);
      break;
    }

    xref = xref + alpha * p;
    if(FLAGS_random_rhs) {
      orthogonalize(ns, xref);
    }

    if(FLAGS_fcg) {
      r_prev = r;
    }
    // Ideally we would just use the update r = r - alpha*q to keep
    // track of the residual vector. However this estimate tends to
    // drift over time due to round off errors. Thus every
    // residual_reset_period iterations, we calculate the residual as
    // r = b - Ax. We do not do this every iteration because this
    // requires an additional matrix vector multiply which would
    // double the complexity of the CG algorithm.
    if (summary.num_iterations % options_.residual_reset_period == 0) {
      tmp.setZero();
      A->RightMultiply(x, tmp.data());
      r = bref - tmp;
      summary.flops += A->num_nonzeros();
    } else {
      r = r - alpha * q;
    }

    // Quadratic model based termination.
    //   Q1 = x'Ax - 2 * b' x.
    const double Q1 = -1.0 * xref.dot(bref + r);

    // For PSD matrices A, let
    //
    //   Q(x) = x'Ax - 2b'x
    //
    // be the cost of the quadratic function defined by A and b. Then,
    // the solver terminates at iteration i if
    //
    //   i * (Q(x_i) - Q(x_i-1)) / Q(x_i) < q_tolerance.
    //
    // This termination criterion is more useful when using CG to
    // solve the Newton step. This particular convergence test comes
    // from Stephen Nash's work on truncated Newton
    // methods. References:
    //
    //   1. Stephen G. Nash & Ariela Sofer, Assessing A Search
    //   Direction Within A Truncated Newton Method, Operation
    //   Research Letters 9(1990) 219-221.
    //
    //   2. Stephen G. Nash, A Survey of Truncated Newton Methods,
    //   Journal of Computational and Applied Mathematics,
    //   124(1-2), 45-59, 2000.
    //
    const double zeta = summary.num_iterations * (Q1 - Q0) / Q1;
    if (zeta < per_solve_options.q_tolerance && !use_rtol &&
        summary.num_iterations >= options_.min_num_iterations) {
      summary.termination_type = LINEAR_SOLVER_SUCCESS;
      summary.message =
          StringPrintf("Iteration: %d Convergence: zeta = %e < %e. |r| = %e",
                       summary.num_iterations,
                       zeta,
                       per_solve_options.q_tolerance,
                       r.norm());
      break;
    }
    Q0 = Q1;

    // Residual based termination.
    norm_r = r.norm();
    if (norm_r <= tol_r &&
        summary.num_iterations >= options_.min_num_iterations) {
      summary.termination_type = LINEAR_SOLVER_SUCCESS;
      summary.message =
          StringPrintf("Iteration: %d Convergence. |r| = %e <= %e.",
                       summary.num_iterations,
                       norm_r,
                       tol_r);
      break;
    }

    if (summary.num_iterations >= options_.max_num_iterations) {
      break;
    }
  }

  return summary;
}

}  // namespace internal
}  // namespace ceres
