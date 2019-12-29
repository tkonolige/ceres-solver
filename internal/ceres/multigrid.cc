#include <gflags/gflags.h>

#include "ceres/multigrid.h"
#include "ceres/visibility.h"
#include "ceres/trust_region_minimizer.h"
#include "ceres/detect_structure.h"
#include "ceres/block_random_access_sparse_matrix.h"

#include <iostream>

JULIA_DEFINE_FAST_TLS()

DEFINE_string(options_file, "", "File to read multigrid options from");

using namespace std;

namespace ceres {
namespace internal {

  // You'd think there would be a way to pass a function pointer from C++ to
  // Julia so that we could do callbacks, but there are 2 things stopping us:
  //   1. Its technically incorrect to cast a function pointer to a regular
  //   pointer because some architectures might have different sizes for pointers.
  //   2. You can't get pointers to non-static member functions in C++.
  // So we'll just do the really unsafe thing of using a global.
  // XXX: Clearly this is hacky and not thread safe.
  static void * _complement_XXX;
  void right_multiply(const double* a, double* b) {
    ((ImplicitSchurComplement*)_complement_XXX)->RightMultiply(a, b);
  }

  void check_error() {
    if (jl_exception_occurred()) {
      const char *p = (const char *)jl_unbox_voidpointer(jl_eval_string("pointer(sprint(showerror, ccall(:jl_exception_occurred, Any, ())))"));

      throw std::runtime_error(p);
    }
  }

  jl_value_t* eval_string(const std::string& str) {
    auto ret = jl_eval_string(str.data());
    check_error();
    return ret;
  }

  jl_function_t* get_function(const std::string& fnname, const std::string& module) {
    auto mod = (jl_module_t*)jl_get_global(jl_main_module, jl_symbol(module.c_str()));
    if(mod == NULL) {
      throw("Could not get module " + module);
    }
    check_error();
    jl_function_t* func = jl_get_function(mod, fnname.c_str());
    if(func == NULL) {
      throw("Could not get function " + fnname + " from module " + module);
    }
    return func;
  }

  jl_value_t* wrap_array(std::vector<double>& vec) {
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    return (jl_value_t*)jl_ptr_to_array_1d(array_type, vec.data(), vec.size(), 0);
  }

  jl_value_t* wrap_array(std::vector<int32_t>& vec) {
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_int32_type, 1);
    return (jl_value_t*)jl_ptr_to_array_1d(array_type, vec.data(), vec.size(), 0);
  }

  jl_value_t* wrap_array(const int32_t* vec, int len) {
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_int32_type, 1);
    return (jl_value_t*)jl_ptr_to_array_1d(array_type, (void*)vec, len, 0);
  }

  jl_value_t* wrap_array(const double* vec, int len) {
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    return (jl_value_t*)jl_ptr_to_array_1d(array_type, (void*)vec, len, 0);
  }

  void MultigridPreconditioner::InitStorage(const CompressedRowBlockStructure* bs) {
    const int num_eliminate_blocks = options_.elimination_groups[0];
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

    lhs_.reset(new BlockRandomAccessSparseMatrix(blocks_, block_pairs));
    dummy_rhs.reset(new double[lhs_->num_rows()]);

    size_t num_rows = 0;
    for(auto& row: bs->rows) {
      num_rows += row.block.size;
    }
    dummy_b.reset(new double[num_rows]);
  }

  MultigridPreconditioner::MultigridPreconditioner(const CompressedRowBlockStructure& bs,
      const Preconditioner::Options& options,
      ImplicitSchurComplement* complement) :
    options_(options), schur_complement_(complement), use_implicit_(complement != NULL) {
    // set up eliminator for explicit schur complement
    if (use_implicit_ && eliminator_.get() == NULL) {
      const int num_eliminate_blocks = options_.elimination_groups[0];
      const int num_f_blocks = bs.cols.size() - num_eliminate_blocks;

      DetectStructure(bs,
          num_eliminate_blocks,
          &options_.row_block_size,
          &options_.e_block_size,
          &options_.f_block_size);
      InitStorage(&bs);

      LinearSolver::Options opts;
      opts.elimination_groups = options_.elimination_groups;
      opts.row_block_size = options_.row_block_size;
      opts.e_block_size = options_.e_block_size;
      opts.f_block_size = options_.f_block_size;
      opts.context = options_.context;
      eliminator_.reset(SchurEliminatorBase::Create(opts));

      CHECK(eliminator_);
      const bool kFullRankETE = true;
      eliminator_->Init(num_eliminate_blocks, kFullRankETE, &bs);
    }

    std::vector<std::set<int>> visibility;
    ComputeVisibility(bs, options.elimination_groups[0], &visibility);
    auto graph = CreateSchurComplementGraph(visibility);
    num_rows_ = options.f_block_size * graph->vertices().size();

    // convert graph to one-indexed csc
    std::vector<int> colptr;
    colptr.push_back(1);
    std::vector<int> rows;
    std::vector<double> values;
    for(int i = 0; i < graph->vertices().size(); i++) {
      for(auto v : graph->Neighbors(i)) {
        rows.push_back(v + 1);
        values.push_back(graph->EdgeWeight(i, v));
      }
      colptr.push_back(rows.size() + 1);
    }

    // TODO: make sure this is done only once
    jl_init();
    eval_string("import bamg");
    eval_string("import LinearAlgebra");
    auto create = get_function("create_multigrid_ceres", "bamg");
    jl_value_t* args[4] = { wrap_array(colptr)
      , wrap_array(rows)
        , wrap_array(values)
        , jl_cstr_to_string(FLAGS_options_file.data())
    };
    mg_ = jl_call(create, args, 4);
    check_error();
    // create global reference to mg_ so that it is not freed by the julia GC
    jl_set_global(jl_main_module, jl_symbol("mg"), mg_);
  }

  bool MultigridPreconditioner::UpdateExplicit(const CompressedRowSparseMatrix& A, const double* D, const TrustRegionMinimizer* minimizer) {
    auto update = get_function("update!", "bamg");
    jl_value_t* args[8] = { mg_
                          , wrap_array(A.rows(), A.num_rows()+1)
                          , wrap_array(A.cols(), A.num_nonzeros())
                          , wrap_array(A.values(), A.num_nonzeros())
                          , wrap_array(minimizer->jacobian_scaling_.data() + options_.elimination_groups[0]*options_.e_block_size, options_.elimination_groups[1]*options_.f_block_size)
                          , wrap_array(minimizer->x_.data() + options_.elimination_groups[0]*options_.e_block_size, options_.elimination_groups[1]*options_.f_block_size)
                          , jl_box_voidpointer(NULL)
                          , jl_box_int64(-1)
    };
    jl_call(update, args, 8);
    check_error();
    return true;
  }

  bool MultigridPreconditioner::UpdateImpl(const BlockSparseMatrix& A, const double* D, const TrustRegionMinimizer* minimizer) {
    eliminator_->Eliminate(BlockSparseMatrixData(A), dummy_b.get(), D, lhs_.get(), dummy_rhs.get());

    // TODO: this seems like its going to allocate a lot
    BlockRandomAccessSparseMatrix* sc = down_cast<BlockRandomAccessSparseMatrix*>(
        const_cast<BlockRandomAccessMatrix*>(lhs_.get()));
    const TripletSparseMatrix* tsm = sc->matrix();

    std::unique_ptr<CompressedRowSparseMatrix> l;
    l.reset(CompressedRowSparseMatrix::FromTripletSparseMatrix(*tsm));
    l->set_storage_type(CompressedRowSparseMatrix::UNSYMMETRIC);

    // XXX: we need this for the hacky way we pass callbacks to Julia. See the comment for right_multiply
    _complement_XXX = schur_complement_;

    auto update = get_function("update!", "bamg");
    auto rm = jl_box_voidpointer(reinterpret_cast<void*>(right_multiply));
    JL_GC_PUSH1(&rm);
    {
      int64_t nnz_ = schur_complement_->num_nonzeros();
      auto nnz = jl_box_int64(nnz_);
      JL_GC_PUSH1(&nnz);
      jl_value_t* args[8] = { mg_
                            , wrap_array(l->rows(), l->num_rows()+1)
                            , wrap_array(l->cols(), l->num_nonzeros())
                            , wrap_array(l->values(), l->num_nonzeros())
                            , wrap_array(minimizer->jacobian_scaling_.data() + options_.elimination_groups[0]*options_.e_block_size, options_.elimination_groups[1]*options_.f_block_size)
                            , wrap_array(minimizer->x_.data() + options_.elimination_groups[0]*options_.e_block_size, options_.elimination_groups[1]*options_.f_block_size)
                            , rm
                            , nnz
      };
      jl_call(update, args, 8);
      check_error();
      JL_GC_POP();
    }
    JL_GC_POP();
    return true;
  }

  void MultigridPreconditioner::RightMultiply(const double* x, double* y) const {
    auto lmul = get_function("ldiv!", "LinearAlgebra");
    jl_call3(lmul, wrap_array(y, num_rows_), mg_, wrap_array(x, num_rows_));
    check_error();
  }

  int MultigridPreconditioner::num_rows() const {
    return num_rows_;
  }

  int64_t MultigridPreconditioner::num_nonzeros() const {
    auto wrk = get_function("flops", "bamg");
    auto w = jl_call1(wrk, mg_);
    check_error();
    if (jl_typeis(w, jl_int64_type)) {
      return jl_unbox_int64(w);
    } else {
      throw(std::runtime_error("Could not calculate nnz"));
    }
  }

}
}
