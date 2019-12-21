#include "ceres/multigrid.h"
#include "ceres/visibility.h"
#include "ceres/trust_region_minimizer.h"

namespace ceres {
namespace internal {

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

  MultigridPreconditioner::MultigridPreconditioner(const CompressedRowBlockStructure& bs,
      const Preconditioner::Options& options) : options_(options) {
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
    mg_ = jl_call3(create, wrap_array(colptr), wrap_array(rows), wrap_array(values));
    check_error();
    // create global reference to mg_ so that it is not freed by the julia GC
    jl_set_global(jl_main_module, jl_symbol("mg"), mg_);
  }

  bool MultigridPreconditioner::UpdateImpl(const CompressedRowSparseMatrix& A, const double* D, const TrustRegionMinimizer* minimizer) {
    auto update = get_function("update!", "bamg");
    jl_value_t* args[6] = { mg_
                          , wrap_array(A.rows(), A.num_rows()+1)
                          , wrap_array(A.cols(), A.num_nonzeros())
                          , wrap_array(A.values(), A.num_nonzeros())
                          , wrap_array(minimizer->jacobian_scaling_.data() + options_.elimination_groups[0]*options_.e_block_size, options_.elimination_groups[1]*options_.f_block_size)
                          , wrap_array(minimizer->x_.data() + options_.elimination_groups[0]*options_.e_block_size, options_.elimination_groups[1]*options_.f_block_size)
                          };
    jl_call(update, args, 6);
    check_error();
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
