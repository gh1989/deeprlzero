#ifndef ALPHAZERO_THREAD_H_
#define ALPHAZERO_THREAD_H_

#include <functional>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

namespace alphazero {

// Executes a parallel loop using OpenMP if available; otherwise, runs sequentially.
// @param num_iterations Number of iterations of the loop.
// @param func A lambda function taking an integer index representing the loop body.
inline void ParallelFor(int num_iterations, const std::function<void(int)>& func) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) proc_bind(spread)
  for (int i = 0; i < num_iterations; ++i) {
    func(i);
  }
#else
  for (int i = 0; i < num_iterations; ++i) {
    func(i);
  }
#endif  // _OPENMP
}

// Returns a thread-local instance of type T using the provided factory function.
// The instance is created on the first call in each thread.
// @param create A function that returns a pointer to a new T instance.
template <typename T, typename Factory>
T &GetThreadLocalInstance(Factory create) {
  thread_local std::unique_ptr<T> instance;
  if (!instance) {
    instance.reset(create());
  }
  return *instance;
}

}  // namespace alphazero

#endif  // ALPHAZERO_THREAD_H_