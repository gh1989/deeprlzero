#ifndef THREAD_H_
#define THREAD_H_

#include <functional>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace deeprlzero {

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
#endif
}

template <typename T, typename Factory>
T &GetThreadLocalInstance(Factory create) {
  thread_local std::unique_ptr<T> instance;
  if (!instance) {
    instance.reset(create());
  }
  return *instance;
}

}

#endif