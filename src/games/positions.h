#ifndef GAMES_GAME_H
#define GAMES_GAME_H

#include <array>
#include <vector>
#include <memory>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <random>
#include <mutex>
#include <atomic>
#include <future>
#include <stdexcept>
#include <thread>
#include <concepts>

#include "traits.h"

namespace deeprlzero {

struct GamePositions {
  std::vector<torch::Tensor> boards;
  std::vector<std::vector<float>> policies;
  std::vector<float> values;
  
  // Helper to get size
  size_t size() const {
    return boards.size();
  }
  
  // Helper to clear all arrays
  void clear() {
    boards.clear();
    policies.clear();
    values.clear();
  }
  
  // Helper to append another GamePositions
  void Append(const GamePositions& other) {
    boards.insert(boards.end(), other.boards.begin(), other.boards.end());
    policies.insert(policies.end(), other.policies.begin(), other.policies.end());
    values.insert(values.end(), other.values.begin(), other.values.end());
  }
};

}

#endif