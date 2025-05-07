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

#include "../config.h"
#include "../network.h"
#include "../logger.h"




namespace deeprlzero {

class Game {
 public:
  virtual ~Game() = default;
  virtual std::vector<int> GetValidMoves() const = 0;
  virtual void MakeMove(int move) = 0;
  virtual float GetGameResult() const = 0;
  virtual bool IsTerminal() const = 0;
  virtual int GetCurrentPlayer() const = 0;
  virtual torch::Tensor GetCanonicalBoard() const = 0;
  virtual std::unique_ptr<Game> Clone() const = 0;
  virtual int GetActionSize() const = 0;
  virtual int GetInputChannels() const = 0;
  virtual void UndoMove(int move) = 0;
  virtual void Reset() = 0;
};

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