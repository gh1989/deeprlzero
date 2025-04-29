#ifndef GAME_H
#define GAME_H

#include <array>
#include <vector>
#include <memory>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <random>
#include <mutex>
#include <atomic>

#include "config.h"
#include "network.h"
#include "logger.h"
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

class TicTacToe : public Game {
 public:
  static constexpr int kBoardSize = 3;
  static constexpr int kNumActions = 9;
  static constexpr int kNumChannels = 3;
  
  TicTacToe();
  
  virtual std::vector<int> GetValidMoves() const override;
  void MakeMove(int move) override;
  void Reset() override;
  float GetGameResult() const override;
  bool IsTerminal() const override;
  int GetCurrentPlayer() const override { return current_player_; }
  torch::Tensor GetCanonicalBoard() const override;
  std::unique_ptr<Game> Clone() const override;
  int GetActionSize() const override { return kNumActions; }
  int GetInputChannels() const override { return kNumChannels; }
  void UndoMove(int move) override;

  // Add visualization method
  std::string ToString() const {
    std::string result;
    for (int i = 0; i < kBoardSize; ++i) {
      for (int j = 0; j < kBoardSize; ++j) {
        switch (board_[i][j]) {
          case 1: result += " X "; break;
          case -1: result += " O "; break;
          case 0: result += " . "; break;
        }
      }
      result += "\n";
    }
    return result;
  }

 private:
  std::array<std::array<int, kBoardSize>, kBoardSize> board_;
  int current_player_;
  
  bool CheckWin(int player) const;
  bool IsBoardFull() const;
  
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

template <typename GameType>
class SelfPlay {
  static_assert(std::is_base_of<Game, GameType>::value, "GameType must derive from Game");
 public:
   SelfPlay(std::shared_ptr<NeuralNetwork> network, const Config& config, float temperature)
    : config_(config), current_temperature_(temperature) {
    network_ = std::dynamic_pointer_cast<NeuralNetwork>(network->NetworkClone(torch::kCPU));
    network_->to(torch::kCPU);
    network_->eval();
  }

  // Execute a single episode and return the positions
  GamePositions ExecuteEpisode();
  
  // Execute multiple episodes in parallel and merge the results
  GamePositions ExecuteEpisodesParallel();

 private:
  std::shared_ptr<NeuralNetwork> network_;
  const Config& config_;
  float current_temperature_;
  std::mt19937 rng_{std::random_device{}()};  
};

std::vector<GamePositions> AllEpisodes();

inline float CalculateAverageExplorationMetric(const std::vector<GamePositions>& episodes) {
  if (episodes.empty()) {
    return 0.0f;
  }
  
  float total_entropy = 0.0f;
  int total_moves = 0;
  
  for (const auto& episode : episodes) {
    for (const auto& policy : episode.policies) {
      total_entropy += NeuralNetwork::CalculatePolicyEntropy(policy);
      total_moves++;
    }
  }
  Logger& logger = Logger::GetInstance();
  float exploration_metric = total_entropy / total_moves; 
  logger.LogFormat("Exploration metric: {:.4f}", exploration_metric);
  return exploration_metric;
}

}

#endif