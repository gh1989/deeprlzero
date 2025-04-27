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

#include "core/config.h"
#include "core/network.h"

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

struct GameEpisode {
  std::vector<torch::Tensor> boards;           
  std::vector<std::vector<float>> policies;      
  float outcome;                                 
};

template <typename GameType>
class SelfPlay {
  static_assert(std::is_base_of<Game, GameType>::value, "GameType must derive from Game");
 public:
   SelfPlay(std::shared_ptr<NeuralNetwork> network, const Config& config, float temperature)
    : config_(config), current_temperature_(temperature) {
    network_ = std::dynamic_pointer_cast<NeuralNetwork>(network->clone(torch::kCPU));
    network_->to(torch::kCPU);
    network_->eval();
  }

GameEpisode ExecuteEpisode();
  std::vector<GameEpisode> ExecuteEpisodesParallel(); 

 private:
  std::shared_ptr<NeuralNetwork> network_;
  const Config& config_;
  float current_temperature_;
  std::mt19937 rng_{std::random_device{}()};  
};

std::vector<GameEpisode> AllEpisodes();

inline float CalculateAverageExplorationMetric(const std::vector<GameEpisode>& episodes) {
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
  
  return total_moves > 0 ? total_entropy / total_moves : 0.0f;
}

}

#endif