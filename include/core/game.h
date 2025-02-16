#ifndef ALPHAZERO_GAME_H
#define ALPHAZERO_GAME_H

#include <vector>
#include <memory>
#include <torch/torch.h>

namespace alphazero {

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

} // namespace alphazero

#endif // ALPHAZERO_GAME_H 