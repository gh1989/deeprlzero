#ifndef ALPHAZERO_GAME_H
#define ALPHAZERO_GAME_H

#include <vector>
#include <memory>
#include <torch/torch.h>

namespace alphazero {

class Game {
 public:
  virtual ~Game() = default;
  
  // Returns valid moves in current position (0-8 for TicTacToe)
  virtual std::vector<int> GetValidMoves() const = 0;
  
  // Makes a move and updates game state
  virtual void MakeMove(int move) = 0;
  
  // Returns 1 for win, -1 for loss, 0 for draw/ongoing
  virtual float GetGameResult() const = 0;
  
  // Returns true if game is over
  virtual bool IsTerminal() const = 0;
  
  // Returns current player (1 or -1)
  virtual int GetCurrentPlayer() const = 0;
  
  // Returns board representation for neural network
  virtual torch::Tensor GetCanonicalBoard() const = 0;
  
  // Creates a deep copy of current game state
  virtual std::unique_ptr<Game> Clone() const = 0;
  
  // Returns total number of possible actions
  virtual int GetActionSize() const = 0;
  
  // Returns number of input channels for neural network
  virtual int GetInputChannels() const = 0;
  
  // Add this new pure virtual method
  virtual void UndoMove(int move) = 0;
};

} // namespace alphazero

#endif // ALPHAZERO_GAME_H 