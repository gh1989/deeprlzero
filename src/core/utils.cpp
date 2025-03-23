#include "core/utils.h"
#include "core/tictactoe.h"
#include "core/self_play.h"
#include <vector>
#include <memory>
#include <torch/torch.h>

namespace alphazero {

// Recursive helper function to generate all episodes.
static void GenerateAllEpisodesHelper(std::unique_ptr<TicTacToe> game,
                                      std::vector<torch::Tensor>& boards,
                                      std::vector<std::vector<float>>& policies,
                                      std::vector<float>& values,
                                      std::vector<int>& players,
                                      std::vector<GameEpisode>& episodes) {
  // If the game is over, record the episode.
  if (game->IsTerminal()) {
    float final_result = game->GetGameResult();
    GameEpisode episode;
    episode.boards = boards;
    episode.policies = policies;
    // Compute per-move values from the perspective of the move maker.
    episode.values.resize(players.size());
    for (size_t i = 0; i < players.size(); ++i) {
      episode.values[i] = (players[i] == 1) ? final_result : -final_result;
    }
    episode.outcome = final_result;
    episodes.push_back(episode);
    return;
  }

  // Get all valid moves in the current state.
  std::vector<int> valid_moves = game->GetValidMoves();
  for (int move : valid_moves) {
    // Clone the current game state.
    std::unique_ptr<Game> game_clone = game->Clone();
    // We know the game is TicTacToe.
    TicTacToe* ttt_ptr = static_cast<TicTacToe*>(game_clone.release());
    std::unique_ptr<TicTacToe> new_game(ttt_ptr);
    // Execute the move.
    new_game->MakeMove(move);
    // Record the new board state.
    torch::Tensor board_tensor = new_game->GetCanonicalBoard();
    boards.push_back(board_tensor);
    // Record a dummy uniform policy (each action gets equal probability).
    int action_size = new_game->GetActionSize();
    std::vector<float> uniform_policy(action_size, 1.0f / action_size);
    policies.push_back(uniform_policy);
    // Record a placeholder value.
    values.push_back(0.0f);
    // Record the player who actually made the move.
    // (After MakeMove the current player has been switched, so the move maker is the opposite.)
    int move_player = -new_game->GetCurrentPlayer();
    players.push_back(move_player);

    // Recurse into the new game state.
    GenerateAllEpisodesHelper(std::move(new_game), boards, policies, values, players, episodes);

    // Backtrack.
    boards.pop_back();
    policies.pop_back();
    values.pop_back();
    players.pop_back();
  }
}

std::vector<GameEpisode> AllEpisodes() {
  std::vector<GameEpisode> episodes;
  std::vector<torch::Tensor> boards;
  std::vector<std::vector<float>> policies;
  std::vector<float> values;
  std::vector<int> players;
  
  // Start with a fresh TicTacToe game.
  auto game = std::make_unique<TicTacToe>();
  // Begin recursion. (No board state is recorded initially; each branch
  // records the board state after the first move.)
  GenerateAllEpisodesHelper(std::move(game), boards, policies, values, players, episodes);
  return episodes;
}

}  // namespace alphazero