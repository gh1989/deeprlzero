#include <iostream>
#include <string>
#include <memory>
#include <random>
#include <torch/torch.h>

#include "games/tictactoe.h"
#include "mcts.h"
#include "network.h"
#include "config.h"

using namespace deeprlzero;

void printInstructions() {
  std::cout << "=== Tic-Tac-Toe vs AI ===" << std::endl;
  std::cout << "Enter moves as numbers 0-8:" << std::endl;
  std::cout << " 0 | 1 | 2 " << std::endl;
  std::cout << "---+---+---" << std::endl;
  std::cout << " 3 | 4 | 5 " << std::endl;
  std::cout << "---+---+---" << std::endl;
  std::cout << " 6 | 7 | 8 " << std::endl << std::endl;
}

int main(int argc, char* argv[]) {
  // Parse command line arguments
  std::string model_path = "comprehensive_model.pt";
  bool human_first = true;
  int simulation_count = 800;
  
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--ai-first") {
      human_first = false;
    } else if (arg == "--simulations" && i + 1 < argc) {
      simulation_count = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  --model PATH        Path to model file (default: comprehensive_model.pt)" << std::endl;
      std::cout << "  --ai-first          Let AI play first" << std::endl;
      std::cout << "  --simulations N     Number of MCTS simulations (default: 800)" << std::endl;
      return 0;
    }
  }
  
  // Setup config with model path
  Config config;
  config.model_path = model_path;
  config.num_simulations = simulation_count;

  // Create and load the network using the proper method
  std::cout << "Loading model from " << config.model_path << std::endl;
  std::shared_ptr<NeuralNetwork> network;
  try {
    // Use the static method to load the best network
    network = LoadBestNetwork(config);
    if (!network) {
      throw std::runtime_error("Model loading failed - returned null pointer");
    }
    network->eval();
  } catch (const std::exception& e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
    return 1;
  }

  // Create MCTS with the loaded model and config
  MCTS mcts(network, config);
  
  printInstructions();
  
  // Game loop
  while (true) {
    TicTacToe game;
    bool human_turn = human_first;
    
    while (!IsTerminal(game)) {
      // Display current board
      std::cout << "\nCurrent board:" << std::endl;
      std::cout << ToString(game) << std::endl;
      
      if (human_turn) {
        // Human player's turn
        std::cout << "Your turn (X). Enter move (0-8): ";
        int move;
        while (true) {
          if (!(std::cin >> move)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Enter a number 0-8: ";
            continue;
          }
          
          auto valid_moves = GetValidMoves(game);
          bool is_valid = false;
          for (int valid_move : valid_moves) {
            if (move == valid_move) {
              is_valid = true;
              break;
            }
          }
          
          if (is_valid) {
            break;
          } else {
            std::cout << "Invalid move. Enter a number for an empty square: ";
          }
        }
        
        MakeMove(game, move);
      } else {
        // AI's turn
        std::cout << "AI thinking..." << std::endl;
        mcts.ResetRoot();
        
        for (int sim = 0; sim < config.num_simulations; ++sim) {
          mcts.Search(game, mcts.GetRoot());
        }
        
        int action = mcts.SelectMove(game, 0.0f); // Deterministic play
        std::cout << "AI chooses: " << action << std::endl;
        MakeMove(game, action);
      }
      
      // Switch turns
      human_turn = !human_turn;
    }
    
    // Display final board
    std::cout << "\nFinal board:" << std::endl;
    std::cout << ToString(game) << std::endl;
    
    // Display result
    float result = GetGameResult(game);
    if (result > 0) {
      std::cout << "Player X wins!" << std::endl;
    } else if (result < 0) {
      std::cout << "Player O wins!" << std::endl;
    } else {
      std::cout << "It's a draw!" << std::endl;
    }
    
    // Ask to play again
    std::cout << "\nPlay again? (y/n): ";
    char play_again;
    std::cin >> play_again;
    if (play_again != 'y' && play_again != 'Y') {
      break;
    }
  }
  
  return 0;
}
