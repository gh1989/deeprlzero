#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <fstream>
#include "config.h"
#include "games/chess.h"
#include "network.h"
#include "mcts.h"
#include "trainer.h"
#include "logger.h"

using namespace deeprlzero;

// Helper function to visualize MCTS tree for chess positions
void PrintMCTSTree(Node* node, const Chess* game, int depth = 0, int max_depth = 2);

// Generate training positions from different sources
GamePositions GenerateOpeningPositions();
GamePositions GenerateEndgamePositions();  
GamePositions GenerateMiddlegamePositions();
GamePositions GenerateTacticalPositions();
GamePositions LoadPositionsFromPGN(const std::string& pgn_file);
GamePositions LoadPositionsFromFEN(const std::string& fen_file);

// Evaluation functions specific to chess
void EvaluateOnTestSuite(std::shared_ptr<NeuralNetwork> network, const Config& config);
void EvaluateOnTacticalPuzzles(std::shared_ptr<NeuralNetwork> network, const Config& config);

// Training functions
void runOpeningTraining(std::shared_ptr<NeuralNetwork> network, Config config);
void runEndgameTraining(std::shared_ptr<NeuralNetwork> network, Config config);
void runTacticalTraining(std::shared_ptr<NeuralNetwork> network, Config config);
void runMixedTraining(std::shared_ptr<NeuralNetwork> network, Config config);

void TestBasicChessNetwork(std::shared_ptr<NeuralNetwork> network) {
  try {
    Chess chess;
    torch::Tensor board = chess.GetCanonicalBoard();
    torch::Tensor batch = board.unsqueeze(0);
    auto outputs = network->forward(batch);
  } catch (const std::exception& e) {
    std::cerr << "Error in basic test: " << e.what() << std::endl;
    throw;
  }
}

// Main training orchestrator
void runComprehensiveChessTraining(std::shared_ptr<NeuralNetwork> network, Config config) {
  TestBasicChessNetwork(network);
  runTacticalTraining(network, config);
}

int main(int argc, char** argv) {
  Config config = Config::ParseCommandLine(argc, argv);
  std::shared_ptr<NeuralNetwork> network = std::make_shared<NeuralNetwork>(config);
  
  // Train on curated chess positions
  runComprehensiveChessTraining(network, config);
  
  return 0;
}

GamePositions GenerateTacticalPositions() {
  std::cout << "Loading hardcoded tactical positions..." << std::endl;
  
  GamePositions positions;
  
  // Collection of tactical positions with their solutions
  struct TacticalPuzzle {
    std::string fen;
    std::string description;
    std::vector<std::string> solution_moves;  // In algebraic notation
    float value;  // Expected outcome after solution
  };
  
  std::vector<TacticalPuzzle> puzzles = {
    {"6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "Back rank mate", {"e1e8"}, 1.0f},
    {"rnbqkb1r/ppp2ppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 1", "Queen fork", {"d1d5"}, 1.0f},
    {"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1", "Knight fork", {"f3e5"}, 1.0f},
    {"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", "Pin the knight", {"c4b5"}, 1.0f},
    {"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1", "Discovered attack", {"f3d4"}, 1.0f},
    {"6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1", "Queen mate", {"e1e8"}, 1.0f},
    {"r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP1NPPP/R1BQK2R w KQ - 0 1", "Deflection", {"c4f7"}, 1.0f},
    {"rnbqkb1r/ppp2ppp/3p1n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1", "Double attack", {"f3g5"}, 1.0f},
    {"6k1/5ppp/8/8/8/8/5PPP/4R1K1 b - - 0 1", "Back rank skewer", {"e8e1"}, 1.0f},
    {"6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1", "Smothered mate threat", {"f1f8"}, 1.0f}
  };
  
  for (const auto& puzzle : puzzles) {
    try {
      // Create chess position from FEN
      Chess chess;
      chess.SetFromString(puzzle.fen);
      
      // Get the board tensor
      torch::Tensor board = chess.GetCanonicalBoard();
      
      // Create policy vector (all zeros initially)
      std::vector<float> policy(chess.GetActionSize(), 0.0f);
      
      // Set solution moves to have probability 1.0
      float prob_per_move = 1.0f / puzzle.solution_moves.size();
      for (const std::string& move_str : puzzle.solution_moves) {
        Move move = MoveFromUci(move_str);
        int move_index = chess.MoveToIndex(move);
        if (move_index >= 0 && move_index < policy.size()) {
          policy[move_index] = prob_per_move;
        } else {
          std::cerr << "Warning: Invalid UCI move " << move_str 
                    << " in position " << puzzle.fen << std::endl;
        }
      }
      
      // Add to training data
      positions.boards.push_back(board);
      positions.policies.push_back(policy);
      positions.values.push_back(puzzle.value);
      
      std::cout << "Loaded: " << puzzle.description << std::endl;
      
    } catch (const std::exception& e) {
      std::cerr << "Error loading position " << puzzle.fen << ": " << e.what() << std::endl;
    }
  }
  
  std::cout << "Loaded " << positions.boards.size() << " tactical positions" << std::endl;
  return positions;
}

GamePositions GenerateOpeningPositions() {
  // Load master games database
  // Policy = frequency of moves played by 2400+ players
  return GamePositions();
}

std::vector<float> GetStockfishPolicy(const Chess& position) {
  // Run stockfish with MultiPV=5-10
  // Convert eval differences to policy distribution
  // e.g., softmax(stockfish_evals / temperature)
  return std::vector<float>();
}

GamePositions GenerateEndgamePositions() {
  // For 3-7 piece endings, use Syzygy tablebase perfect play
  // Policy = 1.0 for optimal moves, 0.0 for losing moves
  return GamePositions();
}

void runTacticalTraining(std::shared_ptr<NeuralNetwork> network, Config config)
{
  auto positions = GenerateTacticalPositions();
  auto optimizer = std::make_shared<torch::optim::Adam>(network->parameters(), config.learning_rate);
  Train(optimizer, network, config, positions);  
}