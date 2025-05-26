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
    std::string fen = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1";
    chess.SetFromString(fen);
    torch::Tensor board = chess.GetCanonicalBoard();
    torch::Tensor batch = board.unsqueeze(0);
    auto action_size = chess.GetActionSize();
    auto outputs = network->forward(batch);
    auto policy = outputs.first.squeeze().detach().cpu();
    auto value = outputs.second.item<float>();
    std::cout << "Policy: " << policy << std::endl;
    std::cout << "Value: " << value << std::endl;
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
  
  // Test with just one simple position first
  try {
    Chess chess;
    std::string fen = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1";
    chess.SetFromString(fen);
    
    torch::Tensor board = chess.GetCanonicalBoard();
    std::vector<float> policy(chess.GetActionSize(), 0.0f);
    
    // Just set uniform policy for now
    if (!policy.empty()) {
      policy[0] = 1.0f;  // Set first legal move
    }
    
    positions.boards.push_back(board.clone());  // Clone the tensor
    positions.policies.push_back(std::move(policy));
    positions.values.push_back(1.0f);
    
    std::cout << "Loaded 1 test position" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  
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
  //auto optimizer = std::make_shared<torch::optim::Adam>(network->parameters(), config.learning_rate);
  //Train(optimizer, network, config, positions);  
}