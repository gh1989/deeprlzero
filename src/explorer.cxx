#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include "config.h"
#include "game.h"
#include "network.h"
#include "network.h"
#include "mcts.h"
#include "trainer.h"
#include "logger.h"

using namespace deeprlzero;

// Helper function to visualize MCTS tree at important nodes
void PrintMCTSTree(Node* node, const Game* game, int depth = 0, int max_depth = 2) {
  if (!node || depth > max_depth) return;
  
  std::string indent(depth * 2, ' ');
  std::cout << indent << "Node";
  if (node->action >= 0) {
    int row = node->action / TicTacToe::kBoardSize;
    int col = node->action % TicTacToe::kBoardSize; 
    std::cout << " (" << row+1 << "," << col+1 << ")";
  } else {
    std::cout << " (root)";
  }
  
  std::cout << " Visits: " << node->visit_count 
            << " Value: " << std::fixed << std::setprecision(3) << node->GetValue()
            << " Prior: " << std::fixed << std::setprecision(3) << node->prior
            << std::endl;
  
  // Only show children for nodes with enough visits
  if (node->visit_count > 0 && depth < max_depth) {
    // Find all children with visits
    std::vector<std::pair<int, Node*>> visited_children;
    for (size_t i = 0; i < node->children.size(); i++) {
      if (node->children[i] && node->children[i]->visit_count > 0) {
        visited_children.push_back({i, node->children[i].get()});
      }
    }
    
    // Sort by visit count
    std::sort(visited_children.begin(), visited_children.end(),
              [](const auto& a, const auto& b) {
                return a.second->visit_count > b.second->visit_count;
              });
    
    // Print top visited children
    int shown = 0;
    for (const auto& [action, child] : visited_children) {
      if (shown++ >= 5) {
        std::cout << indent << "  ... " << (visited_children.size() - 5) 
                  << " more children" << std::endl;
        break;
      }
      
      int row = action / TicTacToe::kBoardSize;
      int col = action % TicTacToe::kBoardSize;
      std::cout << indent << "  ├─ ";
      std::cout << "(" << row+1 << "," << col+1 << ")";
      std::cout << " Visits: " << child->visit_count 
                << " Value: " << std::fixed << std::setprecision(3) << child->GetValue() 
                << std::endl;
                
      PrintMCTSTree(child, game, depth + 1, max_depth);
    }
  }
}

GamePositions GenerateAllTicTacToeGames() {
  std::cout << "Generating all tic-tac-toe positions..." << std::endl;
  
  GamePositions all_positions;
  std::set<std::string> visited_positions;
  
  // Start with an empty game
  auto game = std::make_shared<TicTacToe>();
  
  // Use BFS to systematically explore all positions
  std::queue<std::shared_ptr<TicTacToe>> position_queue;
  position_queue.push(game);
  
  while (!position_queue.empty()) {
    auto current_game = position_queue.front();
    position_queue.pop();
    
    // Create a string representation to track visited positions
    // We can use the canonical board tensor instead of direct board access
    auto canonical_board = current_game->GetCanonicalBoard();
    auto board_accessor = canonical_board.accessor<float, 3>();
    
    // First channel is our pieces, second is opponent's
    std::string board_key = "";
    for (int i = 0; i < TicTacToe::kBoardSize; i++) {
      for (int j = 0; j < TicTacToe::kBoardSize; j++) {
        // Encode our pieces as '1', opponent as '2', empty as '0'
        if (board_accessor[0][i][j] > 0.5f) {
          board_key += "1";
        } else if (board_accessor[1][i][j] > 0.5f) {
          board_key += "2";
        } else {
          board_key += "0";
        }
      }
    }
    // Add current player to the key
    board_key += std::to_string(current_game->GetCurrentPlayer());
    
    // Skip if we've already seen this position
    if (visited_positions.find(board_key) != visited_positions.end()) {
      continue;
    }
    visited_positions.insert(board_key);
    
    // Add the current position to our collection
    auto valid_moves = current_game->GetValidMoves();
    
    // Only add non-terminal positions
    if (!valid_moves.empty()) {
      // Create a uniform policy over valid moves for now
      std::vector<float> policy(TicTacToe::kNumActions, 0.0f);
      float prob = 1.0f / valid_moves.size();
      for (int move : valid_moves) {
        policy[move] = prob;
      }
      
      all_positions.boards.push_back(current_game->GetCanonicalBoard());
      all_positions.policies.push_back(policy);
      
      // For now, use 0 as the value (can be updated with minimax later)
      all_positions.values.push_back(0.0f);
      
      // Explore all valid next positions
      for (int move : valid_moves) {
        auto next_game = std::make_shared<TicTacToe>(*current_game);
        next_game->MakeMove(move);
        position_queue.push(next_game);
      }
    }
  }
  
  std::cout << "Generated " << all_positions.boards.size() << " unique positions" << std::endl;
  return all_positions;
}

void runComprehensiveTraining(std::shared_ptr<NeuralNetwork> network, Config config) {
  std::cout << "\n==== Comprehensive Tic-tac-toe Training ====" << std::endl;
  std::cout << "Generating all possible valid games..." << std::endl;
  
  // Generate all possible games
  GamePositions all_episodes = GenerateAllTicTacToeGames();
  
  // Select CUDA device
  torch::Device device(torch::kCUDA, 0);
  
  // Move network to GPU
  network->to(device);
  
  // Prepare training data
  std::vector<torch::Tensor> all_boards;
  std::vector<torch::Tensor> all_policies;
  std::vector<float> all_values;
  
  for (const auto& episode : all_episodes.boards) {
    all_boards.push_back(episode);
    
    // Create policy tensor
    all_policies.push_back(torch::from_blob(
        const_cast<float*>(all_episodes.policies[&episode - &all_episodes.boards[0]].data()),
        {TicTacToe::kNumActions}, 
        torch::kFloat32).clone());
    
    // Add value targets (adjusted for player perspective)
    float perspective = (all_episodes.values[&episode - &all_episodes.boards[0]] == 1.0f) ? 1.0f : -1.0f;
    all_values.push_back(all_episodes.values[&episode - &all_episodes.boards[0]] * perspective);
  }
  
  // Stack tensors into batches and move to GPU
  torch::Tensor states_batch = torch::stack(all_boards).clone().to(device).set_requires_grad(true);
  torch::Tensor policy_batch = torch::stack(all_policies).clone().to(device).set_requires_grad(true);
  torch::Tensor values = torch::tensor(all_values).reshape({-1, 1}).clone().to(device).set_requires_grad(true);
  
  std::cout << "Prepared " << states_batch.sizes()[0] << " training positions" << std::endl;
  std::cout << "Training on GPU: " << (device.is_cuda() ? "Yes" : "No") << std::endl;
  
  // Check network prediction on empty board before training
  network->eval();
  {
    torch::NoGradGuard no_grad;
    TicTacToe empty_game;
    torch::Tensor empty_state = empty_game.GetCanonicalBoard().to(device);
    auto [initial_policy, initial_value] = network->forward(empty_state.unsqueeze(0));
    
    std::cout << "\nBefore comprehensive training:" << std::endl;
    std::cout << "Policy output for empty board:" << std::endl;
    torch::Tensor softmax_policy = torch::softmax(initial_policy, 1);
    for (int i = 0; i < 9; i++) {
      std::cout << softmax_policy[0][i].item<float>() << " ";
      if (i % 3 == 2) std::cout << std::endl;
    }
    std::cout << "Value estimate: " << initial_value.item<float>() << std::endl;
  }
  
  // Train the network
  network->train();
  
  // Configure optimizer
  torch::optim::Adam optimizer(
      network->parameters(),
      torch::optim::AdamOptions(0.01).weight_decay(0.0001));
  
  // Train for multiple epochs
  int num_epochs = config.num_epochs;
  std::cout << "\nTraining for " << num_epochs << " epochs on all possible positions..." << std::endl;
  
  int batch_size = config.training_batch_size;
  int num_batches = (states_batch.sizes()[0] + batch_size - 1) / batch_size;
  float gamma_alpha = config.gamma_alpha;
  float gamma_beta = 1-gamma_alpha;

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    float epoch_policy_loss = 0.0f;
    float epoch_value_loss = 0.0f;
    float epoch_total_loss = 0.0f;
    auto indices = torch::randperm(states_batch.sizes()[0], device);
    
    for (int batch = 0; batch < num_batches; batch++) {
      optimizer.zero_grad();
      
      // Get mini-batch
      int start_idx = batch * batch_size;
      int end_idx = std::min(start_idx + batch_size, static_cast<int>(states_batch.sizes()[0]));
      auto batch_indices = indices.slice(0, start_idx, end_idx);
    
      auto batch_states = states_batch.index_select(0, batch_indices);
      auto batch_policies = policy_batch.index_select(0, batch_indices);
      auto batch_values = values.index_select(0, batch_indices);
      auto [policy_output, value_output] = network->forward(batch_states);
      torch::Tensor log_softmax_policy = torch::log_softmax(policy_output, 1);
      torch::Tensor policy_loss = -torch::sum(batch_policies * log_softmax_policy) / batch_policies.size(0);
      torch::Tensor value_loss = torch::mse_loss(value_output, batch_values);
      torch::Tensor total_loss = gamma_alpha * policy_loss + gamma_beta * value_loss;

      total_loss.backward();
      optimizer.step();
      
      epoch_policy_loss += policy_loss.item<float>();
      epoch_value_loss += value_loss.item<float>();
      epoch_total_loss += total_loss.item<float>();
    }
    
    // Print epoch stats
    if (epoch % 2 == 0 || epoch == num_epochs - 1) {
      std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                << ", Policy Loss: " << (epoch_policy_loss / num_batches)
                << ", Value Loss: " << (epoch_value_loss / num_batches)
                << ", Total Loss: " << (epoch_total_loss / num_batches) << std::endl;
    }
  }
  
  // Evaluate network after training
  network->eval();
  {
    torch::NoGradGuard no_grad;
    TicTacToe empty_game;
    torch::Tensor empty_state = empty_game.GetCanonicalBoard().to(device);
    auto [final_policy, final_value] = network->forward(empty_state.unsqueeze(0));
    
    std::cout << "\nAfter comprehensive training:" << std::endl;
    std::cout << "Policy output for empty board (after softmax):" << std::endl;
    torch::Tensor softmax_policy = torch::softmax(final_policy, 1);
    for (int i = 0; i < 9; i++) {
      std::cout << softmax_policy[0][i].item<float>() << " ";
      if (i % 3 == 2) std::cout << std::endl;
    }
    std::cout << "Middle square preference: " << softmax_policy[0][4].item<float>() * 100.0f << "%" << std::endl;
    std::cout << "Value estimate: " << final_value.item<float>() << std::endl;
    
    // Test a few key positions
    std::cout << "\nTesting key positions:" << std::endl;
    
    // Test center position
    TicTacToe center_game;
    center_game.MakeMove(4);  // X in center
    torch::Tensor center_state = center_game.GetCanonicalBoard().to(device);
    auto [center_policy, center_value] = network->forward(center_state.unsqueeze(0));
    torch::Tensor center_softmax = torch::softmax(center_policy, 1);
    
    std::cout << "Position with X in center:" << std::endl;
    std::cout << center_game.ToString();
    std::cout << "Value estimate: " << center_value.item<float>() << std::endl;
    
    // Show top move recommendations
    std::vector<std::pair<int, float>> move_probs;
    for (int i = 0; i < 9; i++) {
      if (i != 4) {  // Skip the center which is already occupied
        move_probs.push_back({i, center_softmax[0][i].item<float>()});
      }
    }
    
    // Sort by probability
    std::sort(move_probs.begin(), move_probs.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "Top move recommendations:" << std::endl;
    for (int i = 0; i < std::min(3, static_cast<int>(move_probs.size())); i++) {
      int row = move_probs[i].first / 3;
      int col = move_probs[i].first % 3;
      std::cout << "  (" << row+1 << "," << col+1 << "): " 
                << move_probs[i].second * 100 << "%" << std::endl;
    }
  }
  
  // Move network back to CPU before creating the trainer
  network->to(torch::kCPU);
  Trainer trainer(network, config);

  // Store and display evaluation results
  auto eval_stats = trainer.EvaluateAgainstRandom();
  std::cout << "\n==== Evaluation Against Random Player ====" << std::endl;
  std::cout << "Wins: " << eval_stats.win_rate << std::endl;
  std::cout << "Losses: " << eval_stats.loss_rate << std::endl;
  std::cout << "Draws: " << eval_stats.draw_rate << std::endl;
  std::cout << "Win rate: " << (eval_stats.win_rate * 100.0f / config.num_evaluation_games) << "%" << std::endl;
  
  // Save the comprehensively trained model
  try {
    std::filesystem::create_directories("models");
    std::string model_path = "models/comprehensive_model.pt";
    torch::save(network, model_path);
    std::cout << "\nComprehensively trained network saved to " << model_path << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error saving model: " << e.what() << std::endl;
  }
}

int main(int argc, char** argv) {
  Config config = Config::ParseCommandLine(argc, argv);
  std::shared_ptr<NeuralNetwork> network = std::make_shared<NeuralNetwork>(config);
  // Train on all possible games
  runComprehensiveTraining(network, config);
  
  return 0;
}