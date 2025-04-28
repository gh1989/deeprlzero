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

void runMiddleSquareDemonstration(std::shared_ptr<NeuralNetwork> network) {
  std::cout << "\n==== Middle Square Learning Demonstration ====" << std::endl;
  
  // Create a board with X in the middle only
  TicTacToe game;
  game.MakeMove(4); // Middle square (1,1) - 0-indexed
  
  std::cout << "Training on position with X in middle:" << std::endl;
  std::cout << game.ToString();
  
  // Create a GameEpisode with just this position
  GameEpisode demo_episode;
  
  // Create a policy that strongly prefers the middle
  std::vector<float> middle_policy(TicTacToe::kNumActions, 0.0f);
  middle_policy[4] = 1.0f; // 100% probability on middle
  
  // Add to our episode
  demo_episode.boards.push_back(game.GetCanonicalBoard());
  demo_episode.policies.push_back(middle_policy);
  demo_episode.outcome = 1.0f; // X is winning
  
  // Check network prediction on empty board before training
  network->eval();
  float initial_middle_preference = 0.0f;  // Declare this outside the block
  {
    torch::NoGradGuard no_grad;
    TicTacToe empty_game;
    torch::Tensor empty_state = empty_game.GetCanonicalBoard();
    auto [initial_policy, initial_value] = network->forward(empty_state.unsqueeze(0));
    
    std::cout << "\nBefore training (empty board):" << std::endl;
    std::cout << "Policy output for each position:" << std::endl;
    for (int i = 0; i < 9; i++) {
      std::cout << initial_policy[0][i].item<float>() << " ";
      if (i % 3 == 2) std::cout << std::endl;
    }
    std::cout << "Middle square preference: " << initial_policy[0][4].item<float>() << std::endl;
    initial_middle_preference = initial_policy[0][4].item<float>();  // Save this value for later
  }
  
  // Train the network on just this single example
  network->train();
  
  // Prepare training data
  torch::Tensor state_batch = demo_episode.boards[0].unsqueeze(0).clone().set_requires_grad(true);
  torch::Tensor policy_batch = torch::from_blob(
      const_cast<float*>(middle_policy.data()),
      {1, TicTacToe::kNumActions},
      torch::kFloat32).clone().set_requires_grad(true);
  torch::Tensor value_batch = torch::tensor({{1.0f}}).clone().set_requires_grad(true);
  
  // Configure optimizer with higher learning rate
  torch::optim::Adam optimizer(
      network->parameters(),
      torch::optim::AdamOptions(0.05).weight_decay(0.0001));  // Increased from 0.01 to 0.05
  
  // Train for 2000 epochs (previously 1000)
  std::cout << "\nTraining for 2000 epochs on this single example..." << std::endl;
  
  for (int epoch = 0; epoch < 2000; epoch++) {
    optimizer.zero_grad();
    
    auto output = network->forward(state_batch);
    torch::Tensor policy_output = output.first;
    torch::Tensor value_output = output.second;
    
    // Use cross-entropy loss for policy (better for classification tasks)
    torch::Tensor log_softmax_policy = torch::log_softmax(policy_output, 1);
    torch::Tensor policy_loss = -torch::sum(policy_batch * log_softmax_policy) / policy_batch.size(0);
    
    // Keep MSE for value loss
    torch::Tensor value_loss = torch::mse_loss(value_output, value_batch);
    
    // Emphasize policy learning by weighting it higher
    torch::Tensor total_loss = 5.0 * policy_loss + value_loss;
    
    if (epoch % 100 == 0) {
      std::cout << "Epoch " << epoch << ", Policy Loss: " << policy_loss.item<float>() 
                << ", Value Loss: " << value_loss.item<float>() 
                << ", Total Loss: " << total_loss.item<float>() << std::endl;
    }
    
    total_loss.backward();
    optimizer.step();
  }
    network->eval();
  Config config;
  network->to(torch::kCPU);
  Trainer trainer(network, config);
  trainer.EvaluateAgainstRandom();
/*
  // Check network prediction after training
  {
    torch::NoGradGuard no_grad;
    TicTacToe empty_game;
    torch::Tensor empty_state = empty_game.GetCanonicalBoard();
    auto [final_policy, final_value] = network->forward(empty_state.unsqueeze(0));
    
    // Apply softmax to get probabilities
    torch::Tensor softmax_policy = torch::softmax(final_policy, 1);
    
    std::cout << "\nAfter training (empty board):" << std::endl;
    std::cout << "Policy output for each position (after softmax):" << std::endl;
    for (int i = 0; i < 9; i++) {
      std::cout << softmax_policy[0][i].item<float>() << " ";
      if (i % 3 == 2) std::cout << std::endl;
    }
    std::cout << "Middle square preference: " << softmax_policy[0][4].item<float>() * 100.0f << "%" << std::endl;
    std::cout << "Did the network learn to prefer the middle? " 
              << (softmax_policy[0][4].item<float>() > 0.5f ? "YES!" : "No") << std::endl;
    
    // Show raw outputs too
    std::cout << "\nRaw policy logits:" << std::endl;
    for (int i = 0; i < 9; i++) {
      std::cout << final_policy[0][i].item<float>() << " ";
      if (i % 3 == 2) std::cout << std::endl;
    }
  }
  */
}

void GenerateAllTicTacToeGames(std::vector<GameEpisode>& episodes, std::shared_ptr<TicTacToe> game, 
                               std::vector<torch::Tensor>& boards, 
                               std::vector<std::vector<float>>& policies, 
                               int depth = 0) {
  // If the game is over, we've found a complete path
  if (game->IsTerminal()) {
    GameEpisode episode;
    episode.boards = boards;
    episode.policies = policies;
    episode.outcome = game->GetGameResult();
    episodes.push_back(episode);
    return;
  }

  // Get all valid moves for current game state
  auto valid_moves = game->GetValidMoves();
  
  // Try each valid move
  for (int move : valid_moves) {
    // Create policy that focuses on this move
    std::vector<float> policy(TicTacToe::kNumActions, 0.0f);
    policy[move] = 1.0f;
    
    // Save the current state
    boards.push_back(game->GetCanonicalBoard());
    policies.push_back(policy);
    
    // Make the move
    game->MakeMove(move);
    
    // Recursive DFS to explore all possible game continuations
    GenerateAllTicTacToeGames(episodes, std::make_shared<TicTacToe>(*game), boards, policies, depth + 1);
    
    // Backtrack
    boards.pop_back();
    policies.pop_back();
    
    // Undo the move (don't really need this since we use a copy for recursion)
    if (depth < 8) {  // No need to undo on the last move
      game->UndoMove(move);
    }
  }
}

void runComprehensiveTraining(std::shared_ptr<NeuralNetwork> network) {
  std::cout << "\n==== Comprehensive Tic-tac-toe Training ====" << std::endl;
  std::cout << "Generating all possible valid games..." << std::endl;
  
  // Generate all possible games
  std::vector<GameEpisode> all_episodes;
  std::vector<torch::Tensor> boards;
  std::vector<std::vector<float>> policies;
  
  auto game = std::make_shared<TicTacToe>();
  GenerateAllTicTacToeGames(all_episodes, game, boards, policies);
  
  std::cout << "Generated " << all_episodes.size() << " complete games" << std::endl;
  
  // Select CUDA device
  torch::Device device(torch::kCUDA, 0);
  
  // Move network to GPU
  network->to(device);
  
  // Prepare training data
  std::vector<torch::Tensor> all_boards;
  std::vector<torch::Tensor> all_policies;
  std::vector<float> all_values;
  
  for (const auto& episode : all_episodes) {
    for (size_t i = 0; i < episode.boards.size(); i++) {
      all_boards.push_back(episode.boards[i]);
      
      // Create policy tensor
      all_policies.push_back(torch::from_blob(
          const_cast<float*>(episode.policies[i].data()),
          {TicTacToe::kNumActions}, 
          torch::kFloat32).clone());
      
      // Add value targets (adjusted for player perspective)
      float perspective = (i % 2 == 0) ? 1.0f : -1.0f;
      all_values.push_back(episode.outcome * perspective);
    }
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
      torch::optim::AdamOptions(0.001).weight_decay(0.0001));
  
  // Train for multiple epochs
  int num_epochs = 256;
  std::cout << "\nTraining for " << num_epochs << " epochs on all possible positions..." << std::endl;
  
  int batch_size = 1;
  int num_batches = (states_batch.sizes()[0] + batch_size - 1) / batch_size;
  
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    float epoch_policy_loss = 0.0f;
    float epoch_value_loss = 0.0f;
    float epoch_total_loss = 0.0f;
    
    // Create random indices for this epoch
    auto indices = torch::randperm(states_batch.sizes()[0], device);
    
    for (int batch = 0; batch < num_batches; batch++) {
      // Zero gradients
      optimizer.zero_grad();
      
      // Get mini-batch
      int start_idx = batch * batch_size;
      int end_idx = std::min(start_idx + batch_size, static_cast<int>(states_batch.sizes()[0]));
      auto batch_indices = indices.slice(0, start_idx, end_idx);
      
      auto batch_states = states_batch.index_select(0, batch_indices);
      auto batch_policies = policy_batch.index_select(0, batch_indices);
      auto batch_values = values.index_select(0, batch_indices);
      
      // Forward pass
      auto [policy_output, value_output] = network->forward(batch_states);
      
      // Use cross-entropy loss for policy
      torch::Tensor log_softmax_policy = torch::log_softmax(policy_output, 1);
      torch::Tensor policy_loss = -torch::sum(batch_policies * log_softmax_policy) / batch_policies.size(0);
      
      // MSE loss for value
      torch::Tensor value_loss = torch::mse_loss(value_output, batch_values);
      
      // Combined loss (weight policy higher)
      torch::Tensor total_loss = 5.0 * policy_loss + value_loss;
      
      // Backward pass and optimization
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
  Config config;
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
  Config config;
  std::shared_ptr<NeuralNetwork> network = std::make_shared<NeuralNetwork>(config);
  // Train on all possible games
  runComprehensiveTraining(network);
  
  return 0;
}