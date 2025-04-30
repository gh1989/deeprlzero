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
  std::cout << "Generating all legal tic-tac-toe positions with minimax values..." << std::endl;
  
  GamePositions all_positions;
  std::map<std::string, float> minimax_values;
  std::map<std::string, std::vector<float>> minimax_policies;
  
  bool debug_policy = false;  // Set to true for debugging
  
  // Recursive function to explore game tree and compute minimax values
  std::function<float(const TicTacToe&, std::vector<float>&)> exploreMinimax;
  exploreMinimax = [&all_positions, &minimax_values, &minimax_policies, &exploreMinimax, &debug_policy](
      const TicTacToe& state, std::vector<float>& best_policy) -> float {
    
    // Create key for this position
    std::string key = state.GetBoardString() + std::to_string(state.GetCurrentPlayer());
    
    // If already computed, return cached value
    if (minimax_values.count(key)) {
      best_policy = minimax_policies[key];
      return minimax_values[key];
    }
    
    // If terminal, return game result
    if (state.IsTerminal()) {
      float result = state.GetGameResult();
      minimax_values[key] = result;
      return result;
    }
    
    // Get valid moves and prepare to compute minimax
    auto valid_moves = state.GetValidMoves();
    float best_value = -2.0f; // Initialize to worse than worst possible value
    std::vector<float> policy(TicTacToe::kNumActions, 0.0f);
    std::vector<int> best_moves; // Track optimal moves
    
    // First check for immediate winning moves
    for (int move : valid_moves) {
      TicTacToe next_state(state);
      next_state.MakeMove(move);
      
      // If this move leads to a win, it's the only good move
      if (next_state.IsTerminal() && next_state.GetGameResult() == 1.0f) {
        policy[move] = 1.0f;  // Assign 100% probability to winning move
        minimax_values[key] = 1.0f;
        minimax_policies[key] = policy;
        best_policy = policy;
        
        // Add non-terminal position to training data
        all_positions.boards.push_back(state.GetCanonicalBoard());
        all_positions.policies.push_back(policy);
        all_positions.values.push_back(1.0f);
        
        return 1.0f;
      }
    }
    
    // If no immediate win, proceed with regular minimax
    for (int move : valid_moves) {
      TicTacToe next_state(state);
      next_state.MakeMove(move);
      
      std::vector<float> child_policy;
      float child_value = -exploreMinimax(next_state, child_policy); // Negamax formulation
      
      if (child_value > best_value + 0.001f) {  // Use small epsilon to avoid floating point issues
        best_value = child_value;
        // Reset list of best moves
        best_moves.clear();
        best_moves.push_back(move);
      }
      else if (std::abs(child_value - best_value) <= 0.001f) {  // Equal value move
        best_moves.push_back(move);
      }
    }
    
    // Set policy to concentrate on best moves only
    for (int move : best_moves) {
      policy[move] = 1.0f / best_moves.size();  // Equal probability among best moves only
    }
    
    // Cache the results
    minimax_values[key] = best_value;
    minimax_policies[key] = policy;
    best_policy = policy;
    
    // Add non-terminal position to training data
    if (!state.IsTerminal()) {
      all_positions.boards.push_back(state.GetCanonicalBoard());
      all_positions.policies.push_back(policy);
      all_positions.values.push_back(best_value);
    }
    
    if (debug_policy && state.GetBoardString() == "XXXOO....") {
      std::cout << "DEBUG: Policy for position:\n";
      std::cout << state.ToString() << std::endl;
      std::cout << "Player: " << state.GetCurrentPlayer() << std::endl;
      std::cout << "Value: " << best_value << std::endl;
      std::cout << "Policy: ";
      for (int i = 0; i < TicTacToe::kNumActions; i++) {
        std::cout << policy[i] << " ";
        if (i % 3 == 2) std::cout << std::endl << "        ";
      }
      std::cout << std::endl;
    }
    
    return best_value;
  };
  
  // Add after the exploreMinimax lambda but before starting exploration
  // Enable this to verify policy computation during generation
  bool verify_policy = true;
  
  // Add these test cases
  std::vector<std::string> test_positions = {
    ".........",  // Empty board
    "X........",  // X in top-left
    "XO.......",  // X top-left, O top-middle
    "XXXOO...."   // Forcing position
  };
  
  // Start exploration from the initial state
  TicTacToe initial_game;
  std::vector<float> initial_policy;
  exploreMinimax(initial_game, initial_policy);
  
  // Verify policies for test positions
  if (verify_policy) {
    std::cout << "\nVerifying policy computation for test positions:" << std::endl;
    
    // Test specific board layouts with clear forcing moves
    std::vector<std::unique_ptr<TicTacToe>> test_boards;
    
    // X can win by completing top row
    auto board1 = std::make_unique<TicTacToe>();
    board1->MakeMove(0); // X top-left
    board1->MakeMove(3); // O middle-left
    board1->MakeMove(1); // X top-middle
    board1->MakeMove(4); // O center
    // X to move - should play top-right (2) to win
    test_boards.push_back(std::move(board1));
    
    // O must block X's win
    auto board2 = std::make_unique<TicTacToe>();
    board2->MakeMove(0); // X top-left
    board2->MakeMove(4); // O center
    board2->MakeMove(1); // X top-middle
    // O to move - must block at top-right (2)
    test_boards.push_back(std::move(board2));
    
    // X must block O's win
    auto board3 = std::make_unique<TicTacToe>();
    board3->MakeMove(0); // X top-left
    board3->MakeMove(4); // O center
    board3->MakeMove(8); // X bottom-right
    board3->MakeMove(6); // O bottom-left
    // X must block by playing at 2 (top-right)
    test_boards.push_back(std::move(board3));
    
    // Fork creation - X should play corner to create a fork
    auto board4 = std::make_unique<TicTacToe>();
    board4->MakeMove(0); // X top-left
    board4->MakeMove(4); // O center
    // X should play bottom-right (8) to create a fork
    test_boards.push_back(std::move(board4));
    
    // Fork blocking - O must block X's potential fork
    auto board5 = std::make_unique<TicTacToe>();
    board5->MakeMove(0); // X top-left
    board5->MakeMove(4); // O center
    board5->MakeMove(8); // X bottom-right
    // O must play in a specific corner or edge to block fork
    test_boards.push_back(std::move(board5));
    
    // Process each test board
    for (const auto& test_state : test_boards) {
      std::string key = test_state->GetBoardString() + std::to_string(test_state->GetCurrentPlayer());
      
      if (minimax_values.count(key)) {
        std::cout << "\nPosition:\n" << test_state->ToString() << std::endl;
        std::cout << "Board string: " << test_state->GetBoardString() << std::endl;
        std::cout << "Current player: " << test_state->GetCurrentPlayer() << std::endl;
        std::cout << "Minimax value: " << minimax_values[key] << std::endl;
        std::cout << "Policy: ";
        for (int i = 0; i < TicTacToe::kNumActions; i++) {
          std::cout << minimax_policies[key][i] << " ";
          if (i % 3 == 2) std::cout << std::endl << "        ";
        }
        
        // Check policy sum
        float sum = std::accumulate(minimax_policies[key].begin(), 
                                   minimax_policies[key].end(), 0.0f);
        std::cout << "Policy sum: " << sum << std::endl;
        
        // Find best move
        int best_move = std::max_element(minimax_policies[key].begin(),
                                        minimax_policies[key].end()) - 
                                        minimax_policies[key].begin();
        std::cout << "Best move: (" << best_move/3+1 << "," << best_move%3+1 << ") with prob: " 
                  << minimax_policies[key][best_move] << std::endl;
      } else {
        std::cout << "Position not found: " << test_state->ToString() << std::endl;
        std::cout << "Board string: " << test_state->GetBoardString() << std::endl;
        std::cout << "Current player: " << test_state->GetCurrentPlayer() << std::endl;
      }
    }
  }
  
  // Count statistics
  int terminal_positions = 0;
  int player1_positions = 0;
  int player2_positions = 0;
  
  for (const auto& [key, value] : minimax_values) {
    // Reconstruct state to count stats
    TicTacToe state;
    state.SetFromString(key.substr(0, 9), key[9] - '0');
    
    if (state.IsTerminal()) {
      terminal_positions++;
    } else if (state.GetCurrentPlayer() == 1) {
      player1_positions++;
    } else {
      player2_positions++;
    }
  }
  
  const int expected_total_positions = 5478; 
  const int expected_non_terminal_positions = 4520;
  
  assert(minimax_values.size() == expected_total_positions && 
         "Incorrect number of total positions generated");
  assert(all_positions.boards.size() == expected_non_terminal_positions && 
         "Incorrect number of non-terminal positions generated");
  
  std::cout << "Found " << minimax_values.size() << " total positions" << std::endl;
  std::cout << "  - Terminal positions: " << terminal_positions << std::endl;
  std::cout << "  - Player 1 positions: " << player1_positions << std::endl;
  std::cout << "  - Player 2 positions: " << player2_positions << std::endl;
  std::cout << "  - Training positions: " << all_positions.boards.size() << std::endl;
  
  std::cout << "Computed minimax values for all positions" << std::endl;
  return all_positions;
}

void runComprehensiveTraining(std::shared_ptr<NeuralNetwork> network, Config config) {
  std::cout << "\nComprehensive Tic-tac-toe Training" << std::endl;
  std::cout << "Generating all possible valid games..." << std::endl;
  
  // Generate all possible games
  GamePositions all_episodes = GenerateAllTicTacToeGames();
    Trainer trainer(network, config);

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
  network->NetworkClone(device);

  // 1. Set higher weight for policy loss
  float gamma_alpha = config.gamma_alpha;
  float gamma_beta = 1-gamma_alpha;

  // 2. Use a fixed higher learning rate
  torch::optim::Adam optimizer(
      network->parameters(),
      torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));
  
  // 3. Increase batch size
  int batch_size = config.training_batch_size;  // Use larger batches
  int num_batches = (states_batch.sizes()[0] + batch_size - 1) / batch_size;

  // 4. Add policy visualization before/after training
  std::cout << "\nSample policy targets (first 3 positions):" << std::endl;
  for (int i = 0; i < std::min(3, static_cast<int>(all_policies.size())); i++) {
    std::cout << "Position " << i << " policy:" << std::endl;
    for (int j = 0; j < 9; j++) {
      std::cout << all_policies[i][j].item<float>() << " ";
      if (j % 3 == 2) std::cout << std::endl;
    }
    std::cout << "Value: " << all_values[i] << std::endl << std::endl;
  }

  // Train for multiple epochs
  int num_epochs = config.num_epochs;
  std::cout << "\nTraining for " << num_epochs << " epochs on all possible positions..." << std::endl;
  
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
      
      // Add policy gradient check - previously misplaced outside the loop
      if (epoch == 0 && batch == 0) {
        auto softmax_policy = torch::softmax(policy_output, 1);
        std::cout << "\nInitial network policy output (first sample):" << std::endl;
        for (int j = 0; j < 9; j++) {
          std::cout << softmax_policy[0][j].item<float>() << " ";
          if (j % 3 == 2) std::cout << std::endl;
        }
        std::cout << "\nTarget policy:" << std::endl;
        for (int j = 0; j < 9; j++) {
          std::cout << batch_policies[0][j].item<float>() << " ";
          if (j % 3 == 2) std::cout << std::endl;
        }
        std::cout << std::endl;
      }
      
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

  // Store and display evaluation results
  auto eval_stats = trainer.EvaluateAgainstRandom();
  std::cout << eval_stats.WinStats();

  // Save the comprehensively trained model
  try {
    std::filesystem::create_directories("models");
    std::string model_path = "models/comprehensive_model.pt";
    torch::save(network, model_path);
    std::cout << "\nComprehensively trained network saved to " << model_path << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error saving model: " << e.what() << std::endl;
  }

  // Add after creating the policy tensors
  std::cout << "\nVerifying policy correctness:" << std::endl;

  // Check for specific positions in the generated data
  bool found_empty = false;
  for (size_t i = 0; i < all_episodes.boards.size(); i++) {
    // Convert tensor back to game state for comparison
    auto board_flat = all_episodes.boards[i].flatten().cpu();
    float* board_data = board_flat.data_ptr<float>();
    
    // Check if this is the empty board
    bool is_empty = true;
    for (int j = 0; j < 9; j++) {
      if (std::abs(board_data[j]) > 0.1 || std::abs(board_data[j+9]) > 0.1) {
        is_empty = false;
        break;
      }
    }
    
    if (is_empty) {
      found_empty = true;
      std::cout << "Empty board policy: ";
      for (int j = 0; j < 9; j++) {
        std::cout << all_episodes.policies[i][j] << " ";
        if (j % 3 == 2) std::cout << std::endl << "                  ";
      }
      std::cout << "Sum: " << std::accumulate(all_episodes.policies[i].begin(), 
                                            all_episodes.policies[i].end(), 0.0f) << std::endl;
      break;
    }
  }
  if (!found_empty) {
    std::cout << "Empty board not found in dataset!" << std::endl;
  }

  // Verify a few policies in the training tensor match the original data
  /*
  for (int i = 0; i < std::min(3, static_cast<int>(all_episodes.boards.size())); i++) {
    std::cout << "Position " << i << " - Original vs Tensor:" << std::endl;
    for (int j = 0; j < 9; j++) {
      std::cout << all_episodes.policies[i][j] << " vs " << all_policies[i][j].item<float>() << std::endl;
      if (std::abs(all_episodes.policies[i][j] - all_policies[i][j].item<float>()) > 1e-5) {
        std::cout << "  MISMATCH at position " << j << "!" << std::endl;
      }
    }
  }

  // Check if policy tensors sum to 1
  for (int i = 0; i < std::min(5, static_cast<int>(all_policies.size())); i++) {
    float sum = torch::sum(all_policies[i]).item<float>();
    std::cout << "Policy " << i << " sum: " << sum << std::endl;
    if (std::abs(sum - 1.0f) > 1e-5) {
      std::cout << "  WARNING: Policy doesn't sum to 1.0!" << std::endl;
    }
  }
  */
}

int main(int argc, char** argv) {
  Config config = Config::ParseCommandLine(argc, argv);
  std::shared_ptr<NeuralNetwork> network = std::make_shared<NeuralNetwork>(config);
  // Train on all possible games
  runComprehensiveTraining(network, config);
  
  return 0;
}