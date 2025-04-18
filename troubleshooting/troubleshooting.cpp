#include <iostream>
#include "core/mcts.h"
#include "core/neural_network.h"
#include "core/network_manager.h"
#include "core/tictactoe.h"
#include "core/thread.h"
#include "core/utils.h"
#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <cassert>

#include "core/trainer.h"
#include "core/self_play.h"
#include "core/logger.h"

namespace alphazero {

Config TestConfig() {
    Config config;
    config.learning_rate = 0.001f;
    config.l2_reg = 1e-4f;
    config.temperature = 1.0f;
    config.num_evaluation_games = 10;
    return config;
}

void PrintTestResult(const std::string& test_name, bool passed) {
    std::cout << test_name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

// Test case for NeuralNetwork clone functionality.
void TestNeuralNetworkClone() {
  std::cout << "\nRunning NeuralNetwork Clone Test...\n";
  bool passed = true;

  // Create an original instance of the mock neural network.
  Config config;
  auto original = std::make_unique<NeuralNetwork>(config);

  // Clone the network.
  auto cloned = std::dynamic_pointer_cast<NeuralNetwork>(original->clone());

  // Ensure that the cloned pointer is different from the original.
  if (original.get() == cloned.get()) {
    std::cout << "Clone returned the same pointer as original!\n";
    passed = false;
  }


  torch::Tensor input = torch::rand({1, 1, 3, 3});
  auto game = std::make_unique<TicTacToe>();

  // Compute the forward pass from both the original and the cloned network.
  auto [policy_original, value_original] = original->forward(input);
  auto [policy_cloned, value_cloned] = cloned->forward(input);

  // Verify that both policy and value outputs are the same.
  if (!torch::allclose(policy_original, policy_cloned)) {
    std::cout << "Policy outputs differ between original and clone!\n";
    passed = false;
  }

  if (!torch::allclose(value_original, value_cloned)) {
    std::cout << "Value outputs differ between original and clone!\n";
    passed = false;
  }

  PrintTestResult("NeuralNetwork Clone Test", passed);
}

void TestCanonicalBoard() {
  std::cout << "\nRunning Canonical Board Test...\n";
  bool passed = true;

  auto game = std::make_unique<TicTacToe>();
  auto canonical_board = game->GetCanonicalBoard();
  std::cout << "Canonical Board: \n" << canonical_board << "\n";

  game->MakeMove(4);
  canonical_board = game->GetCanonicalBoard();
  std::cout << "Canonical Board after move: \n" << canonical_board << "\n";

  game->MakeMove(5);
  canonical_board = game->GetCanonicalBoard();
  std::cout << "Canonical Board after move: \n" << canonical_board << "\n";
}

void TestForwardPassDifferentPlayers() {
  std::cout << "\nRunning Forward Pass Different Players Test...\n";
  bool passed = true;

  // Create an original instance of the mock neural network.
  Config config;
  auto network = std::make_unique<NeuralNetwork>(config);

  auto game = std::make_unique<TicTacToe>();
  game->MakeMove(4);
  auto input = game->GetCanonicalBoard().unsqueeze(0);
  std::cout << "Input: \n" << input << "\n";
  auto [policy, value] = network->forward(input);
  std::cout << "Policy: \n" << policy << "\n";
  std::cout << "Value: \n" << value << "\n";

  game->MakeMove(5);
  auto input2 = game->GetCanonicalBoard().unsqueeze(0);
  std::cout << "Input2: \n" << input2 << "\n";
  auto [policy2, value2] = network->forward(input2);
  std::cout << "Policy: \n" << policy2 << "\n";
  std::cout << "Value: \n" << value2 << "\n";
}

// Test case for computing policy loss.
void TestComputePolicyLoss() {
  bool passed = true;
  
  // Create a Trainer instance.
  Config config = TestConfig();
  auto network = std::make_shared<NeuralNetwork>(config);
  Trainer trainer(network, config);
  
  // --- Test 1 ---
  // When the predicted policy equals the target policy, the loss should be zero.
  // Use a one-hot vector to guarantee zero entropy.
  auto policy_same = torch::tensor({{1.0f, 0.0f, 0.0f}}, torch::kFloat);
  auto loss_same = trainer.ComputePolicyLoss(policy_same, policy_same);
  bool test1 = (std::abs(loss_same.item<float>()) < 1e-5);
  if (!test1) {
    std::cout << "Test 1 failed: Expected policy loss 0 for identical inputs, got " 
              << loss_same.item<float>() << "\n";
  }
  
  // --- Test 2 ---
  // If the predicted policy and target policy are different, the loss must be positive.
  auto policy_preds = torch::tensor({{0.1f, 0.2f, 0.7f}}, torch::kFloat);
  auto policy_target = torch::tensor({{1.0f, 0.0f, 0.0f}}, torch::kFloat);
  auto loss_diff = trainer.ComputePolicyLoss(policy_preds, policy_target);
  bool test2 = (loss_diff.item<float>() > 0.0f);
  if (!test2) {
    std::cout << "Test 2 failed: Expected policy loss > 0 for different inputs, got " 
              << loss_diff.item<float>() << "\n";
  }
  
  passed = test1 && test2;
  PrintTestResult("Policy Loss Test", passed);
}

// Test case for computing value loss.
void TestComputeValueLoss() {
  bool passed = true;
  Config config = TestConfig();
  auto network = std::make_shared<NeuralNetwork>(config);
  Trainer trainer(network, config);
  
  // --- Test 1 ---
  // When prediction and target are identical the loss should be zero.
  auto value_preds_same = torch::tensor({1.0f}, torch::kFloat);
  auto value_targets_same = torch::tensor({1.0f}, torch::kFloat);
  auto loss_same = trainer.ComputeValueLoss(value_preds_same, value_targets_same);
  bool test1 = (std::abs(loss_same.item<float>()) < 1e-5);
  if (!test1) {
    std::cout << "Test 1 failed: Expected value loss 0 for identical inputs, got " 
              << loss_same.item<float>() << "\n";
  }
  
  // --- Test 2 ---
  // When values differ the loss should be positive (MSE between 0 and 1 equals 1).
  auto value_preds_diff = torch::tensor({0.0f}, torch::kFloat);
  auto value_targets_diff = torch::tensor({1.0f}, torch::kFloat);
  auto loss_diff = trainer.ComputeValueLoss(value_preds_diff, value_targets_diff);
  bool test2 = (loss_diff.item<float>() > 0.0f);
  if (!test2) {
    std::cout << "Test 2 failed: Expected value loss > 0 for different inputs, got " 
              << loss_diff.item<float>() << "\n";
  }
  
  passed = test1 && test2;
  PrintTestResult("Value Loss Test", passed);
}

// Test case to ensure that the network parameters are updated during a training step.
void CheckNetworkUpdates() {
  bool passed = false;
  
  // Create a simple NeuralNetwork instance.
  // Here, we use arbitrary parameters (input channels=1, filters=16, num_actions=9, residual blocks=1)
  Config config;
  auto network = std::make_shared<NeuralNetwork>(1, 16, 9, 1);
  
  // Create an Adam optimizer for the network.
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(0.001));
  
  // Prepare a dummy input state: batch of 1 with 1 channel and 3x3 board.
  auto state = torch::rand({1, 1, 3, 3}, torch::kFloat);
  
  // Create a target policy: one-hot vector for 9 actions.
  auto policy_target = torch::zeros({1, 9}, torch::kFloat);
  policy_target[0][0] = 1.0f;
  
  // Create a target value (for example, a win).
  auto value_target = torch::tensor({1.0f}, torch::kFloat);
  
  // Create a Trainer instance.
  Trainer trainer(network, config);
  
  // Compute network predictions.
  auto output = network->forward(state);
  auto policy_preds = output.first;
  auto value_preds = output.second;
  
  // Compute the losses.
  auto loss_policy = trainer.ComputePolicyLoss(policy_preds, policy_target);
  auto loss_value = trainer.ComputeValueLoss(value_preds, value_target);
  auto loss = loss_policy + loss_value;
  
  // Save copies of network parameters before performing the update.
  std::vector<torch::Tensor> params_before;
  for (const auto &param : network->parameters()) {
    params_before.push_back(param.clone());
  }
  
  // Do a training update.
  optimizer.zero_grad();
  loss.backward();
  optimizer.step();
  
  // Check whether any network parameter has been updated.
  auto params_after = network->parameters();
  for (size_t i = 0; i < params_after.size(); ++i) {
    if (!torch::allclose(params_after[i], params_before[i])) {
      passed = true;
      break;
    }
  }
  
  if (!passed) {
    std::cout << "No network parameters were updated during training.\n";
  }
  PrintTestResult("Network Update Test", passed);
}

// Test the ExecuteEpisode() functionality of SelfPlay.
void TestExecuteEpisodeSelfPlay() {
  std::cout << "Running TestExecuteEpisodeSelfPlay..." << std::endl;

  Config config = TestConfig();
  auto network = std::make_shared<NeuralNetwork>(1, 16, 9, 1);
  SelfPlay<TicTacToe> self_play(network, config, config.initial_temperature);

  GameEpisode episode = self_play.ExecuteEpisode();
  bool passed = true;

  if (episode.boards.empty()) {
    std::cout << "Test failed: No moves were generated during the episode." << std::endl;
    passed = false;
  }

  // For TicTacToe, an episode should not exceed 9 moves.
  if (episode.boards.size() > 9) {
    std::cout << "Test failed: Episode generated " << episode.boards.size()
              << " moves which exceeds the maximum of 9 moves." << std::endl;
    passed = false;
  }

  // Validate each move in the episode.
  for (size_t i = 0; i < episode.boards.size(); ++i) {
    const auto &board = episode.boards[i];
    if (!board.defined()) {
      std::cout << "Test failed: Move " << i << " has an undefined board tensor." << std::endl;
      passed = false;
      break;
    }
    if (board.numel() != 9) {
      std::cout << "Test failed: Move " << i << " board tensor does not contain 9 elements (has "
                << board.numel() << " elements)." << std::endl;
      passed = false;
      break;
    }
    if (episode.policies[i].size() != 9) {
      std::cout << "Test failed: Move " << i << " policy vector has " << episode.policies[i].size()
                << " elements instead of 9." << std::endl;
      passed = false;
      break;
    }
    if (episode.values[i] < -1.0f || episode.values[i] > 1.0f) {
      std::cout << "Test failed: Move " << i << " has a value out of range: " << episode.values[i] << std::endl;
      passed = false;
      break;
    }
  }

  std::cout << "SelfPlay::ExecuteEpisode generated " << episode.boards.size() << " moves." << std::endl;
  PrintTestResult("ExecuteEpisode SelfPlay Test", passed);
}

void PrintSelfPlayEpisode() {
  std::cout << "Printing a self–play episode:" << std::endl;

  Config config = TestConfig();
  auto network = std::make_shared<NeuralNetwork>(1, 16, 9, 1);
  SelfPlay<TicTacToe> self_play(network, config, config.initial_temperature);

  GameEpisode episode = self_play.ExecuteEpisode();

  std::cout << "Generated " << episode.boards.size() << " moves in the episode." << std::endl;

  for (size_t i = 0; i < episode.boards.size(); ++i) {
    std::cout << "Move " << i << ":" << std::endl;
    std::cout << "Board: " << episode.boards[i] << std::endl;
    std::cout << "Policy: [ ";
    for (const auto &prob : episode.policies[i]) {
      std::cout << prob << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Value: " << episode.values[i] << std::endl;
    std::cout << "--------------------------" << std::endl;
  }
}

// TestMCTSExplorationStats verifies that MCTS properly records its exploration statistics.
void TestMCTSExplorationStats() {
  // Create a dummy configuration.
  Config config = TestConfig();
  // (Initialize config fields if necessary.)

  // Create a neural network instance.
  // Constructor: NeuralNetwork(int64_t input_channels, int64_t num_filters, 
  //                             int64_t num_actions, int64_t num_residual_blocks);
  auto network = std::make_shared<NeuralNetwork>(1, 32, 9, 3);

  // Instantiate MCTS with the network and configuration.
  MCTS mcts(network, config);

  // Retrieve the root node for the search tree.
  // It is assumed that GetRoot() returns a pointer to the current root Node.
  auto* root = mcts.GetRoot();
  assert(root != nullptr && "MCTS did not create a valid root node.");

  // Create a TicTacToe game instance.
  auto game = std::make_unique<TicTacToe>();

  // Run MCTS search for a fixed number of iterations.
  // Your MCTS search function's signature is: void Search(Game& state, Node* node);
  const int kSimulations = 100;
  for (int i = 0; i < kSimulations; ++i) {
    mcts.Search(game.get(), root);
  }

  // Retrieve exploration statistics using GetStats().
  auto stats = mcts.GetStats();
  stats.LogStatistics();
}

void TestSaveTestNetwork() {
    std::cout << "Running SaveTestNetwork..." << std::endl;
    Config test_config;
    test_config.model_path = "test_model";
    NetworkManager network_manager(test_config);
    network_manager.SetBestNetwork(network_manager.CreateInitialNetwork());
    network_manager.SaveBestNetwork();
}

void TestNetworkManager() {
    std::cout << "Running TestNetworkManager..." << std::endl;
    Config test_config;
    test_config.model_path = "test_model";
    NetworkManager network_manager(test_config);
    network_manager.SetBestNetwork(network_manager.CreateInitialNetwork());
    network_manager.LoadBestNetwork();
    EvaluationStats eval_stats{0.5f, 0.0f, 0.0f};
    network_manager.AcceptOrRejectNewNetwork(network_manager.GetBestNetwork(), eval_stats);
}

void TestMergeAndClearStats() {
  // Create an aggregator MCTSStats object (initially all values are zero).
  MCTSStats aggregate_stats;

  // Create a dummy move_stats object and set test values.
  MCTSStats move_stats;
  move_stats.RecordNodeStats(1, 1, 1.0f, 1.0f, 1.0f, true);
  
  // Log the dummy move stats.
  Logger::GetInstance().LogFormat(
      "Before merging: move_stats -> Simulations: {}, Expanded Nodes: {}",
      move_stats.GetNumSimulations(), move_stats.GetNumExpandedNodes());

  // Merge the move stats into the aggregate stats (accumulating instead of replacing).
  aggregate_stats.MergeStats(move_stats);
  aggregate_stats.LogStatistics();

  std::cout << "TestMergeAndClearStats passed!" << std::endl;
}

void TestMCTSHasStats() {
    Config config = TestConfig();
    auto network = std::make_shared<NeuralNetwork>(1, 16, 9, 1);
    MCTS mcts(network, config);
    auto game = std::make_unique<TicTacToe>();
    mcts.Search(game.get(), mcts.GetRoot());
    auto stats = mcts.GetStats();
    stats.LogStatistics();
}

void TestSelfPlayStats() {
    Config config = TestConfig();
    auto network = std::make_shared<NeuralNetwork>(1, 16, 9, 1);
    SelfPlay<TicTacToe> self_play(network, config, config.initial_temperature);
    auto stats = self_play.GetStats();
    stats.LogStatistics();
}

void TestParallelFor() {
  Config config = TestConfig();
  auto network = std::make_shared<NeuralNetwork>(1, 16, 9, 1);
  // Accumulate full game episodes rather than partial examples.
  std::vector<GameEpisode> episodes;

  // Aggregated statistics from all episodes.
  MCTSStats aggregated_stats;

  // Run self-play episodes in parallel.
  ParallelFor(config.episodes_per_iteration, [&](int episode_idx) {
    // Get a thread-local SelfPlay instance using the factory function.
    auto &local_self_play = GetThreadLocalInstance<SelfPlay<TicTacToe>>([&]() {
      return new SelfPlay<TicTacToe>(network, config, config.initial_temperature);
    });

    // Execute a self-play episode.
    GameEpisode episode = local_self_play.ExecuteEpisode();

    // Enter critical section to safely update shared data.
#ifdef _OPENMP
#pragma omp critical
#endif
    {
      // Append the full game episode to the episodes vector.
      episodes.push_back(std::move(episode));
      // Merge the thread-local stats into the aggregated stats.
      aggregated_stats.MergeStats(local_self_play.GetStats());
      // Clear the thread-local stats for the next episode.
      local_self_play.ClearStats();
    }
  });

  std::cout << "Episodes: " << episodes.size()
            << ", Episodes per iteration: " << config.episodes_per_iteration
            << std::endl;
  aggregated_stats.LogStatistics();
  std::cout << "TestParallelFor passed!" << std::endl;
}

void TestAllEpisodes() {
  auto episodes = AllEpisodes();
  std::cout << "All episodes: " << episodes.size() << std::endl;
}

void TestEpisodesQuality() {
    std::vector<GameEpisode> episodes;
    auto network = std::make_shared<NeuralNetwork>(1, 16, 9, 1);
    Config config = TestConfig();
    auto self_play = std::make_unique<SelfPlay<TicTacToe>>(network, config, config.initial_temperature);
    for (int i = 0; i < 100; ++i) {
      episodes.push_back(self_play->ExecuteEpisode());
    }

    std::cout << "Episodes: " << episodes.size() << std::endl;
  
    for (const auto& episode : episodes) {
      std::cout << std::format("Episode: {} Policy: {} Values: {} Outcome: {}", episode.boards.size(), episode.policies.size(), episode.values.size(), episode.outcome) << std::endl;

      // Print board rows across all moves in this episode
      for (int row = 0; row < 3; row++) {
        for (size_t move = 0; move < episode.boards.size(); move++) {
            auto board_tensor = episode.boards[move];
            // Access the tensor data correctly
            auto accessor = board_tensor.accessor<float, 3>();
            
            // Print row for this move's board
            for (int col = 0; col < 3; col++) {
                float value = accessor[0][row][col];  // Using proper tensor indexing
                if (value == 0) std::cout << ". ";
                else if (value == 1) std::cout << "X ";
                else std::cout << "O ";
            }
            std::cout << "   "; // Space between boards
        }
        std::cout << std::endl;
      }
      std::cout << std::endl; // Space between episodes
    }
  }

  void TestEvaluationLogicIssueDeterministic() {
    Config config = TestConfig();
    int wins = 0, draws = 0, losses = 0;
    const int total_games = config.num_evaluation_games;
    
    auto network = std::make_shared<NeuralNetwork>(1, 16, 9, 1);
    auto opponent = std::make_shared<NeuralNetwork>(1, 16, 9, 1);

    // Check if networks are identical by comparing their parameters
    bool networks_identical = true;
    auto main_params = network->parameters();
    auto opp_params = opponent->parameters();
    
    if (main_params.size() != opp_params.size()) {
        networks_identical = false;
    } else {
        for (size_t i = 0; i < main_params.size(); i++) {
            if (!torch::equal(main_params[i], opp_params[i])) {
                networks_identical = false;
                break;
            }
        }
    }
    
    if (networks_identical) {
        throw std::runtime_error("Evaluator: Cannot evaluate a network against an identical network!");
    }

    auto output_main = network->forward(torch::zeros({1, 1, 3, 3}));
    auto output_opponent = opponent->forward(torch::zeros({1, 1, 3, 3}));

    std::cout << "Output main: " << output_main << std::endl;
    std::cout << "Output opponent: " << output_opponent << std::endl;

    MCTS mcts_main(network, config);
    MCTS mcts_opponent(opponent, config);
    
    for (int i = 0; i < total_games; ++i) {
        auto game = std::make_unique<TicTacToe>();
        bool network_plays_first = (i % 2 == 0);
        
        //std::cout << "\nGame " << i << ": Main network plays " 
        //          << (network_plays_first ? "first" : "second") << "\n";
        int move_count = 0;
        
        while (!game->IsTerminal()) {
            bool is_network_turn = ((game->GetCurrentPlayer() == 1) == network_plays_first);
            
            if (is_network_turn) {
                mcts_main.ResetRoot();
                for (int sim = 0; sim < config.num_simulations; ++sim) {
                    mcts_main.Search(game.get(), mcts_main.GetRoot());
                }
                int action = mcts_main.SelectMove(game.get(), 0.0f);
                //std::cout << "Main network move " << move_count << ": " << action << "\n";
                game->MakeMove(action);
            } else {
                mcts_opponent.ResetRoot();
                for (int sim = 0; sim < config.num_simulations; ++sim) {
                    mcts_opponent.Search(game.get(), mcts_opponent.GetRoot());
                }
                int action = mcts_opponent.SelectMove(game.get(), 0.0f);
                //std::cout << "Opponent move " << move_count << ": " << action << "\n";
                game->MakeMove(action);
            }
            move_count++;
        }
        
        float result = game->GetGameResult();
        float perspective_result = network_plays_first ? result : -result;
        
        if (perspective_result > 0) {
            wins++;
            std::cout << "W";
        } else if (perspective_result == 0) {
            draws++;
            std::cout << "D";
        } else {
            losses++;
            std::cout << "L";
        }
    }
    
    EvaluationStats stats;
    stats.win_rate = static_cast<float>(wins) / total_games;
    stats.draw_rate = static_cast<float>(draws) / total_games;
    stats.loss_rate = static_cast<float>(losses) / total_games;
    std::cout << "EvaluationStats: " << stats.win_rate << ", " << stats.draw_rate << ", " << stats.loss_rate << std::endl;
  }
}

int main() {
  using namespace alphazero;
  try {
    std::cout << "Starting MCTS Tests\n";
    std::cout << "===================\n";
    TestNeuralNetworkClone();
    TestCanonicalBoard();
    TestForwardPassDifferentPlayers();
    TestComputePolicyLoss();
    TestComputeValueLoss();
    CheckNetworkUpdates();
    TestExecuteEpisodeSelfPlay();
    PrintSelfPlayEpisode();
    TestMCTSExplorationStats();
    TestSaveTestNetwork();
    TestNetworkManager();
    TestMergeAndClearStats();
    TestSelfPlayStats();
    TestMCTSHasStats();
    TestEvaluationLogicIssueDeterministic();
    //TestParallelFor();
    //TestAllEpisodes();
    //TestEpisodesQuality();
    std::cout << "\nAll tests completed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}