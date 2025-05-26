#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "mcts.h"
#include "config.h"
#include "games/variant.h"

namespace deeprlzero {

MCTS::MCTS(std::shared_ptr<NeuralNetwork> network, const Config& config)
    : network_(network), config_(config) {
  assert(network_ != nullptr && "Neural network cannot be null");
  root_ = std::make_unique<Node>(config);
}

void MCTS::ResetRoot() { root_ = std::make_unique<Node>(config_); }

std::vector<float> MCTS::GetActionProbabilities(const GameVariant& state,
                                                float temperature) {
  std::vector<float> pi(GetActionSize(state), 0.0f);
  float sum = 0.0f;

  for (size_t i = 0; i < root_->children.size(); ++i) {
    if (root_->children[i]) {
      float count = static_cast<float>(root_->children[i]->visit_count);
      if (temperature == 0.0f) {
        pi[i] = count;
      } else {
        pi[i] = std::pow(count, 1.0f / temperature);
      }
      sum += pi[i];
    }
  }

  if (sum > 0) {
    for (float& p : pi) {
      p /= sum;
    }
  }

  return pi;
}

void MCTS::Search(const GameVariant& state, Node* node) {
  auto mutable_state = Clone(state);

  if (IsTerminal(mutable_state)) {
    float value = GetGameResult(mutable_state);
    Backpropagate(node, value);
    return;
  }

  if (!node->IsExpanded()) {
    ExpandNode(node, mutable_state);
  }

  auto [action, child] = SelectAction(node, mutable_state);
  last_move_ = action;
  MakeMove(mutable_state, action);
  Search(mutable_state, child);
  UndoMove(mutable_state, last_move_);
}

std::pair<int, Node*> MCTS::SelectAction(Node* node, const GameVariant& state) {
  assert(node != nullptr && "Node cannot be null");
  assert(node->IsExpanded() && "Node must be expanded before selection");

  float best_score = -std::numeric_limits<float>::infinity();
  int best_action = -1;
  Node* best_child = nullptr;

  auto valid_moves = GetValidMoves(state);
  assert(!valid_moves.empty() && "Must have at least one valid move");
  float parent_visit_sqrt =
      std::sqrt(static_cast<float>(node->visit_count) + 1e-6f);

  for (int move : valid_moves) {
    assert(node->children[move] &&
           "Expanded node must have all valid children");

    Node* child = node->children[move].get();

    // Q-value: Use virtual loss for unvisited nodes
    float q_value;
    if (child->visit_count == 0) {  // Update the current node's statistics.
      q_value = 0.0f;               // Default value for unvisited nodes
    } else {
      q_value = child->value_sum / child->visit_count;
    }

    float u_value = config_.c_puct * child->prior * parent_visit_sqrt /
                    (1.0f + child->visit_count);

    float score = q_value + u_value;

    if (score > best_score) {
      best_score = score;
      best_action = move;
      best_child = child;
    }
  }

  assert(best_action != -1 && "Must select a valid action");
  assert(best_child != nullptr && "Selected child cannot be null");
  return {best_action, best_child};
}

float MCTS::Backpropagate(Node* node, float value) {
  assert(node != nullptr && "Node cannot be null");
  assert(value >= -1.0f && value <= 1.0f && "Value must be in [-1, 1]");

  node->value_sum += value;
  node->visit_count += 1;

  if (node->parent) {
    return Backpropagate(node->parent, -value);  // Flip value for opponent
  }

  return value;
}

std::pair<std::vector<float>, float> MCTS::GetPolicyValue(const GameVariant& state) {

  // move to device and forward pass
  auto device = network_->parameters().begin()->device();
  auto state_tensor = GetCanonicalBoard(state).unsqueeze(0).to(device);
  auto [policy_tensor, value_tensor] = network_->forward(state_tensor);
 
  // Verify tensor validity
  policy_tensor = policy_tensor.to(torch::kCPU).contiguous();
  value_tensor = value_tensor.to(torch::kCPU).contiguous();
  if (!policy_tensor.defined() || policy_tensor.numel() == 0) {
    throw std::runtime_error("Invalid policy tensor");
  }

  // "template accessor" is a way to access the tensor data in a type-safe manner
  auto policy_accessor = policy_tensor.template accessor<float, 2>(); 
  std::vector<float> policy(
      policy_accessor[0].data(),
      policy_accessor[0].data() + policy_accessor.size(1));

  // Get scalar value
  float value = value_tensor.item().toFloat();

  return {policy, value};
}

int MCTS::SelectMove(const GameVariant& state, float temperature) {
  auto valid_moves = GetValidMoves(state);
  auto probs = GetActionProbabilities(state, temperature);

  if (temperature == 0.0f) {
    float max_prob = -1.0f;
    int best_move = -1;
    for (int move : valid_moves) {
      if (probs[move] > max_prob) {
        max_prob = probs[move];
        best_move = move;
      }
    }
    assert(best_move != -1 && "Must find a valid move");
    return best_move;
  } else {

    std::vector<float> valid_probs;
    valid_probs.reserve(valid_moves.size());
    for (int move : valid_moves) {
      valid_probs.push_back(probs[move]);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(valid_probs.begin(),
                                         valid_probs.end());
    return valid_moves[dist(gen)];
  }
}

void MCTS::ExpandNode(Node* node, const GameVariant& state) {
  std::vector<int> valid_moves = GetValidMoves(state);

  if (node->children.empty()) {
    node->children.resize(GetActionSize(state));
  }

  auto [policy, value] = GetPolicyValue(state);

  float valid_policy_sum = 0.0f;
  for (int move : valid_moves) {
    valid_policy_sum += policy[move];
  }

  if (valid_policy_sum < 1e-6f) {
    float uniform_prob = 1.0f / valid_moves.size();
    for (int move : valid_moves) {
      policy[move] = uniform_prob;
    }
  } else {
    for (int move : valid_moves) {
      policy[move] /= valid_policy_sum;
    }
  }

  for (int move : valid_moves) {
    auto child = std::make_unique<Node>(config_);
    child->prior = policy[move];
    child->parent = node;
    child->action = move;
    child->depth = node->depth + 1;
    node->children[move] = std::move(child);
  }

  node->SetExpanded();
}

float MCTS::FullSearch(const GameVariant& state, Node* node) {
  if (IsTerminal(state)) {
    return GetGameResult(state);
  }

  if (!node->IsExpanded()) {
    ExpandNode(node, state);
  }

  float best_value = -std::numeric_limits<float>::infinity();

  for (auto& child_ptr : node->children) {
    if (!child_ptr) {
      continue;
    }
    GameVariant new_state = Clone(state);
    MakeMove(new_state, child_ptr->action);

    /// Negamax
    float child_value = -FullSearch(new_state, child_ptr.get());

    if (child_value > best_value) {
      best_value = child_value;
    }

    child_ptr->visit_count += 1;
    child_ptr->value_sum += child_value;
  }

  node->visit_count += 1;
  node->value_sum += best_value;

  return best_value;
}

void MCTS::AddDirichletNoiseToRoot(const GameVariant& state) {
  std::vector<int> valid_moves = GetValidMoves(state);
  
  // Create Dirichlet distribution for valid moves
  std::random_device rd;
  std::mt19937 gen(rd());
  std::gamma_distribution<float> gamma_dist(config_.gamma_alpha);
  
  // Sample noise for each valid move
  std::vector<float> noise;
  float noise_sum = 0.0f;
  for (int move : valid_moves) {
    float noise_sample = gamma_dist(gen);
    noise.push_back(noise_sample);
    noise_sum += noise_sample;
  } 

  const float noise_weight = 0.50f;
  for (size_t i = 0; i < valid_moves.size(); i++) {
    int move = valid_moves[i];
    if (root_->children[move]) {
      float normalized_noise = noise[i] / noise_sum;
      root_->children[move]->prior = (1.0f - noise_weight) * root_->children[move]->prior + noise_weight * normalized_noise;
    }
  }
}

} 