#include "core/mcts.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <numeric>
#include <iostream>
#include <cassert>
#include <unordered_set>
#include <mutex>    
#include <chrono>
#include "core/game.h"
#include "core/config.h"

namespace deeprlzero {

MCTS::MCTS(std::shared_ptr<NeuralNetwork> network, const Config& config)
    : network_(network), config_(config) {
    assert(network_ != nullptr && "Neural network cannot be null");
    root_ = std::make_unique<Node>(config);
}

void MCTS::ResetRoot() {
    root_ = std::make_unique<Node>(config_);
}

std::vector<float> MCTS::GetActionProbabilities(const Game* state, float temperature) {
    std::vector<float> pi(state->GetActionSize(), 0.0f);
    float sum = 0.0f;

    for (size_t i = 0; i < root_->children.size(); ++i) {
        if (root_->children[i]) {
            float count = static_cast<float>(root_->children[i]->visit_count);
            //std::cout << "Node " << i << " count: " << count << std::endl;
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

void MCTS::Search(const Game* state, Node* node) {
    std::unique_ptr<Game> mutable_state = state->Clone();
    
    if (mutable_state->IsTerminal()) {
        float value = mutable_state->GetGameResult();
        Backpropagate(node, value);
        return;
    }
    
    if (!node->IsExpanded()) {
        ExpandNode(node, mutable_state.get());
    }
    
    auto [action, child] = SelectAction(node, mutable_state.get());
    last_move_ = action;
    mutable_state->MakeMove(action);
    Search(mutable_state.get(), child);
    mutable_state->UndoMove(last_move_);
}

std::pair<int, Node*> MCTS::SelectAction(Node* node, const Game* state) {
    assert(node != nullptr && "Node cannot be null");
    assert(node->IsExpanded() && "Node must be expanded before selection");
    
    float best_score = -std::numeric_limits<float>::infinity();
    int best_action = -1;
    Node* best_child = nullptr;
    
    auto valid_moves = state->GetValidMoves();
    assert(!valid_moves.empty() && "Must have at least one valid move");
    float parent_visit_sqrt = std::sqrt(static_cast<float>(node->visit_count) + 1e-6f);
    
    for (int move : valid_moves) {
        assert(node->children[move] && "Expanded node must have all valid children");
        
        Node* child = node->children[move].get();
        
        // Q-value: Use virtual loss for unvisited nodes
        float q_value;
        if (child->visit_count == 0) {  // Update the current node's statistics.
            q_value = 0.0f;  // Default value for unvisited nodes
        } else {
            q_value = child->value_sum / child->visit_count;
        }
        
        float u_value = config_.c_puct * 
                       child->prior * 
                       parent_visit_sqrt / 
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

std::pair<std::vector<float>, float> MCTS::GetPolicyValue(const Game* state) {
    
    if (!state) {
        throw std::runtime_error("Null state in GetPolicyValue");
    }
    
    auto device = network_->parameters().begin()->device();
    auto state_tensor = state->GetCanonicalBoard().unsqueeze(0).to(device);
    auto [policy_tensor, value_tensor] = network_->forward(state_tensor);
    policy_tensor = policy_tensor.to(torch::kCPU).contiguous();
    value_tensor = value_tensor.to(torch::kCPU).contiguous();
    
    // Verify tensor validity
    if (!policy_tensor.defined() || policy_tensor.numel() == 0) {
        throw std::runtime_error("Invalid policy tensor");
    }
    
    auto policy_accessor = policy_tensor.accessor<float,2>();
    std::vector<float> policy(policy_accessor[0].data(), 
                             policy_accessor[0].data() + policy_accessor.size(1));
    
    // Get scalar value
    float value = value_tensor.item().toFloat();
    
    return {policy, value};
}

int MCTS::SelectMove(const Game* state, float temperature) {
    auto valid_moves = state->GetValidMoves();
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
        // Temperature sampling remains unchanged
        std::vector<float> valid_probs;
        valid_probs.reserve(valid_moves.size());
        for (int move : valid_moves) {
            valid_probs.push_back(probs[move]);
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<int> dist(valid_probs.begin(), valid_probs.end());
        return valid_moves[dist(gen)];
    }
}

void MCTS::ExpandNode(Node* node, const Game* state) {
    std::vector<int> valid_moves = state->GetValidMoves();
    
    if (node->children.empty()) {
        node->children.resize(state->GetActionSize());
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

float MCTS::FullSearch(const Game* state, Node *node) {
  if (state->IsTerminal()) {
    return state->GetGameResult();
  }

  if (!node->IsExpanded()) {
    ExpandNode(node, state);
  }

  float best_value = -std::numeric_limits<float>::infinity();

  for (auto &child_ptr : node->children) {
    if (!child_ptr) {
      continue;
    }
    std::unique_ptr<Game> new_state = state->Clone();
    new_state->MakeMove(child_ptr->action);

    // Using negamax: the value for the child is the negative of FullSearch on the new state.
    float child_value = -FullSearch(new_state.get(), child_ptr.get());

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

}  // namespace deeprlzero 