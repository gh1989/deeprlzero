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
#include "core/game.h"
#include "core/tictactoe.h"
#include "core/config.h"

namespace alphazero {

MCTS::MCTS(std::shared_ptr<NeuralNetwork> network, const Config& config)
    : network_(network), config_(config) {
    assert(network_ != nullptr && "Neural network cannot be null");
    root_ = std::make_unique<Node>(config);
}

void MCTS::ResetRoot() {
    root_ = std::make_unique<Node>(config_);
}

std::vector<float> MCTS::GetActionProbabilities(const Game* state, float temperature) {
    // Run simulations from current root
    for (int i = 0; i < config_.num_simulations; ++i) {
        Search(state, root_.get());
    }
        
    std::vector<float> action_probs(state->GetActionSize(), 0.0f);
    float visit_sum = 0.0f;
    
    // Sum up visit counts
    for (size_t action = 0; action < root_->children.size(); ++action) {
        if (root_->children[action]) {
            action_probs[action] = root_->children[action]->visit_count;
            visit_sum += action_probs[action];
        }
    }
    
    if (visit_sum <= 0.0f) {
        throw std::runtime_error("No visits recorded in MCTS search. Root expanded: " + 
                               std::to_string(root_->IsExpanded()));
    }
    
    // Normalize probabilities
    for (float& prob : action_probs) {
        prob /= visit_sum;
    }
    
    return action_probs;
}

void MCTS::Search(const Game* state, Node* node) {
    // We need to create a mutable copy of the state for this search
    std::unique_ptr<Game> mutable_state = state->Clone();
    
    if (mutable_state->IsTerminal()) {
        float value = mutable_state->GetGameResult();
        Backpropagate(node, value);
        return;
    }
    
    if (!node->IsExpanded()) {
        ExpandNode(node, mutable_state.get());
        return;
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
        
    float sqrt_total = std::sqrt(static_cast<float>(node->visit_count + 1));  // Add 1 to prevent division by zero
    
    for (int move : valid_moves) {
        if (!node->children[move]) {
            continue;
        }
        
        Node* child = node->children[move].get();
        float q_value = child->visit_count > 0 ? child->GetValue() : 0.0f;
        
        // Add noise to root node for exploration
        float prior = child->prior;
        if (node == root_.get()) {
            static thread_local std::mt19937 gen(std::random_device{}());
            std::gamma_distribution<float> gamma(config_.gamma_alpha, config_.gamma_beta);
            float noise = gamma(gen);
            prior = config_.prior_alpha * prior + (1 - config_.prior_alpha) * noise;
        }
        
        float u_value = config_.c_puct * prior * sqrt_total / (1 + child->visit_count);
        float score = q_value + u_value;
        
        if (score > best_score) {
            best_score = score;
            best_action = move;
            best_child = child;
        }
    }
    
    // If no existing children were selected, create a new child for the first valid move
    if (best_action == -1) {
        // std::cout << "Creating new child for first valid move" << std::endl;
        int move = valid_moves[0];
        auto child = std::make_unique<Node>(config_);
        child->parent = node;
        child->action = move;
        child->prior = 1.0f / valid_moves.size();
        best_child = child.get();
        node->children[move] = std::move(child);
        best_action = move;
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
    static std::once_flag device_flag;
    static const torch::Device device(torch::kCPU);
    
    // Move network to CPU once and set to eval mode
    std::call_once(device_flag, [this]() {
        network_->to(device);
        network_->eval();
    });
    
    // Get input tensor (ensuring it's on CPU)
    torch::Tensor board = state->GetCanonicalBoard().to(device);
    
    // Forward pass (batched)
    torch::NoGradGuard no_grad;
    auto result = network_->forward(board.unsqueeze(0));
    torch::Tensor raw_policy = result.first;
    torch::Tensor raw_value = result.second;

    // Ensure the tensor is contiguous before accessing its data pointer.
    raw_policy = raw_policy.contiguous();

    // Convert to required format.
    std::vector<float> policy_vec(raw_policy.data_ptr<float>(),
                                  raw_policy.data_ptr<float>() + raw_policy.numel());
    float value_scalar = raw_value.item<float>();
    
    return {policy_vec, value_scalar};
}

int MCTS::SelectMove(const Game* state, float temperature) {
    auto valid_moves = state->GetValidMoves();
    auto probs = GetActionProbabilities(state, temperature);
      
    if (temperature == 0.0f) {
        // Find the highest probability among valid moves
        float max_prob = 0.0f;
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
        // Normalize probabilities for valid moves only
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

float MCTS::FullSearch(const Game* state, Node *node) {
  // If we're at a terminal state, return the actual game result.
  if (state->IsTerminal()) {
    return state->GetGameResult();
  }

  // If the node hasn't been expanded yet, expand it.
  if (!node->IsExpanded()) {
    ExpandNode(node, state);
  }

  // Initialize best_value to negative infinity.
  float best_value = -std::numeric_limits<float>::infinity();

  // Iterate over all children to perform an exhaustive search.
  for (auto &child_ptr : node->children) {
    if (!child_ptr) {
      continue;
    }
    // Clone the current state so that we can apply the child's move without side effects.
    std::unique_ptr<Game> new_state = state->Clone();
    new_state->MakeMove(child_ptr->action);

    // Using negamax: the value for the child is the negative of FullSearch on the new state.
    float child_value = -FullSearch(new_state.get(), child_ptr.get());

    // Update best_value with the maximum over the children.
    if (child_value > best_value) {
      best_value = child_value;
    }

    // Optionally update the child statistics as part of backpropagation.
    child_ptr->visit_count += 1;
    child_ptr->value_sum += child_value;
  }

  // Update the current node's statistics.
  node->visit_count += 1;
  node->value_sum += best_value;

  return best_value;
}

void MCTS::ExpandNode(Node* node, const Game* state) {
    // Get all valid moves from the current game state.
    std::vector<int> valid_moves = state->GetValidMoves();
    
    // Ensure the children vector is resized to the action space size.
    if (node->children.empty()) {
        node->children.resize(state->GetActionSize());
    }
    
    // Get the policy 
    auto [policy, _] = GetPolicyValue(state);
    
    // Create new child nodes for each valid move.
    for (int move : valid_moves) {
        auto child = std::make_unique<Node>(config_);
        child->prior = policy[move];
        child->parent = node;
        child->action = move;
        child->depth = node->depth + 1;
        node->children[move] = std::move(child);
    }
}

}  // namespace alphazero 