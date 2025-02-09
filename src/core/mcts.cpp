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

namespace alphazero {

MCTS::MCTS(std::shared_ptr<NeuralNetwork> network, float c_puct, int num_simulations)
    : network_(network), c_puct_(c_puct), num_simulations_(num_simulations) {
    if (!network_) {
        throw std::invalid_argument("Neural network cannot be null");
    }
    if (num_simulations_ <= 0) {
        throw std::invalid_argument("Number of simulations must be positive");
    }
    if (c_puct_ <= 0) {
        throw std::invalid_argument("c_puct must be positive");
    }
    root_ = std::make_unique<Node>();
}

void MCTS::ResetRoot() {
    root_ = std::make_unique<Node>();
}

std::vector<float> MCTS::GetActionProbabilities(const Game& state, float temperature) {
    if (!root_) {
        root_ = std::make_unique<Node>();
    }
        
    // Run simulations from current root
    for (int i = 0; i < num_simulations_; ++i) {
        Search(state, root_.get());
    }
        
    std::vector<float> action_probs(state.GetActionSize(), 0.0f);
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

void MCTS::Search(const Game& state, Node* node) {
    assert(node != nullptr && "Node cannot be null");
    
    if (state.IsTerminal()) {
        float value = state.GetGameResult();
        Backpropagate(node, value);
        return;
    }
    
    if (!node->IsExpanded()) {
        // Get policy and value from neural network
        auto [policy, value] = GetPolicyValue(state);
        auto valid_moves = state.GetValidMoves();
          
        assert(!valid_moves.empty() && "Must have at least one valid move");
        assert(policy.size() == state.GetActionSize() && "Policy size must match action size");
        
        // Initialize children vector if needed
        if (node->children.empty()) {
            node->children.resize(state.GetActionSize());
        }
        
        // Create child nodes
        for (int move : valid_moves) {
            assert(move >= 0 && move < state.GetActionSize() && "Move must be within valid range");
            auto child = std::make_unique<Node>();
            child->prior = policy[move];
            child->parent = node;
            child->action = move;
            node->children[move] = std::move(child);
        }
        
        // Increment visit count for expansion
        node->visit_count++;
        
        // Select a child to visit
        auto [action, child] = SelectAction(node, state);
        auto next_state = state.Clone();
        next_state->MakeMove(action);
        Search(*next_state, child);
        return;
    }
    
    // Select action according to PUCT formula
    auto [action, child] = SelectAction(node, state);
    assert(child != nullptr && "Selected child cannot be null");
    assert(action >= 0 && action < state.GetActionSize() && "Selected action must be valid");
    
    auto next_state = state.Clone();
    assert(next_state != nullptr && "Cloned state cannot be null");
    next_state->MakeMove(action);
    Search(*next_state, child);
}

std::pair<int, Node*> MCTS::SelectAction(Node* node, const Game& state) {
    assert(node != nullptr && "Node cannot be null");
    assert(node->IsExpanded() && "Node must be expanded before selection");
    
    float best_score = -std::numeric_limits<float>::infinity();
    int best_action = -1;
    Node* best_child = nullptr;
    
    auto valid_moves = state.GetValidMoves();
    assert(!valid_moves.empty() && "Must have at least one valid move");
        
    float sqrt_total = std::sqrt(static_cast<float>(node->visit_count));
    
    // First pass: try to select from existing children
    for (int move : valid_moves) {
        if (!node->children[move]) {
            continue;
        }
        
        Node* child = node->children[move].get();
        float q_value = child->visit_count > 0 ? child->GetValue() : 0.0f;
        float u_value = c_puct_ * child->prior * sqrt_total / (1 + child->visit_count);
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
        auto child = std::make_unique<Node>();
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

std::pair<std::vector<float>, float> MCTS::GetPolicyValue(const Game& state) {
    static std::once_flag device_flag;
    static const torch::Device device(torch::kCPU);
    
    // Move network to CPU once and set to eval mode
    std::call_once(device_flag, [this]() {
        network_->to(device);
        network_->eval();
    });
    
    // Get input tensor (ensuring it's on CPU)
    torch::Tensor board = state.GetCanonicalBoard().to(device);
    
    // Forward pass (batched)
    torch::NoGradGuard no_grad;
    auto [policy, value] = network_->forward(board.unsqueeze(0));
    
    // Convert to required format
    std::vector<float> policy_vec(policy.data_ptr<float>(), 
                                policy.data_ptr<float>() + policy.numel());
    float value_scalar = value.item<float>();
    
    return {policy_vec, value_scalar};
}

int MCTS::SelectMove(const Game& state, float temperature) {
    auto valid_moves = state.GetValidMoves();
    auto probs = GetActionProbabilities(state, temperature);
      
    // Create a set for O(1) lookup
    std::unordered_set<int> valid_move_set(valid_moves.begin(), valid_moves.end());
    
    if (temperature == 0.0f) {
        // Find the highest probability among valid moves
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

}  // namespace alphazero 