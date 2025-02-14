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

MCTS::MCTS(std::shared_ptr<NeuralNetwork> network, const Config& config)
    : network_(network), config_(config) {
    if (!network_) {
        throw std::invalid_argument("Neural network cannot be null");
    }
    root_ = std::make_unique<Node>(config);
}

void MCTS::ResetRoot() {
    root_ = std::make_unique<Node>(config_);
}

std::vector<float> MCTS::GetActionProbabilities(const Game& state, float temperature) {
    if (!root_) {
        root_ = std::make_unique<Node>(config_);
    }
        
    // Create a mutable copy of the state for simulation using Clone()
    std::unique_ptr<Game> mutable_state = state.Clone();
    
    // Run simulations from current root
    for (int i = 0; i < config_.num_simulations; ++i) {
        Search(*mutable_state, root_.get());
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

void MCTS::Search(Game& state, Node* node) {
    if (state.IsTerminal()) {
        float value = state.GetGameResult();
        Backpropagate(node, value);
        return;
    }
    
    if (!node->IsExpanded()) {
        auto [policy, value] = GetPolicyValue(state);
        auto valid_moves = state.GetValidMoves();
        
        if (node->children.empty()) {
            node->children.resize(state.GetActionSize());
        }
        
        for (int move : valid_moves) {
            auto child = std::make_unique<Node>(config_);
            child->prior = policy[move];
            child->parent = node;
            child->action = move;
            node->children[move] = std::move(child);
        }
        
        Backpropagate(node, value);
        return;
    }
    
    auto [action, child] = SelectAction(node, state);
    last_move_ = action;  // Store the move
    state.MakeMove(action);
    Search(state, child);
    state.UndoMove(last_move_);  // Use the stored move when undoing
}

std::pair<int, Node*> MCTS::SelectAction(Node* node, const Game& state) {
    assert(node != nullptr && "Node cannot be null");
    assert(node->IsExpanded() && "Node must be expanded before selection");
    
    float best_score = -std::numeric_limits<float>::infinity();
    int best_action = -1;
    Node* best_child = nullptr;
    
    auto valid_moves = state.GetValidMoves();
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