#ifndef ALPHAZERO_MCTS_H
#define ALPHAZERO_MCTS_H

#include "game.h"
#include "neural_network.h"
#include "config.h"
#include <memory>
#include <vector>
#include <cmath>
#include "core/mcts_stats.h"

namespace alphazero {

struct Node {
    float value_sum = 0.0f;
    int visit_count = 0;
    bool expanded = false;
    float prior = 0.0f;
    std::vector<std::unique_ptr<Node>> children;
    Node* parent = nullptr;
    int action = -1;
    int depth = 0;

    explicit Node(const Config& config) 
        : value_sum(0.0f),
          visit_count(0),
          expanded(false),
          prior(0.0f),
          parent(nullptr),
          action(-1),
          depth(0) {
        children.resize(config.action_size);  // No second argument
    }

    float GetValue() const {
        if (visit_count == 0) return 0.0f;
        return value_sum / visit_count;
    }
    
    bool IsExpanded() const {
        return expanded;
    }

    bool SetExpanded(bool expanded_=true) {
        expanded = expanded_;
        return expanded;
    }
};

struct SearchStats {
  int num_searches = 0;
  int num_expansions = 0;
};

class MCTS {
public:
    MCTS(std::shared_ptr<NeuralNetwork> network, const Config& config);
    
    std::vector<float> GetActionProbabilities(const Game* state, float temperature = 1.0f);
    int SelectMove(const Game* state, float temperature);

    void ResetRoot();

    Node* GetRoot() const { return root_.get(); }

    // Add stats getter
    const MCTSStats& GetStats() const { return stats_; }
    void ClearStats() { stats_ = MCTSStats(); }
    float Backpropagate(Node* node, float value);
    void Search(const Game* state, Node* node);
    float FullSearch(const Game* state, Node* node);

private:

    std::pair<int, Node*> SelectAction(Node* node, const Game* state);
    void ExpandNode(Node* node, const Game* state);

    std::pair<std::vector<float>, float> GetPolicyValue(const Game* state);
    
    std::shared_ptr<NeuralNetwork> network_;
    const Config& config_;
    std::unique_ptr<Node> root_;
    
    // Add this member variable to track the last move
    int last_move_ = -1;
    MCTSStats stats_;
        
    int GetNodeDepth(Node* node) const {
        int depth = 0;
        while (node->parent != nullptr) {
            depth++;
            node = node->parent;
        }
        return depth;
    }
};

} // namespace alphazero

#endif // ALPHAZERO_MCTS_H 