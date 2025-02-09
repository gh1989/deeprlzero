#ifndef ALPHAZERO_MCTS_H
#define ALPHAZERO_MCTS_H

#include "game.h"
#include "neural_network.h"
#include <memory>
#include <vector>
#include <cmath>

namespace alphazero {

struct Node {
    float value_sum = 0.0f;
    int visit_count = 0;
    float prior = 0.0f;
    std::vector<std::unique_ptr<Node>> children;
    Node* parent = nullptr;
    int action = -1;
    
    Node() : parent(nullptr), visit_count(0), value_sum(0.0f), prior(0.0f), action(-1) {
        children.resize(9); // TicTacToe specific (!)
    }

    float GetValue() const {
        if (visit_count == 0) return 0.0f;
        return value_sum / visit_count;
    }
    
    bool IsExpanded() const {
        for (const auto& child : children) {
            if (child) return true;
        }
        return false;
    }
};

class MCTS {
public:
    MCTS(std::shared_ptr<NeuralNetwork> network, float c_puct, int num_simulations);
    
    std::vector<float> GetActionProbabilities(const Game& state, float temperature = 1.0f);
    int SelectMove(const Game& state, float temperature = 0.0f);

    void ResetRoot();

    // Add batch processing support
    static constexpr int BATCH_SIZE = 64;  // Adjust based on your GPU
    std::vector<Game*> evaluation_queue_;
    
    // Batch evaluation method
    std::vector<std::pair<std::vector<float>, float>> 
    BatchEvaluate(const std::vector<Game*>& states);

private:
    void Search(const Game& state, Node* node);
    std::pair<int, Node*> SelectAction(Node* node, const Game& state);
    void ExpandNode(Node* node, const Game& state);
    float Backpropagate(Node* node, float value);
    std::pair<std::vector<float>, float> GetPolicyValue(const Game& state);
    
    std::shared_ptr<NeuralNetwork> network_;
    float c_puct_;
    int num_simulations_;
    std::unique_ptr<Node> root_;
};

} // namespace alphazero

#endif // ALPHAZERO_MCTS_H 