#ifndef MCTS_H
#define MCTS_H

#include <cmath>
#include <memory>
#include <vector>

#include "config.h"
#include "games/variant.h"
#include "network.h"

namespace deeprlzero {

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
    ///sadly no way to get action size here
    //children.resize(config.action_size);  
  }

  float GetValue() const {
    if (visit_count == 0) return 0.0f;
    return value_sum / visit_count;
  }

  bool IsExpanded() const { return expanded; }

  bool SetExpanded(bool expanded_ = true) {
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

  std::vector<float> GetActionProbabilities(const GameVariant& state,
                                            float temperature = 1.0f);
  int SelectMove(const GameVariant& state, float temperature);

  void ResetRoot();

  Node* GetRoot() const { return root_.get(); }

  float Backpropagate(Node* node, float value);
  void Search(const GameVariant& state, Node* node);
  float FullSearch(const GameVariant& state, Node* node);
  void AddDirichletNoiseToRoot(const GameVariant& state);
  
 private:
  std::pair<int, Node*> SelectAction(Node* node, const GameVariant& state);
  void ExpandNode(Node* node, const GameVariant& state);

  std::pair<std::vector<float>, float> GetPolicyValue(const GameVariant& state);

  std::shared_ptr<NeuralNetwork> network_;
  const Config& config_;
  std::unique_ptr<Node> root_;

  // Add this member variable to track the last move
  int last_move_ = -1;

  int GetNodeDepth(Node* node) const {
    int depth = 0;
    while (node->parent != nullptr) {
      depth++;
      node = node->parent;
    }
    return depth;
  }

};

}  

#endif