#ifndef ALPHAZERO_MCTS_STATS_H_
#define ALPHAZERO_MCTS_STATS_H_

#include <vector>
#include <string>
#include "core/logger.h"
#include <iostream>

namespace alphazero {

class MCTSStats {
public:
    struct NodeStats {
        int depth = 0;
        int visit_count = 0;
        float q_value = 0.0f;
        float prior = 0.0f;
        float puct_score = 0.0f;
        bool was_explored = false;
    };

    void RecordNodeStats(int depth, int visits, float q_value, float prior, float puct_score, bool explored) {
        Logger& logger = Logger::GetInstance();
        stats_.push_back({depth, visits, q_value, prior, puct_score, explored});
        logger.LogFormat("Recorded stats: Depth={}, Visits={}", depth, visits);  
    }

    int GetNumSimulations() const {
        return stats_.size();
    }

    int GetNumExpandedNodes() const {
        return std::count_if(stats_.begin(), stats_.end(), [](const NodeStats& stat) {
            return stat.was_explored;
        });
    }

    [[nodiscard]] std::expected<void, Logger::Error> LogStatistics() const {
        Logger& logger = Logger::GetInstance();

        if (stats_.empty()) {
            if (auto result = logger.LogFormat("No MCTS statistics recorded."); !result) {
                return result;
            }
            return {};
        }

        // Calculate exploration statistics
        int total_nodes = stats_.size();
        int explored_nodes = 0;
        float avg_depth = 0.0f;
        float avg_visits = 0.0f;
        float avg_q_value = 0.0f;
        float avg_puct = 0.0f;
        int max_depth = 0;
        int max_visits = 0;

        for (const auto& stat : stats_) {
            if (stat.was_explored) explored_nodes++;
            avg_depth += stat.depth;
            avg_visits += stat.visit_count;
            avg_q_value += stat.q_value;
            avg_puct += stat.puct_score;
            max_depth = std::max(max_depth, stat.depth);
            max_visits = std::max(max_visits, stat.visit_count);
        }

        float exploration_rate = static_cast<float>(explored_nodes) / total_nodes;
        avg_depth /= total_nodes;
        avg_visits /= total_nodes;
        avg_q_value /= total_nodes;
        avg_puct /= total_nodes;

        // Log comprehensive statistics with error checking
        if (auto result = logger.LogFormat("MCTS Statistics Summary:"); !result) {
            return result;
        }
        if (auto result = logger.LogFormat("Total Nodes: {}", total_nodes); !result) {
            return result;
        }
        if (auto result = logger.LogFormat("Exploration Rate: {:.2f}%", exploration_rate * 100); !result) {
            return result;
        }
        if (auto result = logger.LogFormat("Average Depth: {:.2f}", avg_depth); !result) {
            return result;
        }
        if (auto result = logger.LogFormat("Max Depth: {}", max_depth); !result) {
            return result;
        }
        if (auto result = logger.LogFormat("Average Visits: {:.2f}", avg_visits); !result) {
            return result;
        }
        if (auto result = logger.LogFormat("Max Visits: {}", max_visits); !result) {
            return result;
        }
        if (auto result = logger.LogFormat("Average Q-Value: {:.3f}", avg_q_value); !result) {
            return result;
        }
        if (auto result = logger.LogFormat("Average PUCT Score: {:.3f}", avg_puct); !result) {
            return result;
        }

        return {};
    }

    void MergeStats(const MCTSStats& other) {
        stats_.insert(stats_.end(), other.stats_.begin(), other.stats_.end());
    }

    private:
    std::vector<NodeStats> stats_;
};

} // namespace alphazero

#endif // ALPHAZERO_MCTS_STATS_H_ 