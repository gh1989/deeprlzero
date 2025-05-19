#ifndef EVAL_STATS_H
#define EVAL_STATS_H

#include <string>
#include <format> 

namespace deeprlzero {

/// hold detailed evaluation outcomes.
struct EvaluationStats {
  float win_rate_first;
  float draw_rate_first;
  float loss_rate_first;

  float win_rate_second;
  float draw_rate_second;
  float loss_rate_second;

  EvaluationStats(int wins_first, int draws_first, int losses_first, int wins_second, int draws_second, int losses_second, int total_games)
    : win_rate_first(static_cast<float>(wins_first) / total_games),
      draw_rate_first(static_cast<float>(draws_first) / total_games),
      loss_rate_first(static_cast<float>(losses_first) / total_games),
      win_rate_second(static_cast<float>(wins_second) / total_games),
      draw_rate_second(static_cast<float>(draws_second) / total_games),
      loss_rate_second(static_cast<float>(losses_second) / total_games) {}

  std::string WinStats() const {
    return std::format("Moving first: Win rate: {}%, Draw rate: {}%, Loss rate: {}%\n"
                       "Moving second: Win rate: {}%, Draw rate: {}%, Loss rate: {}%\n",
                       win_rate_first * 200, draw_rate_first * 200, loss_rate_first * 200,
                       win_rate_second * 200, draw_rate_second * 200, loss_rate_second * 200);
  }

  float WinLossRatio() const {
    float draw_rate = draw_rate_first + draw_rate_second;
    float win_rate = win_rate_first + win_rate_second;
    float loss_rate = loss_rate_first + loss_rate_second;
    return (win_rate + 0.5f * draw_rate) / (loss_rate + draw_rate + win_rate);
  }

  bool IsBetterThan(const EvaluationStats& other) const {
    return WinLossRatio() > other.WinLossRatio();
  }
};

}

#endif
