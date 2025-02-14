#ifndef ALPHAZERO_NETWORK_MANAGER_H_
#define ALPHAZERO_NETWORK_MANAGER_H_

#include "core/neural_network.h"
#include "core/config.h"
#include <memory>
#include <string>

namespace alphazero {

class NetworkManager {
 public:
  explicit NetworkManager(const Config& config);
  
  std::shared_ptr<NeuralNetwork> CreateInitialNetwork();
  void SetBestNetwork(std::shared_ptr<NeuralNetwork> network);
  bool LoadBestNetwork();
    
  bool AcceptNewNetwork(std::shared_ptr<NeuralNetwork> network, float win_rate);
  void UpdateTemperature();
  float GetCurrentTemperature() const { return current_temperature_; }
  std::shared_ptr<NeuralNetwork> GetBestNetwork() const { return best_network_; }
  
  void SaveBestNetwork() const;
  
 private:
  const Config& config_;
  std::shared_ptr<NeuralNetwork> best_network_;
  float best_win_rate_ = 0.0f;
  float current_temperature_;
  int best_iteration_ = 0;
  int current_iteration_ = 0;
};

}  // namespace alphazero

#endif  // ALPHAZERO_NETWORK_MANAGER_H_ 