#ifndef ALPHAZERO_CONFIG_H_
#define ALPHAZERO_CONFIG_H_

#include <iostream>
#include <string>

namespace alphazero {

class Config {
 public:
  // Network configuration
  int num_filters = 32;
  int num_residual_blocks = 3;
  float learning_rate = 1e-3;
    
  // MCTS configuration
  int num_simulations = 100;
  float c_puct = 3.0;
  float temperature = 1.0;
    
  // Training configuration
  int batch_size = 2048;        // Increased from 32 for better GPU utilization
  int num_epochs = 100;         // Keep this value, it's already good
  int num_iterations = 25;
  int episodes_per_iteration = 25;
    
  std::string model_path = "alphazero_model.pt";

  int num_threads = 24;  // Default to max threads

  static Config ParseCommandLine(int argc, char** argv) {
    Config config;
    
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      
      if (i + 1 >= argc) {
        std::cerr << "Missing value for argument: " << arg << std::endl;
        exit(1);
      }

      if (arg == "-f" || arg == "--filters") {
        config.num_filters = std::stoi(argv[++i]);
      } else if (arg == "-r" || arg == "--residual-blocks") {
        config.num_residual_blocks = std::stoi(argv[++i]);
      } else if (arg == "-l" || arg == "--learning-rate") {
        config.learning_rate = std::stof(argv[++i]);
      } else if (arg == "-s" || arg == "--simulations") {
        config.num_simulations = std::stoi(argv[++i]);
      } else if (arg == "-p" || arg == "--cpuct") {
        config.c_puct = std::stof(argv[++i]);
      } else if (arg == "-t" || arg == "--temperature") {
        config.temperature = std::stof(argv[++i]);
      } else if (arg == "-b" || arg == "--batch-size") {
        config.batch_size = std::stoi(argv[++i]);
      } else if (arg == "-e" || arg == "--epochs") {
        config.num_epochs = std::stoi(argv[++i]);
      } else if (arg == "-i" || arg == "--iterations") {
        config.num_iterations = std::stoi(argv[++i]);
      } else if (arg == "-g" || arg == "--games") {
        config.episodes_per_iteration = std::stoi(argv[++i]);
      } else if (arg == "-m" || arg == "--model") {
        config.model_path = argv[++i];
      } else if (arg == "-h" || arg == "--help") {
        PrintUsage();
        exit(0);
      } else {
        std::cerr << "Unknown argument: " << arg << std::endl;
        PrintUsage();
        exit(1);
      }
    }
    
    return config;
  }

 private:
  static void PrintUsage() {
    std::cout << "Usage: train_alphazero [options]\n"
              << "Options:\n"
              << "  -f, --filters <n>          Number of filters (default: 32)\n"
              << "  -r, --residual-blocks <n>  Number of residual blocks (default: 3)\n"
              << "  -l, --learning-rate <f>    Learning rate (default: 0.001)\n"
              << "  -s, --simulations <n>      Number of MCTS simulations (default: 100)\n"
              << "  -p, --cpuct <f>           C_PUCT value (default: 3.0)\n"
              << "  -t, --temperature <f>      Temperature (default: 1.0)\n"
              << "  -b, --batch-size <n>      Batch size (default: 2048)\n"
              << "  -e, --epochs <n>          Number of epochs (default: 100)\n"
              << "  -i, --iterations <n>      Number of iterations (default: 25)\n"
              << "  -g, --games <n>           Games per iteration (default: 25)\n"
              << "  -m, --model <path>        Model path (default: alphazero_model.pt)\n"
              << "  -h, --help                Print this help message\n";
  }
};

}  // namespace alphazero

#endif  // ALPHAZERO_CONFIG_H_ 