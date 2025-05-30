#ifndef CONFIG_H_
#define CONFIG_H_

#include <iostream>
#include <string>
#include <cmath>

namespace deeprlzero {

class Config {
 public:
  // there are a bunch of things in here that are not really used...
  //
  // Network configuration
  int num_filters = 16;
  int num_residual_blocks = 1;
  float learning_rate = 1e-3;
  float prior_alpha = 0.75f;
  float weight_decay = 1e-4;
  // Temperature annealing
  float initial_temperature = 1.0f;
  float min_temperature = 0.1f;
  float temperature_decay = 0.95f;
  bool exhaustive_self_play = false;
  // MCTS configuration
  int num_simulations = 64;
  float c_puct = std::sqrt(2);
  float temperature = 1.5;
  int mcts_batch_size = 64;
  float gamma_alpha = 0.3f;
  float gamma_beta = 0.7f;

  // Training configuration
  int training_batch_size = 2048;
  int num_epochs = 40;
  int num_iterations = 25;
  int episodes_per_iteration = 128;

  // Evaluation configuration
  int num_evaluation_games = 32;
  float acceptance_threshold = 0.55f;

  std::string model_path = "deeprlzero_model.pt";
  std::string log_file_path = "deeprlzero_log.txt";

  int num_threads = 24;
  float l2_reg = 1e-4;
  float dropout_rate = 0.3;

  float loss_threshold = 0.25f;

  // Evaluation temperatures
  float eval_temperature = 0.0f;
  float self_play_temperature = 1.0f;

  static Config ParseCommandLine(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      // Check if we have a value for this argument
      if (i + 1 >= argc && arg != "-x" && arg != "--exhaustive" && arg != "-h" && arg != "--help") {
        std::cerr << "Missing value for argument: " << arg << std::endl;
        exit(1);
      }

      if (arg == "-f" || arg == "--filters") {
        config.num_filters = std::stoi(argv[++i]);
      } else if (arg == "-r" || arg == "--residual-blocks") {
        config.num_residual_blocks = std::stoi(argv[++i]);
      } else if (arg == "--learning-rate") {
        config.learning_rate = std::stof(argv[++i]);
      } else if (arg == "-s" || arg == "--simulations") {
        config.num_simulations = std::stoi(argv[++i]);
      } else if (arg == "-p" || arg == "--cpuct") {
        config.c_puct = std::stof(argv[++i]);
      } else if (arg == "-d" || arg == "--decay") {
        config.temperature_decay = std::stof(argv[++i]);
      } else if (arg == "-t" || arg == "--temperature") {
        config.temperature = std::stof(argv[++i]);
      } else if (arg == "-b" || arg == "--batch-size") {
        config.training_batch_size = std::stoi(argv[++i]);
      } else if (arg == "-e" || arg == "--epochs") {
        config.num_epochs = std::stoi(argv[++i]);
      } else if (arg == "-i" || arg == "--iterations") {
        config.num_iterations = std::stoi(argv[++i]);
      } else if (arg == "--games") {
        config.episodes_per_iteration = std::stoi(argv[++i]);
      } else if (arg == "-x" || arg == "--exhaustive") {
        config.exhaustive_self_play = true;
      } else if (arg == "-m" || arg == "--model") {
        config.model_path = argv[++i];
      } else if (arg == "-n" || arg == "--eval-games") {
        config.num_evaluation_games = std::stoi(argv[++i]);
      } else if (arg == "--loss-threshold") {
        config.loss_threshold = std::stof(argv[++i]);
      } else if (arg == "--threads") {
        config.num_threads = std::stoi(argv[++i]);
      } else if (arg == "-a" || arg == "--acceptance-threshold") {
        config.acceptance_threshold = std::stof(argv[++i]);
      } else if (arg == "--gamma-alpha") {
        config.gamma_alpha = std::stof(argv[++i]);
      } else if (arg == "-o" || arg == "--training-batch-size") {
        config.training_batch_size = std::stoi(argv[++i]);
      } else if (arg == "-w" || arg == "--weight-decay") {
        config.weight_decay = std::stof(argv[++i]);
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
    std::cout
        << "Usage: train_deeprlzero [options]\n"
        << "Options:\n"
        << "  -f, --filters <n>          Number of filters (default: 32)\n"
        << "  -r, --residual-blocks <n>  Number of residual blocks (default: "
           "3)\n"
        << "  -l, --learning-rate <f>    Learning rate (default: 0.001)\n"
        << "  -s, --simulations <n>      Number of MCTS simulations (default: "
           "100)\n"
        << "  -p, --cpuct <f>           C_PUCT value (default: 3.0)\n"
        << "  -t, --temperature <f>      Temperature (default: 1.0)\n"
        << "  -b, --batch-size <n>      Batch size (default: 2048)\n"
        << "  -e, --epochs <n>          Number of epochs (default: 100)\n"
        << "  -i, --iterations <n>      Number of iterations (default: 25)\n"
        << "  -g, --games <n>           Games per iteration (default: 25)\n"
        << "  -m, --model <path>        Model path (default: "
           "deeprlzero_model.pt)\n"
        << "  -n, --eval-games <n>      Number of evaluation games (default: "
           "200)\n"
        << "  -a, --acceptance-threshold <f>  Acceptance threshold (default: "
           "0.52)\n"
        << "  -h, --help                Print this help message\n";
  }
};

}  

#endif