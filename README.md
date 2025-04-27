Build:
```
mkdir build
cd build
cmake ..
make
```

Run:
```
./deeprlzero -f 32 -r 3 -l 0.001 -s 100 -p 3.0 -t 1.0 -b 2048 -e 100 -i 25 -g 25 -m deeprlzero_model.pt -n 200 -a 0.52
```

Options:
-f, --filters <n> Number of filters (default: 32)
-r, --residual-blocks <n> Number of residual blocks (default: 3)
-l, --learning-rate <f> Learning rate (default: 0.001)
-s, --simulations <n> Number of MCTS simulations (default: 100)
-p, --cpuct <f> C_PUCT value (default: 3.0)
-t, --temperature <f> Temperature (default: 1.0)
-b, --batch-size <n> Batch size (default: 2048)
-e, --epochs <n> Number of epochs (default: 100)
-i, --iterations <n> Number of iterations (default: 25)
-g, --games <n> Games per iteration (default: 25)
-m, --model <path> Model path (default: deeprlzero_model.pt)
-n, --eval-games <n> Number of evaluation games (default: 200)
-a, --acceptance-threshold <f> Acceptance threshold (default: 0.52)
-h, --help Print this help message