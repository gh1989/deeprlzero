#include "core/game.h"

namespace alphazero {

std::vector<int> Game::GetValidMoves() const {
    return std::vector<int>();
}

void Game::MakeMove(int move) {}

float Game::GetGameResult() const {
    return 0.0f;
}

bool Game::IsTerminal() const {
    return false;
}

int Game::GetActionSize() const {
    return 0;
}

int Game::GetInputChannels() const {
    return 0;
}

}  // namespace alphazero 