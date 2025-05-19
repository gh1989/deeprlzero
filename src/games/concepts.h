#ifndef GAMES_GAME_CONCEPTS_H
#define GAMES_GAME_CONCEPTS_H

#include <concepts>
#include <memory>
#include <vector>
#include <torch/torch.h>

namespace deeprlzero {

template <typename T>
concept GameConcept = requires(T game, int player, int move) {
    { game.GetValidMoves() } -> std::convertible_to<std::vector<int>>;
    { game.GetBoardString() } -> std::convertible_to<std::string>;
    { game.SetFromString(std::string(), 1) } -> std::same_as<void>;
    { game.GetGameResult() } -> std::convertible_to<float>;
    { game.IsTerminal() } -> std::convertible_to<bool>;
    { game.CheckWin(player) } -> std::convertible_to<bool>;
    { game.GetCanonicalBoard() } -> std::convertible_to<torch::Tensor>;
    { game.ToString() } -> std::convertible_to<std::string>;
    { game.GetActionSize() } -> std::convertible_to<int>;
    { game.GetInputChannels() } -> std::convertible_to<int>;
    { game.Reset() } -> std::same_as<void>;
    { game.Clone() } -> std::convertible_to<T>;
    { game.GetCurrentPlayer() } -> std::convertible_to<int>;
    { game.MakeMove(move) } -> std::same_as<void>;
    { game.UndoMove(move) } -> std::same_as<void>;
};

// Game network traits
template <typename T>
concept GameTraitsConcept = requires {
    { T::kBoardSize } -> std::convertible_to<int>;
    { T::kNumActions } -> std::convertible_to<int>;
    { T::kNumChannels } -> std::convertible_to<int>;
    typename T::BoardType;
};

}

#endif