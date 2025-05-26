#ifndef GAMES_VARIANT_H
#define GAMES_VARIANT_H

#include <variant>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <torch/torch.h>

#include "concepts.h"
#include "traits.h"

#include "tictactoe.h"
#include "chess.h"

namespace deeprlzero {

// Type-erased variant of all game types
using GameVariant = std::variant<TicTacToe, Chess>;

// State observation
template <typename G>
requires GameConcept<G>
inline std::vector<int> GetValidMoves(const G& game) {
    return game.GetValidMoves();
}

inline std::vector<int> GetValidMoves(const GameVariant& game) {
    return std::visit([](auto&& arg) { return arg.GetValidMoves(); }, game);
}

template <typename G>
requires GameConcept<G>
inline int GetNumActions(const G& game) {
    return game.GetNumActions();
}

inline int GetNumActions(const GameVariant& game) {
    return std::visit([](auto&& arg) { return arg.GetNumActions(); }, game);
}

template <typename G>
requires GameConcept<G>
inline std::string GetBoardString(const G& game) {
    return game.GetBoardString();
}

inline std::string GetBoardString(const GameVariant& game) {
    return std::visit([](auto&& arg) { return arg.GetBoardString(); }, game);
}

template <typename G>
requires GameConcept<G>
inline void SetFromString(G& game, const std::string& str, int player) {
    game.SetFromString(str, player);
}

inline void SetFromString(GameVariant& game, const std::string& str, int player) {
    std::visit([&](auto&& arg) { arg.SetFromString(str, player); }, game);
}

template <typename G>
requires GameConcept<G>
inline float GetGameResult(const G& game) {
    return game.GetGameResult();
}

inline float GetGameResult(const GameVariant& game) {
    return std::visit([](auto&& arg) { return arg.GetGameResult(); }, game);
}

template <typename G>
requires GameConcept<G>
inline bool IsTerminal(const G& game) {
    return game.IsTerminal();
}

inline bool IsTerminal(const GameVariant& game) {
    return std::visit([](auto&& arg) { return arg.IsTerminal(); }, game);
}

template <typename G>
requires GameConcept<G>
inline bool CheckWin(const G& game, int player) {
    return game.CheckWin(player);
}

inline bool CheckWin(const GameVariant& game, int player) {
    return std::visit([&](auto&& arg) { return arg.CheckWin(player); }, game);
}

template <typename G>
requires GameConcept<G>
inline torch::Tensor GetCanonicalBoard(const G& game) {
    return game.GetCanonicalBoard();
}

inline torch::Tensor GetCanonicalBoard(const GameVariant& game) {
    return std::visit([](auto&& arg) { return arg.GetCanonicalBoard(); }, game);
}   

template <typename G>
requires GameConcept<G>
inline std::string ToString(const G& game) {
    return game.ToString();
}

inline std::string ToString(const GameVariant& game) {
    return std::visit([](auto&& arg) { return arg.ToString(); }, game);
}

template <typename G>
requires GameConcept<G>
inline int GetActionSize(const G& game) {
    return game.GetActionSize();
}

inline int GetActionSize(const GameVariant& game) {
    return std::visit([](auto&& arg) { return arg.GetActionSize(); }, game);
}

template <typename G>
requires GameConcept<G>
inline int GetInputChannels(const G& game) {
    return game.GetInputChannels();
}

inline int GetInputChannels(const GameVariant& game) {
    return std::visit([](auto&& arg) { return arg.GetInputChannels(); }, game);
}

// Game play helpers
template <typename G>
requires GameConcept<G>
inline void MakeMove(G& game, int move) {
    game.MakeMove(move);
}

inline void MakeMove(GameVariant& game, int move) {
    std::visit([&](auto&& arg) { arg.MakeMove(move); }, game);
}

template <typename G>
requires GameConcept<G>
inline G Clone(const G& game) {
    return game.Clone();
}

inline GameVariant Clone(const GameVariant& game) {
    return std::visit([](const auto& g) { return GameVariant(g.Clone()); }, game);
}   

template <typename G>
requires GameConcept<G>
inline void UndoMove(G& game, int move) {
    game.UndoMove(move);
}

inline void UndoMove(GameVariant& game, int move) {
    std::visit([&](auto&& arg) { arg.UndoMove(move); }, game);
}   

template <typename G>
requires GameConcept<G>
inline void Reset(G& game) {
    game.Reset();
}

inline void Reset(GameVariant& game) {
    std::visit([](auto&& arg) { arg.Reset(); }, game);
}

template <typename G>
requires GameConcept<G>
inline int GetCurrentPlayer(const G& game) {
    return game.GetCurrentPlayer();
}

inline int GetCurrentPlayer(const GameVariant& game) {
    return std::visit([](auto&& arg) { return arg.GetCurrentPlayer(); }, game);
}   

template <typename T>
requires GameConcept<T>
inline bool IsGameType(const GameVariant& game) {
    return std::holds_alternative<T>(game);
}

template <typename GameType>
requires GameConcept<GameType>
inline GameVariant CreateGame() {
    return GameVariant(GameType());
}

template <typename Func>
requires std::invocable<Func, GameVariant>
auto VisitGame(const GameVariant& game, Func&& func) {
    return std::visit(std::forward<Func>(func), game);
}

}
