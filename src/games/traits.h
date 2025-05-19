#ifndef GAMES_GAME_TRAITS_H
#define GAMES_GAME_TRAITS_H

#include <array>

namespace deeprlzero {

// Forward declarations of game classes
class TicTacToe;
class Chess;
class Connect4;

// Base traits template (not used directly)
template <typename GameT>
struct GameTraits;

// TicTacToe traits specialization
template <>
struct GameTraits<TicTacToe> {
    static constexpr int kBoardSize = 3;
    static constexpr int kNumActions = 9;
    static constexpr int kNumChannels = 3;
    
    using BoardType = std::array<std::array<int, kBoardSize>, kBoardSize>;
};

// Chess traits specialization
template <>
struct GameTraits<Chess> {
    static constexpr int kBoardSize = 8;
    static constexpr int kNumActions = 8*8*73;
    static constexpr int kNumChannels = 3;
    
    using BoardType = std::array<std::array<int, kBoardSize>, kBoardSize>;
};

// Connect4 traits specialization
template <>
struct GameTraits<Connect4> {
    static constexpr int kBoardSize = 7;
    static constexpr int kNumActions = 7;
    static constexpr int kNumChannels = 1;
    
    using BoardType = std::array<std::array<int, kBoardSize>, kBoardSize>;
};

}

#endif