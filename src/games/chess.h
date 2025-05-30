#ifndef GAMES_CHESS_H
#define GAMES_CHESS_H

#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <bit>
#include <iterator> 
#include <functional>
#include <vector>
#include <map>
#include <numeric>
#include <list>
#include <string>
#include <array>   
#include <memory>
#include <sstream>
#include <unordered_map>
#include <torch/torch.h>

#include "concepts.h"
#include "traits.h"

namespace deeprlzero {

class State;
using Move = unsigned short;

class Chess {
public:
    using Traits = GameTraits<Chess>;
    
    Chess();
    std::vector<int> GetValidMoves() const;
    void MakeMove(int action_index);
    float GetGameResult() const;
    bool IsTerminal() const;
    bool CheckWin(int player) const;
    torch::Tensor GetCanonicalBoard() const;
    Chess Clone() const;
    void UndoMove(int action_index);
    void Reset();
    int GetCurrentPlayer() const;
    
    // static Methods from traits
    static int GetActionSize() { return Traits::kNumActions; }
    static int GetInputChannels() { return Traits::kNumChannels; }
    static int GetNumActions() { return Traits::kNumActions; }
    static int GetBoardSize() { return Traits::kBoardSize; }
    
    // Display methods
    std::string ToString() const;
    std::string GetBoardString() const;
    void SetFromString(const std::string& fen_str, int player=-1);

    // Helper methods
    int MoveToIndex(const Move& move) const;
    Move IndexToMove(int action_index) const;
    bool IsCheckmate() const;
    bool IsStalemate() const;
    bool IsDraw() const;
    void UpdateGameStatus();

private:
    std::unique_ptr<State> state_;
    std::vector<Move> move_history_;
    bool game_over_ = false;
    int result_ = 0;
};

static_assert(GameConcept<Chess>);

/// Implementations
/// Bitboards and iteration over them
struct Bitboard
{
  Bitboard() : bit_number(0) {}
  Bitboard(uint64_t bit_number_) : bit_number(bit_number_) {}
  bool operator==(const Bitboard& other) const {return bit_number == other.bit_number; }
  Bitboard operator<<(int x) const { return Bitboard(bit_number << x); }
  Bitboard operator~() const { return Bitboard(~bit_number); }
  Bitboard operator>>(int x) const { return Bitboard(bit_number >> x); }
  Bitboard operator| (const Bitboard& other) const { return Bitboard(bit_number | other.bit_number); }
  Bitboard& operator|=(const Bitboard& other) { bit_number |= other.bit_number; return *this; }
  Bitboard operator& (const Bitboard& other) const { return Bitboard(bit_number & other.bit_number); }
  Bitboard operator^ (const Bitboard& other) const { return Bitboard(bit_number ^ other.bit_number); }
  Bitboard operator- (int value) const { return Bitboard(bit_number - value); }
  void operator &= (const Bitboard& other) { bit_number &= other.bit_number; }
  operator bool() const { return bit_number != 0; }
  uint64_t PopCnt() const { return std::popcount(bit_number); }
  uint8_t nMSB() const {
    if (bit_number == 0) return 0;
    uint8_t position = 63;
    while ((bit_number & (1ULL << position)) == 0) position--;  
    return position;
  }
  uint8_t nLSB() const {
    if (bit_number == 0) return 64;
    uint8_t position = 0;
    while ((bit_number & (1ULL << position)) == 0) position++;
    return position;
  }
  uint64_t bit_number;
};
/// ...Iterations
class BitIterator {
public:
    BitIterator(Bitboard value, uint8_t index) : value_(value), index_(index) {};
    bool operator!=(const BitIterator& other) const {
        return (value_ != other.value_) || (index_ != other.index_);
    }
  void operator++() {
      value_ &= (value_ - 1);
      if (value_) {
          index_ = value_.nLSB();
      } else {
          index_ = 64;
      }
}  

  unsigned int operator*() const { return index_; }
private:
  Bitboard value_;
  uint8_t index_;
};
/// ...Range
class BitboardRange {
public:
    BitboardRange(Bitboard bb) : bb_(bb) {};
    BitIterator begin() const { return BitIterator(bb_, bb_.nLSB()); }
    BitIterator end() const { return BitIterator(0, 64); }
private:
    Bitboard bb_;
};

/// Introduce squares, pieces, properties of moves
enum Square
{
  a1 = 0, b1, c1, d1, e1, f1, g1, h1,
  a2, b2, c2, d2, e2, f2, g2, h2,
  a3, b3, c3, d3, e3, f3, g3, h3,
  a4, b4, c4, d4, e4, f4, g4, h4,
  a5, b5, c5, d5, e5, f5, g5, h5,
  a6, b6, c6, d6, e6, f6, g6, h6,
  a7, b7, c7, d7, e7, f7, g7, h7,
  a8, b8, c8, d8, e8, f8, g8, h8,
};

enum Piece { KNIGHT, BISHOP, ROOK, QUEEN, KING, PAWN, NUMBER_PIECES, NO_PIECE = -1 };
enum Castling { WQ = 1, WK = 2, BQ = 4, BK = 8, ALL = 15 };
enum SpecialMove { NONE = 0, ENPASSANT = 1, CASTLE = 2, PROMOTE = 3 };

/// Squares interfacing with bitboards
Bitboard GetSquare(Square sqr);
template<typename Square, typename... SqArgs>
static Bitboard GetSquare(Square sqr, SqArgs... others) { return GetSquare(sqr) | GetSquare(others...); }
Bitboard BitMove(Bitboard bb, Square from, Square to);
bool IsOn(Bitboard bb, Square square);
Bitboard OffBit(Bitboard bb, Square off);
Bitboard OnBit(Bitboard bb, Square on);
Square BbSqr(Bitboard bb);
Bitboard SqrBb(Square sqr);

/// ... Moves
typedef unsigned short Move;
constexpr unsigned short to_bits = 6;
constexpr unsigned short flag_bits = 12;
constexpr unsigned short prom_bits = 14;
constexpr unsigned short from_mask = 63;
constexpr unsigned short to_mask = 63 << to_bits;
constexpr unsigned short flag_mask = 3 << flag_bits;
constexpr unsigned short prom_mask = 3 << prom_bits;

/// Operations on moves
Square GetFrom(Move move);
Square GetTo(Move move);
SpecialMove SpecialMoveType(Move move);
Piece PromotionPiece(Move move);
Move ReflectMove(Move move);
Move CreateMove(Square from, Square to);
Move CreatePromotion(Square from, Square to, Piece promo);
Move CreateEnPassant(Square from, Square to);
Move CreateCastle(Square from, Square to);
Move MoveFromUci(const std::string& uci_move);

/// The game's state.
class State
{
private:
    Bitboard bbs[13];
    uint8_t c50;
    uint8_t castle;
    uint32_t plies;
    bool blackMove;

public:
    void Apply(Move move);
  bool operator==(const State& other) const {
    return std::equal(std::begin(bbs), std::end(bbs), std::begin(other.bbs)) &&
        c50 == other.c50 &&
        castle == other.castle && 
        plies == other.plies &&
        blackMove == other.blackMove;
  }

    Bitboard getBitboard(int index) const { return bbs[index]; }
    void setBitboard(int index, Bitboard value) { bbs[index] = value; }
    void addToBitboard(int index, Bitboard toAdd) { bbs[index] |= toAdd; }
    uint8_t getMoveCount() const { return plies / 2; }
    void setMoveCount(uint8_t value) { plies = value; }
    uint8_t getCastleRights() const { return castle; }
    void setCastleRights(uint8_t value) { castle = value; }
    Bitboard getEnPassant() const { return bbs[12]; }
    void setEnPassant(Bitboard value) { bbs[12] = value; }
    uint8_t getFiftyMove() const { return c50; }
    void setFiftyMove(uint8_t value) { c50 = value; }
    void setTurn(bool value) { blackMove = value; }
    uint32_t getPlies() const { return plies; }
    void setPlies(uint32_t value) { plies = value; }
    uint8_t get50MoveCount() const { return c50; }
    bool isBlackMove() const { return blackMove; }
    void setBlackMove(bool value) { blackMove = value; }
    Bitboard enemyOccupation() const { return blackMove ? whiteOccupation() : blackOccupation(); };
    Bitboard moveOccupation() const { return blackMove ? blackOccupation() : whiteOccupation(); }
    Bitboard blackOccupation() const { return combinedBoards(bbs+NUMBER_PIECES, bbs+NUMBER_PIECES*2); }
    Bitboard whiteOccupation() const { return combinedBoards(bbs, bbs+NUMBER_PIECES); }
    template<Piece _Piece> 
    Bitboard movePiece() const   { return bbs[moveBitStart() + _Piece];}
    template<Piece _Piece>
    Bitboard enemyPiece() const { return bbs[enemyBitStart() + _Piece];  }
    template<Piece _Piece>
    Bitboard whitePiece() const { return bbs[_Piece]; }
    template<Piece _Piece>
    Bitboard blackPiece() const { return bbs[NUMBER_PIECES+_Piece]; }

private:
  uint8_t moveBitStart() const { return blackMove ? NUMBER_PIECES : 0; };
  uint8_t moveBitEnd() const { return blackMove ? 2*NUMBER_PIECES : NUMBER_PIECES; };
  uint8_t enemyBitStart() const { return blackMove ? 0 : NUMBER_PIECES; };
  uint8_t enemyBitEnd() const { return blackMove ? NUMBER_PIECES : 2*NUMBER_PIECES; };
  Bitboard combinedBoards(auto begin, auto end) const {
    auto bit_or = [&](const Bitboard &a, const Bitboard &b) { return a | b; };
    return std::accumulate<>(begin, end, Bitboard(0), bit_or);
  }
};

/// String operations and UCI
const std::string piece_strings[6] = { "n", "b", "r", "q", "k", "p" };
std::string PieceStringLower(Piece piece);
std::string SquareName(Square sqr);
Bitboard BitboardFromString(std::string str);
std::string AsUci(Move move);
void PrettyPrint(const State& state);
State StateFromFen(std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
std::string StateToFen(const State& state);

/// The generation of moves.
using TMoveContainer = std::list<Move>;
TMoveContainer GenerateMoves(const State&);
void PawnMoves(
  const State& state,
  TMoveContainer &moves);
template <Piece _Piece>
void JumperMoves(
    const State& state,
  TMoveContainer &moves);
template <Piece _Piece>
void KnightMoves(
    const State& state,
  TMoveContainer &moves);
void KingCastling(
  const State& state,
  TMoveContainer &moves);

/// Helpers and utilities for move generation.
bool SquareConnectedToBitboard(
  Square source,
  Bitboard target,
  Bitboard obstacle,
  const std::array<std::pair<int, int>, 4>& directions);

inline Square Reflect(Square sqr)
{
  int square_int = static_cast<int>(sqr);
  square_int = (square_int % 8) + (7 - (square_int / 8)) * 8;
  Square output = static_cast<Square>(square_int);
  return output;
}

static inline const std::array<std::pair<int, int>, 4> rook_directions {
  std::make_pair(1, 0), std::make_pair(-1, 0), std::make_pair(0, 1), std::make_pair(0, -1)
};

static inline const std::array<std::pair<int, int>, 4> bishop_directions {
  std::make_pair(1, 1), std::make_pair(-1, 1), std::make_pair(1, -1), std::make_pair(-1, -1)
};

inline bool isInsufficientMaterial(const State& state) {
    return false;
}

/// Bitboard maps
static const Bitboard neighbours[64] = {
  0x0000000000000302ULL,  0x0000000000000705ULL,  0x0000000000000e0aULL, 0x0000000000001c14ULL,
  0x0000000000003828ULL,  0x0000000000007050ULL,  0x000000000000e0a0ULL, 0x000000000000c040ULL,
  0x0000000000030203ULL,  0x0000000000070507ULL,  0x00000000000e0a0eULL, 0x00000000001c141cULL,
  0x0000000000382838ULL,  0x0000000000705070ULL,  0x0000000000e0a0e0ULL, 0x0000000000c040c0ULL,
  0x0000000003020300ULL,  0x0000000007050700ULL,  0x000000000e0a0e00ULL, 0x000000001c141c00ULL,
  0x0000000038283800ULL,   0x0000000070507000ULL,  0x00000000e0a0e000ULL, 0x00000000c040c000ULL,
  0x0000000302030000ULL,  0x0000000705070000ULL,  0x0000000e0a0e0000ULL, 0x0000001c141c0000ULL,
  0x0000003828380000ULL,  0x0000007050700000ULL,  0x000000e0a0e00000ULL, 0x000000c040c00000ULL,
  0x0000030203000000ULL,   0x0000070507000000ULL,  0x00000e0a0e000000ULL, 0x00001c141c000000ULL,
  0x0000382838000000ULL,  0x0000705070000000ULL,  0x0000e0a0e0000000ULL, 0x0000c040c0000000ULL,
  0x0003020300000000ULL,  0x0007050700000000ULL,  0x000e0a0e00000000ULL, 0x001c141c00000000ULL,
  0x0038283800000000ULL,   0x0070507000000000ULL,  0x00e0a0e000000000ULL, 0x00c040c000000000ULL,
  0x0302030000000000ULL,  0x0705070000000000ULL,  0x0e0a0e0000000000ULL, 0x1c141c0000000000ULL,
  0x3828380000000000ULL,  0x7050700000000000ULL,  0xe0a0e00000000000ULL, 0xc040c00000000000ULL,
  0x0203000000000000ULL,  0x0507000000000000ULL,  0x0a0e000000000000ULL, 0x141c000000000000ULL,
  0x2838000000000000ULL,  0x5070000000000000ULL,  0xa0e0000000000000ULL, 0x40c0000000000000ULL };


static const Bitboard rook_attacks[64] = {
  0x01010101010101FEULL, 0x02020202020202FDULL, 0x04040404040404FBULL,
  0x08080808080808F7ULL, 0x10101010101010EFULL, 0x20202020202020DFULL,
  0x40404040404040BFULL, 0x808080808080807FULL, 0x010101010101FE01ULL,
  0x020202020202FD02ULL, 0x040404040404FB04ULL, 0x080808080808F708ULL,
  0x101010101010EF10ULL, 0x202020202020DF20ULL, 0x404040404040BF40ULL,
  0x8080808080807F80ULL, 0x0101010101FE0101ULL, 0x020202020202FD02ULL,
  0x040404040404FB04ULL, 0x080808080808F708ULL, 0x101010101010EF10ULL,
  0x202020202020DF20ULL, 0x404040404040BF40ULL, 0x8080808080807F80ULL,
  0x01010101FE010101ULL, 0x02020202FD020202ULL, 0x04040404FB040404ULL,
  0x08080808F7080808ULL, 0x10101010EF101010ULL, 0x20202020DF202020ULL,
  0x40404040BF404040ULL, 0x808080807F808080ULL, 0x010101FE01010101ULL,
  0x02020202FD020202ULL, 0x04040404FB040404ULL, 0x08080808F7080808ULL,
  0x10101010EF101010ULL, 0x20202020DF202020ULL, 0x40404040BF404040ULL,
  0x808080807F808080ULL, 0x0101FE0101010101ULL, 0x02020202FD020202ULL,
  0x04040404FB040404ULL, 0x08080808F7080808ULL, 0x10101010EF101010ULL,
  0x20202020DF202020ULL, 0x40404040BF404040ULL, 0x808080807F808080ULL,
  0x01FE010101010101ULL, 0x02FD020202020202ULL, 0x04FB040404040404ULL,
  0x08F7080808080808ULL, 0x10EF101010101010ULL, 0x20DF202020202020ULL,
  0x40BF404040404040ULL, 0x807F808080808080ULL, 0xFE01010101010101ULL,
  0xFD02020202020202ULL, 0xFB04040404040404ULL, 0xF708080808080808ULL,
  0xEF10101010101010ULL, 0xDF20202020202020ULL, 0xBF40404040404040ULL,
  0x7F80808080808080ULL };

static const Bitboard bishop_attacks[64] = {
  0x8040201008040200ULL, 0x0080402010080500ULL, 0x0000804020110A00ULL,
  0x0000008041221400ULL, 0x0000000182442800ULL, 0x0000010204885000ULL,
  0x000102040810A000ULL, 0x0102040810204000ULL, 0x4020100804020002ULL,
  0x8040201008050005ULL, 0x00804020110A000AULL, 0x0000804122140014ULL,
  0x0000018244280028ULL, 0x0001020488500050ULL, 0x0102040810A000A0ULL,
  0x0204081020400040ULL, 0x2010080402000204ULL, 0x4020100805000508ULL,
  0x804020110A000A11ULL, 0x0080412214001422ULL, 0x0001824428002844ULL,
  0x0102048850005088ULL, 0x02040810A000A010ULL, 0x0408102040004020ULL,
  0x1008040200020408ULL, 0x2010080500050810ULL, 0x4020110A000A1120ULL,
  0x8041221400142241ULL, 0x0182442800284482ULL, 0x0204885000508804ULL,
  0x040810A000A01008ULL, 0x0810204000402010ULL, 0x0804020002040810ULL,
  0x1008050005081020ULL, 0x20110A000A112040ULL, 0x4122140014224180ULL,
  0x8244280028448201ULL, 0x0488500050880402ULL, 0x0810A000A0100804ULL,
  0x1020400040201008ULL, 0x0402000204081020ULL, 0x0805000508102040ULL,
  0x110A000A11204080ULL, 0x2214001422418000ULL, 0x4428002844820100ULL,
  0x8850005088040201ULL, 0x10A000A010080402ULL, 0x2040004020100804ULL,
  0x0200020408102040ULL, 0x0500050810204080ULL, 0x0A000A1120408000ULL,
  0x1400142241800000ULL, 0x2800284482010000ULL, 0x5000508804020100ULL,
  0xA000A01008040201ULL, 0x4000402010080402ULL, 0x0002040810204080ULL,
  0x0005081020408000ULL, 0x000A112040800000ULL, 0x0014224180000000ULL,
  0x0028448201000000ULL, 0x0050880402010000ULL, 0x00A0100804020100ULL,
  0x0040201008040201ULL };

static const Bitboard knight_attacks[64] = {
  0x0000000000020400ULL, 0x0000000000050800ULL, 0x00000000000A1100ULL,
  0x0000000000142200ULL, 0x0000000000284400ULL, 0x0000000000508800ULL,
  0x0000000000A01000ULL, 0x0000000000402000ULL, 0x0000000002040004ULL,
  0x0000000005080008ULL, 0x000000000A110011ULL, 0x0000000014220022ULL,
  0x0000000028440044ULL, 0x0000000050880088ULL, 0x00000000A0100010ULL,
  0x0000000040200020ULL, 0x0000000204000402ULL, 0x0000000508000805ULL,
  0x0000000A1100110AULL, 0x0000001422002214ULL, 0x0000002844004428ULL,
  0x0000005088008850ULL, 0x000000A0100010A0ULL, 0x0000004020002040ULL,
  0x0000020400040200ULL, 0x0000050800080500ULL, 0x00000A1100110A00ULL,
  0x0000142200221400ULL, 0x0000284400442800ULL, 0x0000508800885000ULL,
  0x0000A0100010A000ULL, 0x0000402000204000ULL, 0x0002040004020000ULL,
  0x0005080008050000ULL, 0x000A1100110A0000ULL, 0x0014220022140000ULL,
  0x0028440044280000ULL, 0x0050880088500000ULL, 0x00A0100010A00000ULL,
  0x0040200020400000ULL, 0x0204000402000000ULL, 0x0508000805000000ULL,
  0x0A1100110A000000ULL, 0x1422002214000000ULL, 0x2844004428000000ULL,
  0x5088008850000000ULL, 0xA0100010A0000000ULL, 0x4020002040000000ULL,
  0x0400040200000000ULL, 0x0800080500000000ULL, 0x1100110A00000000ULL,
  0x2200221400000000ULL, 0x4400442800000000ULL, 0x8800885000000000ULL,
  0x100010A000000000ULL, 0x2000204000000000ULL, 0x0004020000000000ULL,
  0x0008050000000000ULL, 0x00110A0000000000ULL, 0x0022140000000000ULL,
  0x0044280000000000ULL, 0x0088500000000000ULL, 0x0010A00000000000ULL,
  0x0020400000000000ULL };

static const Bitboard pawn_attacks[64] = {
  0x0000000000000200ULL,  0x0000000000000500ULL,  0x0000000000000A00ULL,  0x0000000000001400ULL,
  0x0000000000002800ULL,  0x0000000000005000ULL,  0x000000000000A000ULL,  0x0000000000004000ULL,
  0x0000000000020000ULL,  0x0000000000050000ULL,  0x00000000000A0000ULL,  0x0000000000140000ULL,
  0x0000000000280000ULL,  0x0000000000500000ULL,  0x0000000000A00000ULL,  0x0000000000400000ULL,
  0x0000000002000000ULL,  0x0000000005000000ULL,  0x000000000A000000ULL,  0x0000000000080000ULL,
  0x0000000000100000ULL,  0x0000000000200000ULL,   0x0000000000400000ULL,  0x0000000000800000ULL,
  0x0000000001000000ULL,  0x0000000002000000ULL,  0x0000000004000000ULL,  0x0000000008000000ULL,
  0x0000000010000000ULL,  0x0000000020000000ULL,  0x0000000040000000ULL,  0x0000000080000000ULL,
  0x0000000100000000ULL,  0x0000000200000000ULL,  0x0000000400000000ULL,  0x0000000800000000ULL,
  0x0000001000000000ULL,  0x0000002000000000ULL,  0x0000004000000000ULL,  0x0000008000000000ULL,
  0x0000010000000000ULL,  0x0000020000000000ULL,  0x0000040000000000ULL,  0x0000080000000000ULL,
  0x0000100000000000ULL,  0x0000200000000000ULL,  0x0000400000000000ULL,  0x0000800000000000ULL,
  0x0001000000000000ULL,  0x0002000000000000ULL,  0x0004000000000000ULL,  0x0008000000000000ULL,
  0x0010000000000000ULL,  0x0020000000000000ULL,  0x0040000000000000ULL,  0x0080000000000000ULL,
  0x0100000000000000ULL,  0x0200000000000000ULL,  0x0400000000000000ULL,  0x0800000000000000ULL
};

static const Bitboard pawn_attacks_b[64] = {
  0x0000000000000000ULL,  0x0000000000000000ULL,  0x0000000000000000ULL,  0x0000000000000000ULL,
  0x0000000000000000ULL,  0x0000000000000000ULL,  0x0000000000000000ULL,  0x0000000000000000ULL,
  0x0000000000000002ULL,  0x0000000000000005ULL,  0x000000000000000AULL,  0x0000000000000014ULL,
  0x0000000000000028ULL,  0x0000000000000050ULL,  0x00000000000000A0ULL,  0x0000000000000040ULL,
  0x0000000000000200ULL,  0x0000000000000500ULL,  0x0000000000000A00ULL,  0x0000000000001400ULL,
  0x0000000000002800ULL,  0x0000000000005000ULL,  0x000000000000A000ULL,  0x0000000000004000ULL,
  0x0000000000020000ULL,  0x0000000000050000ULL,  0x00000000000A0000ULL,  0x0000000000140000ULL,
  0x0000000000280000ULL,  0x0000000000500000ULL,  0x0000000000A00000ULL,  0x0000000000400000ULL,
  0x0000000002000000ULL,  0x0000000005000000ULL,  0x000000000A000000ULL,  0x0000000014000000ULL,
  0x0000000028000000ULL,  0x0000000050000000ULL,  0x00000000A0000000ULL,  0x0000000040000000ULL,
  0x0000000200000000ULL,  0x0000000500000000ULL,  0x0000000A00000000ULL,  0x0000001400000000ULL,
  0x0000002800000000ULL,  0x0000005000000000ULL,  0x000000A000000000ULL,  0x0000004000000000ULL,
  0x0000020000000000ULL,  0x0000050000000000ULL,  0x00000A0000000000ULL,  0x0000140000000000ULL,
  0x0000280000000000ULL,  0x0000500000000000ULL,  0x0000A00000000000ULL,  0x0000400000000000ULL,
  0x0002000000000000ULL,  0x0005000000000000ULL,  0x000A000000000000ULL,  0x0014000000000000ULL,
  0x0028000000000000ULL,  0x0050000000000000ULL,  0x00A0000000000000ULL,  0x0040000000000000ULL,
};

static const Bitboard files[8] = {
  0x0101010101010101ULL,
  0x0202020202020202ULL,
  0x0404040404040404ULL,
  0x0808080808080808ULL,
  0x1010101010101010ULL,
  0x2020202020202020ULL,
  0x4040404040404040ULL,
  0x8080808080808080ULL };

static const Bitboard ranks[8] = {
  0x00000000000000FFULL,
  0x000000000000FF00ULL,
  0x0000000000FF0000ULL,
  0x00000000FF000000ULL,
  0x000000FF00000000ULL,
  0x0000FF0000000000ULL,
  0x00FF000000000000ULL,
  0xFF00000000000000ULL };

static const Bitboard squares[64] = {
  0x0000000000000001ULL,  0x0000000000000002ULL,  0x0000000000000004ULL,  0x0000000000000008ULL,
  0x0000000000000010ULL,  0x0000000000000020ULL,  0x0000000000000040ULL,  0x0000000000000080ULL,
  0x0000000000000100ULL,  0x0000000000000200ULL,  0x0000000000000400ULL,  0x0000000000000800ULL,
  0x0000000000001000ULL,  0x0000000000002000ULL,  0x0000000000004000ULL,  0x0000000000008000ULL,
  0x0000000000010000ULL,  0x0000000000020000ULL,   0x0000000000040000ULL,  0x0000000000080000ULL,
  0x0000000000100000ULL,  0x0000000000200000ULL,  0x0000000000400000ULL,  0x0000000000800000ULL,
  0x0000000001000000ULL,  0x0000000002000000ULL,  0x0000000004000000ULL,  0x0000000008000000ULL,
  0x0000000010000000ULL,  0x0000000020000000ULL,  0x0000000040000000ULL,  0x0000000080000000ULL,
  0x0000000100000000ULL,  0x0000000200000000ULL,  0x0000000400000000ULL,  0x0000000800000000ULL,
  0x0000001000000000ULL,  0x0000002000000000ULL,  0x0000004000000000ULL,  0x0000008000000000ULL,
  0x0000010000000000ULL,  0x0000020000000000ULL,  0x0000040000000000ULL,  0x0000080000000000ULL,
  0x0000100000000000ULL,  0x0000200000000000ULL,  0x0000400000000000ULL,  0x0000800000000000ULL,
  0x0001000000000000ULL,  0x0002000000000000ULL,  0x0004000000000000ULL,  0x0008000000000000ULL,
  0x0010000000000000ULL,  0x0020000000000000ULL,  0x0040000000000000ULL,  0x0080000000000000ULL,
  0x0100000000000000ULL,  0x0200000000000000ULL,  0x0400000000000000ULL,  0x0800000000000000ULL,
  0x1000000000000000ULL,  0x2000000000000000ULL,  0x4000000000000000ULL,  0x8000000000000000ULL
};

static const Bitboard diagonals[64] = {
  0x8040201008040200ULL,  0x0080402010080400ULL,  0x0000804020100800ULL,  0x0000008040201000ULL,
  0x0000000080402000ULL,  0x0000000000804000ULL,  0x0000000000008000ULL,  0x0000000000000000ULL,
  0x4020100804020000ULL,  0x8040201008040001ULL,  0x0080402010080002ULL,  0x0000804020100004ULL,
  0x0000008040200008ULL,  0x0000000080400010ULL,  0x0000000000800020ULL,  0x0000000000000040ULL,
  0x2010080402000000ULL,  0x4020100804000100ULL,  0x8040201008000201ULL,  0x0080402010000402ULL,
  0x0000804020000804ULL,  0x0000008040001008ULL,  0x0000000080002010ULL,  0x0000000000004020ULL,
  0x1008040200000000ULL,  0x2010080400010000ULL,  0x4020100800020100ULL,  0x8040201000040201ULL,
  0x0080402000080402ULL,  0x0000804000100804ULL,  0x0000008000201008ULL,  0x0000000000402010ULL,
  0x0804020000000000ULL,  0x1008040001000000ULL,  0x2010080002010000ULL,  0x4020100004020100ULL,
  0x8040200008040201ULL,  0x0080400010080402ULL,  0x0000800020100804ULL,  0x0000000040201008ULL,
  0x0402000000000000ULL,  0x0804000100000000ULL,  0x1008000201000000ULL,  0x2010000402010000ULL,
  0x4020000804020100ULL,  0x8040001008040201ULL,  0x0080002010080402ULL,  0x0000004020100804ULL,
  0x0200000000000000ULL,  0x0400010000000000ULL,  0x0800020100000000ULL,  0x1000040201000000ULL,
  0x2000080402010000ULL,  0x4000100804020100ULL,  0x8000201008040201ULL,  0x0000402010080402ULL,
  0x0000000000000000ULL,  0x0001000000000000ULL,  0x0002010000000000ULL,  0x0004020100000000ULL,
  0x0008040201000000ULL,  0x0010080402010000ULL,  0x0020100804020100ULL,  0x0040201008040201ULL,
};

static const Bitboard antidiagonals[64] = {
  0x0000000000000000ULL,  0x0000000000000100ULL,  0x0000000000010200ULL,  0x0000000001020400ULL,
  0x0000000102040800ULL,  0x0000010204081000ULL,  0x0001020408102000ULL,  0x0102040810204000ULL,
  0x0000000000000002ULL,  0x0000000000010004ULL,  0x0000000001020008ULL,  0x0000000102040010ULL,
  0x0000010204080020ULL,  0x0001020408100040ULL,  0x0102040810200080ULL,  0x0204081020400000ULL,
  0x0000000000000204ULL,  0x0000000001000408ULL,  0x0000000102000810ULL,  0x0000010204001020ULL,
  0x0001020408002040ULL,  0x0102040810004080ULL,  0x0204081020008000ULL,  0x0408102040000000ULL,
  0x0000000000020408ULL,  0x0000000100040810ULL,  0x0000010200081020ULL,  0x0001020400102040ULL,
  0x0102040800204080ULL,  0x0204081000408000ULL,  0x0408102000800000ULL,  0x0810204000000000ULL,
  0x0000000002040810ULL,  0x0000010004081020ULL,  0x0001020008102040ULL,  0x0102040010204080ULL,
  0x0204080020408000ULL,  0x0408100040800000ULL,  0x0810200080000000ULL,  0x1020400000000000ULL,
  0x0000000204081020ULL,  0x0001000408102040ULL,  0x0002000810204080ULL,  0x0004001020408000ULL,
  0x0008002040800000ULL,  0x0010004080000000ULL,  0x0020008000000000ULL,  0x0040000000000000ULL,
  0x0080000000000000ULL,  0x0000000000000000ULL,
};

/// Square lookup
static const std::map<unsigned long long, Square> square_lookup = {
  { 0x0000000000000001ULL, a1 },
  { 0x0000000000000002ULL, b1 },
  { 0x0000000000000004ULL, c1 },
  { 0x0000000000000008ULL, d1 },
  { 0x0000000000000010ULL, e1 },
  { 0x0000000000000020ULL, f1 },
  { 0x0000000000000040ULL, g1 },
  { 0x0000000000000080ULL, h1 },
  { 0x0000000000000100ULL, a2 },
  { 0x0000000000000200ULL, b2 },
  { 0x0000000000000400ULL, c2 },
  { 0x0000000000000800ULL, d2 },
  { 0x0000000000001000ULL, e2 },
  { 0x0000000000002000ULL, f2 },
  { 0x0000000000004000ULL, g2 },
  { 0x0000000000008000ULL, h2 },
  { 0x0000000000010000ULL, a3 },
  { 0x0000000000020000ULL, b3 },
  { 0x0000000000040000ULL, c3 },
  { 0x0000000000080000ULL, d3 },
  { 0x0000000000100000ULL, e3 },
  { 0x0000000000200000ULL, f3 },
  { 0x0000000000400000ULL, g3 },
  { 0x0000000000800000ULL, h3 },
  { 0x0000000001000000ULL, a4 },
  { 0x0000000002000000ULL, b4 },
  { 0x0000000004000000ULL, c4 },
  { 0x0000000008000000ULL, d4 },
  { 0x0000000010000000ULL, e4 },
  { 0x0000000020000000ULL, f4 },
  { 0x0000000040000000ULL, g4 },
  { 0x0000000080000000ULL, h4 },
  { 0x0000000100000000ULL, a5 },
  { 0x0000000200000000ULL, b5 },
  { 0x0000000400000000ULL, c5 },
  { 0x0000000800000000ULL, d5 },
  { 0x0000001000000000ULL, e5 },
  { 0x0000002000000000ULL, f5 },
  { 0x0000004000000000ULL, g5 },
  { 0x0000008000000000ULL, h5 },
  { 0x0000010000000000ULL, a6 },
  { 0x0000020000000000ULL, b6 },
  { 0x0000040000000000ULL, c6 },
  { 0x0000080000000000ULL, d6 },
  { 0x0000100000000000ULL, e6 },
  { 0x0000200000000000ULL, f6 },
  { 0x0000400000000000ULL, g6 },
  { 0x0000800000000000ULL, h6 },
  { 0x0001000000000000ULL, a7 },
  { 0x0002000000000000ULL, b7 },
  { 0x0004000000000000ULL, c7 },
  { 0x0008000000000000ULL, d7 },
  { 0x0010000000000000ULL, e7 },
  { 0x0020000000000000ULL, f7 },
  { 0x0040000000000000ULL, g7 },
  { 0x0080000000000000ULL, h7 },
  { 0x0100000000000000ULL, a8 },
  { 0x0200000000000000ULL, b8 },
  { 0x0400000000000000ULL, c8 },
  { 0x0800000000000000ULL, d8 },
  { 0x1000000000000000ULL, e8 },
  { 0x2000000000000000ULL, f8 },
  { 0x4000000000000000ULL, g8 },
  { 0x8000000000000000ULL, h8 } };
} 

#endif