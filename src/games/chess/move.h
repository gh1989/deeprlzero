#ifndef MOVE_H
#define MOVE_H

#include "bitboard.h"
#include "square.h"
#include "types.h"

/* type: Move */
typedef unsigned short Move;

constexpr unsigned short to_bits = 6;
constexpr unsigned short flag_bits = 12;
constexpr unsigned short prom_bits = 14;
constexpr unsigned short from_mask = 63;
constexpr unsigned short to_mask = 63 << to_bits;
constexpr unsigned short flag_mask = 3 << flag_bits;
constexpr unsigned short prom_mask = 3 << prom_bits;

/* Move operations*/
// Source of move
Square GetFrom(Move move);
// Destination of move
Square GetTo(Move move);
// Type of special move
SpecialMove SpecialMoveType(Move move);
// Promotion piece of move
Piece PromotionPiece(Move move);
// Reflect move
Move ReflectMove(Move move);
// Create move
Move CreateMove(Square from, Square to);
// Create promotion move
Move CreatePromotion(Square from, Square to, Piece promo);
// Create en passasnt move
Move CreateEnPassant(Square from, Square to);
// Create castle move
Move CreateCastle(Square from, Square to);

#endif