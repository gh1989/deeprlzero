#ifndef MOVE_GENERATION_H
#define MOVE_GENERATION_H

#include <list>

#include "bitboard.h"
#include "geometry.h"
#include "move.h"
#include "state.h"
#include "types.h"

typedef std::list<Move> TMoveContainer;

/* Move generation */
TMoveContainer GenerateMoves(const State&);

void PawnMoves(
	const State& state,
	TMoveContainer &moves);

template <Piece _Piece>
void JumperMoves(
    const State& state,
	TMoveContainer &moves);

template <Piece _Piece>
void SliderMoves(
    const State& state,
	TMoveContainer &moves);

void KingCastling(
	const State& state,
	TMoveContainer &moves);

bool isCheck(const State& state, bool blackIsAttacking);
bool isBitboardAttacked(const Bitboard& bitboard, const State& state, bool blackIsAttacking);

bool checkLegal(const State& state, Move move);

#endif