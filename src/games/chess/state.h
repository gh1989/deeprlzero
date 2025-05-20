#ifndef STATE_H
#define STATE_H

#include <numeric>
#include "bitboard.h"
#include "move.h"
#include "types.h"

/* class: State */
class State
{
private:
    Bitboard bbs[13];
    uint8_t c50;
    uint8_t castle;
    uint32_t plies;
    bool blackMove;

public:

	bool operator==(const State& other) const {
		return 	+bbs == +other.bbs &&
				c50 == other.c50 &&
				castle == other.castle && 
				plies == other.plies &&
				blackMove == other.blackMove;
	}

 // Getter and Setter for bbs
    Bitboard getBitboard(int index) const {
        return bbs[index];
    }
    void setBitboard(int index, Bitboard value) {
        bbs[index] = value;
    }
	void addToBitboard(int index, Bitboard toAdd) {
		bbs[index] |= toAdd;
	}

    // Getter and Setter for c50
    uint8_t getMoveCount() const {
        return plies / 2;
    }
    void setMoveCount(uint8_t value) {
        plies = value;
    }

    // Getter and Setter for castle
    uint8_t getCastleRights() const {
        return castle;
    }
    void setCastleRights(uint8_t value) {
        castle = value;
    }

	Bitboard getEnPassant() const
	{ return bbs[12]; }

    // Getter and Setter for plies
    uint32_t getPlies() const {
        return plies;
    }
    void setPlies(uint32_t value) {
        plies = value;
    }

	uint8_t get50MoveCount() const {
		return c50;
	}

	void set50MoveCount(uint8_t value) {
		c50 = value;
	}

    // Getter and Setter for blackMove
    bool isBlackMove() const {
        return blackMove;
    }
    void setBlackMove(bool value) {
        blackMove = value;
    }

	State() = default;
	void Apply(Move);

	Bitboard enemyOccupation() const { return blackMove ? whiteOccupation() : blackOccupation(); };
	Bitboard moveOccupation() const { return blackMove ? blackOccupation() : whiteOccupation(); }
	Bitboard blackOccupation() const { return combinedBoards(bbs+NUMBER_PIECES, bbs+NUMBER_PIECES*2); }
	Bitboard whiteOccupation() const { return combinedBoards(bbs, bbs+NUMBER_PIECES); }

	template<Piece _Piece> 
	Bitboard movePiece() const
	{
		return bbs[moveBitStart() + _Piece];
	}
	template<Piece _Piece>
	Bitboard enemyPiece() const
	{
		return bbs[enemyBitStart() + _Piece];
	}
	template<Piece _Piece>
	Bitboard whitePiece() const
	{
		return bbs[_Piece];
	}
	template<Piece _Piece>
	Bitboard blackPiece() const
	{
		return bbs[NUMBER_PIECES+_Piece];
	}

private:

	uint8_t moveBitStart() const { return blackMove ? NUMBER_PIECES : 0; };
	uint8_t moveBitEnd() const { return blackMove ? 2*NUMBER_PIECES : NUMBER_PIECES; };
	uint8_t enemyBitStart() const { return blackMove ? 0 : NUMBER_PIECES; };
	uint8_t enemyBitEnd() const { return blackMove ? NUMBER_PIECES : 2*NUMBER_PIECES; };

	Bitboard combinedBoards(auto begin, auto end) const
	{
		auto bit_or = [&](const Bitboard &a, const Bitboard &b) { return a | b; };
		return std::accumulate<>(begin, end, Bitboard(0), bit_or);
	}
};

#endif