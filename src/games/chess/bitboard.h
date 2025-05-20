#ifndef BITBOARD_H
#define BITBOARD_H

#include <iostream>
#include <limits>
#include <stdint.h>
#include <stdlib.h>
#include <bit>
#include <iterator> 

/* 	class: BitBoard */
struct Bitboard
{
	Bitboard() : bit_number(0) {}
	Bitboard(uint64_t bit_number_) : bit_number(bit_number_) {}

	bool operator==(const Bitboard& other) const {
		return bit_number == other.bit_number;
	}

	Bitboard operator<<(int x) const {
		return Bitboard(bit_number << x);
	}

	Bitboard operator~() const {
		return Bitboard(~bit_number);
	}

	Bitboard operator>>(int x) const {
		return Bitboard(bit_number >> x);
	}

	Bitboard operator| (const Bitboard& other) const {
		return Bitboard(bit_number | other.bit_number);
	}

	Bitboard& operator|=(const Bitboard& other) {
		bit_number |= other.bit_number;
		return *this;
	}

	Bitboard operator& (const Bitboard& other) const {
		return Bitboard(bit_number & other.bit_number);
	}

	Bitboard operator^ (const Bitboard& other) const {
		return Bitboard(bit_number ^ other.bit_number);
	}

	Bitboard operator- (int value) const {
		return bit_number - value;
	}

	void operator &= (const Bitboard& other) {
		bit_number &= other.bit_number;
	}

	operator bool() const {
		return bit_number != 0;
	}

	uint64_t PopCnt() const {
		return std::popcount(bit_number);
	}

	uint8_t nMSB()
	{
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

/* class: BitIterator */
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

/* create a BitIterator range for a Bitboard */
class BitboardRange {
public:
    BitboardRange(Bitboard bb) : bb_(bb) {};
    BitIterator begin() const { return BitIterator(bb_, bb_.nLSB()); }
    BitIterator end() const { return BitIterator(0, 64); }
private:
    Bitboard bb_;
};

#endif