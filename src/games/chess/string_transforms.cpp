#include <sstream>
#include <iostream>
#include <unordered_map>
#include "move.h"
#include "state.h"
#include "string_transforms.h"
#include "types.h"

void PrettyPrint(const State& state)
{
	// collect information
	std::string pos[8][8];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			pos[i][j] = ' ';

	const std::string PIECE_STRINGS = "NBRQKP";
	const std::string piece_strings = "nbrqkp";

	for (int i = 0; i < NUMBER_PIECES; ++i)
	{
		// requires knowledge of implementation in state. Bad
		Bitboard wocc = state.getBitboard(i);
		Bitboard bocc = state.getBitboard(i+NUMBER_PIECES); 
		Piece piece = static_cast<Piece>(i);
		for (int i = 0; i < 64; i++)
		{
			int j = (7 - int(i / 8)) % 8;
			int k = i % 8;
			if (wocc & squares[i])
				pos[j][k] = PIECE_STRINGS[piece];
			if (bocc & squares[i])
				pos[j][k] = piece_strings[piece];
		}
	}

	// print out the board
	std::string baseline = "+---";
	for (auto j = 0; j < 7; j++)
		baseline += "+---";
	baseline += "+\n";

	std::string output = baseline;
	for (auto i = 0; i < 8; i++)
	{
		for (auto j = 0; j < 8; j++)
			output += "| " + pos[i][j] + " ";
		output += "|\n";
		output += baseline;
	}

	std::cout << output;
	Bitboard ep = state.getBitboard(12);

	if (ep)
	{
		std::cout << "en-passant: ";
		for (int i = 0; i < 63; i++)
		{
			if (squares[i] & ep)
				std::cout << SquareName(Square(i));
		}
		std::cout << std::endl;
	}
	std::cout << "fiftycounter: " << state.getMoveCount() << std::endl;
	int castlerights = state.getCastleRights();
	const std::string crights = "QKqk";
	std::cout << "castlerights: " << castlerights << " ";
	for (char c : crights)
	{
		if (castlerights % 2)
			std::cout << c;
		castlerights /= 2;
	}

	std::cout << std::endl;
	std::cout << "plies: " << state.getPlies() << std::endl;
	std::cout << "colour to move: " << (!state.isBlackMove() ? "white" : "black") << std::endl;
}

// Get square name as string
std::string SquareName(Square sqr)
{
	const std::string file_strings[8] = { "a", "b", "c", "d", "e", "f", "g", "h" };
	const std::string rank_strings[8] = { "1", "2", "3", "4", "5", "6", "7", "8" };
	int square_int = static_cast<int>(sqr);
	return file_strings[square_int % 8] + rank_strings[int(square_int / 8)];
}
std::string PieceStringLower(Piece piece) { return piece_strings[piece]; }

Bitboard BitboardFromString(std::string str)
{
	if (str[0] < 'a' || str[1] < '1' || str[0] > 'h' || str[1] > '8')
		throw std::runtime_error("Square string is formatted improperly.");
	uint64_t boardnum = str[0] - 'a' + 8 * (str[1] - '1');
	return Bitboard(1ULL << boardnum);
}

/* UCI and string operations */
std::string AsUci(Move move) {
	std::stringstream ss;
	ss << SquareName(GetFrom(move));
	ss << SquareName(GetTo(move));
	if (SpecialMoveType(move) == PROMOTE)
		ss << PieceStringLower(PromotionPiece(move));
	return  ss.str();
}

Bitboard get_en_passant(const std::string& fen) {
	std::size_t pos = fen.find(" ");
    pos = fen.find(" ", pos+1);
    pos = fen.find(" ", pos+1);
    if (pos == std::string::npos) {
        return 0;
    }
    std::string ep_square = fen.substr(pos+1, 2);
    if (ep_square == "- ") {
        return 0;
    }
    uint64_t enPassantIdx = ep_square[0] - 'a' + 8 * (ep_square[1] - '1');
	return Bitboard(1ULL << enPassantIdx);
}

int get_fifty_move_count(const std::string& fen) {
    std::size_t pos = fen.find(" ");
    pos = fen.find(" ", pos+1);
    pos = fen.find(" ", pos+1);
    pos = fen.find(" ", pos+1);
	std::size_t end = fen.find(" ", pos+1);
    if (pos == std::string::npos) {
        return 0;
    }
    return std::stoi(fen.substr(pos+1, end));
}

bool get_turn(const std::string& fen) {
    std::size_t pos = fen.find(" ");
    return (fen.substr(pos+1, 1) != "w");
}

State StateFromFen(std::string fen) 
{
	State state;
	// Find the position of the board part of the FEN string
	std::size_t pos_end = fen.find(" ");
	std::size_t rank = 7;
	std::size_t file = 0;
	// Go through the board part of the FEN string
	for (int i = 0; i < pos_end; i++) {
		char c = fen[i];
		if (c >= '1' && c <= '8') {
			// Empty squares
			file += (c - '0');
		} else if (c == '/') {
			// Skip to the next rank
			rank--;
			file = 0;
			continue;
		} else {
			// Piece squares
			Bitboard square = (1ULL << (8*rank+file));
			file++;

			bool _whitePiece = !std::islower(c);
			auto C = std::toupper(c);
			std::unordered_map<char, Piece> pieceLookup = {
    			{'P', PAWN},
    			{'N', KNIGHT},
    			{'B', BISHOP},
    			{'R', ROOK},
    			{'Q', QUEEN},
    			{'K', KING}
			};

			Piece _piece = pieceLookup[C]; 
			state.addToBitboard(!_whitePiece*NUMBER_PIECES+_piece, square );
		}
	}

	state.setBitboard(12, get_en_passant(fen));
	state.set50MoveCount( get_fifty_move_count(fen) );
	state.setBlackMove( get_turn(fen) );
	
	// Find the position of the castling part of the FEN string
	std::size_t castling_pos = fen.find_first_of(' ', pos_end + 1);

	// Check if there are any castling rights
	if (fen[castling_pos - 1] == '-') {
		// No castling rights
		state.setCastleRights(0);
	} else {
		auto castle = 0;
		// Parse the castling rights
		if (fen.find('K', castling_pos) != std::string::npos) {
			castle |= WK;
		}
		if (fen.find('Q', castling_pos) != std::string::npos) {
			castle |= WQ;
		}
		if (fen.find('k', castling_pos) != std::string::npos) {
			castle |= BK;
		}
		if (fen.find('q', castling_pos) != std::string::npos) {
			castle |= BQ;
		}
		state.setCastleRights(castle);
	}

	// Find the position of the ply part of the FEN string
	std::size_t move_pos = fen.find_last_of(' ');
	std::string move_str = fen.substr(move_pos + 1);
	state.setPlies( 2*(std::stoi(move_str)-1) + state.isBlackMove() );

	return state;
}