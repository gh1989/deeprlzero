#include "geometry.h"

bool SquareConnectedToBitboard(
	Square source,
	Bitboard target,
	Bitboard obstacle,
	const std::array<std::pair<int, int>, 4>& directions)
{
	for (int i = 0; i < 4; i++)
	{
		std::pair<int, int> dir = directions[i];
		auto dx = dir.first;
		auto dy = dir.second;
		int sidx = source;
		int rank = sidx / 8;
		while (true)
		{
			if (dx > 0 && rank == 7)
				break;
			if (dx < 0 && rank == 0)
				break;
			int file = sidx % 8;
			if (dy > 0 && file == 7)
				break;
			if (dy < 0 && file == 0)
				break;

			sidx += 8 * dx + dy;
			rank += dx;

			if ((sidx > 63) | (sidx < 0))
				break;

			auto tmpbb = squares[sidx];
			if (tmpbb & obstacle)
				break;
			if (tmpbb & target)
				return true;
		}
	}
	return false;
}