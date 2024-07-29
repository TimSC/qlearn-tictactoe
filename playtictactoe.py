import numpy as np
from qlearntictactoe import CheckForWin, GetLegalMoves, QPlay

if __name__=="__main__":

	ai_player_id = 1
	human_player_id = 2
	qPlay = QPlay(ai_player_id, swap_player_ids=True)
	
	environ = np.zeros((3,3), dtype=np.int8)
	
	while 1:
		print (environ)

		while 1:
			pos = input("Play position? (row, col) or pass ")

			if pos == "pass":
				break

			try:
				posSplit = pos.split(",")
				r, c = tuple(map(int, posSplit))
			except ValueError:
				continue

			if environ[r, c] == 0:
				environ[r, c] = human_player_id
				break
		
		win = CheckForWin(environ)
		if win != 0:
			print ("Player {} wins".format(win))
			exit(0)

		legal_moves_x, legal_moves_y = GetLegalMoves(environ)
		if len(legal_moves_x) == 0:
			print ("Draw")
			exit(0)

		print (environ)
		r, c = qPlay.get_move(environ)

		if environ[r, c] == 0:
			environ[r, c] = ai_player_id
		else:
			print ("AI player attempted illegal move")
		
		win = CheckForWin(environ)
		if win != 0:
			print (environ)
			print ("Player {} wins".format(win))
			exit(0)

		legal_moves_x, legal_moves_y = GetLegalMoves(environ)
		if len(legal_moves_x) == 0:
			print (environ)
			print ("Draw")
			exit(0)

