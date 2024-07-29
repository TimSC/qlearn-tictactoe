# Inspired by https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial

import random
import numpy as np

class RandomPlay():
	def __init__(self, player_num):
		self.player_num = player_num

	def get_move(self, environ):
		#return (random.randint(0, 2), random.randint(0, 2))

		legal_moves_x, legal_moves_y = GetLegalMoves(environ)
		assert len(legal_moves_x) > 0
		ch = random.randint(0, len(legal_moves_x)-1)
		return legal_moves_x[ch], legal_moves_y[ch]

class QPlay():
	def __init__(self, player_num, fina='q_table.npy', fina_keys='q_table_keys.npy', 
			swap_player_ids = False):

		self.player_num = player_num
		with open(fina, 'rb') as f:
			self.q = np.load(f)
		with open(fina_keys, 'rb') as f:
			self.q_state_keys = np.load(f)

		if swap_player_ids:
			self.q_state_keys[self.q_state_keys > 0] = 3 - self.q_state_keys[self.q_state_keys > 0]

		self.q_state_dict = {}
		for k, q_row in zip(self.q_state_keys, self.q):
			self.q_state_dict[tuple(k)] = q_row

		self.random_move_probability = 0.05

	def get_move(self, environ):

		s = tuple(environ.flatten())

		# Occasionally make a random move to provide plenty of variety
		if random.random() < self.random_move_probability:
			legal_moves_x, legal_moves_y = GetLegalMoves(environ)
			assert len(legal_moves_x) > 0
			ch = random.randint(0, len(legal_moves_x)-1)
			return legal_moves_x[ch], legal_moves_y[ch]			

		if s in self.q_state_dict:
			#Found matching state
			q_row = self.q_state_dict[s]
			action = np.argmax(q_row)
			return action // 3, action % 3

		else:
			# Look for a similar state
			min_diff = None
			best_rows = []
			for row_state, q_row in zip(self.q_state_keys, self.q):
				row_state = row_state.flatten()
				row_diff = np.sum(np.abs(s - row_state))

				if min_diff is None or row_diff < min_diff:
					min_diff = row_diff
					best_rows = [row_state]
				elif row_diff == min_diff:
					best_rows.append(row_state)

			best_row = random.choice(best_rows)

			q_row = self.q_state_dict[tuple(best_row)]
			action = np.argmax(q_row)
			return action // 3, action % 3

def CheckForWin(environ):
	for r in range(environ.shape[0]):

		if environ[r, 0] != 0:

			win = True
			for c in range(1, environ.shape[0]):
				if environ[r, c] != environ[r, 0]:
					win = False
					break
			if win:
				return environ[r, 0]

	for c in range(environ.shape[0]):

		if environ[0, c] != 0:

			win = True
			for r in range(1, environ.shape[1]):
				if environ[r, c] != environ[0, c]:
					win = False
					break
			if win:
				return environ[0, c]

	if environ[0, 0] != 0:

		win = True
		for d in range(1, environ.shape[0]):
			if environ[d, d] != environ[0, 0]:
				win = False
				break
		if win:
			return environ[0, 0]

	if environ[0, -1] != 0:

		win = True
		for d in range(1, environ.shape[0]):
			if environ[d, environ.shape[1]-d-1] != environ[0, -1]:
				win = False
				break
		if win:
			return environ[0, -1]
	
	return 0

def GetLegalMoves(environ):
	m = np.where(environ == 0)
	return m

if __name__=="__main__":

	c = [0, 0]

	move_count = 0

	q_state_keys = []
	q_state_dict = {}
	q = []

	alpha = 0.7
	gamma = 0.95
	max_epsilon = 1.0
	min_epsilon = 0.05
	max_steps = 100000

	opponent = RandomPlay(2)
	#opponent = QPlay(2, fina='q_table1.npy', fina_keys='q_table_keys1.npy', 
	#	swap_player_ids = True)

	for i in range(max_steps):

		epsilon = max_epsilon + (min_epsilon - max_epsilon) * i / max_steps
		environ = np.zeros((3,3), dtype=np.int8)
		player_num = 1
		#opponent.random_move_probability = epsilon

		if random.randint(0,1) != 0:
			# Opponent goes first
			move2 = opponent.get_move(environ)
			environ[*move2] = opponent.player_num

		# Create state in q table
		environ_flat = tuple(environ.reshape((9,)))
		if environ_flat not in q_state_dict:
			q_row = np.random.rand(9) * 2 - 1
			
			q_state_keys.append(environ_flat)
			q_state_dict[environ_flat] = q_row
			q.append(q_row)
		else:
			q_row = q_state_dict[environ_flat]

		while True:

			old_q_row = q_row
			#print (environ)
 
			# Choose an action (Epsilon-greedy policy)
			if random.random() < epsilon:

				# Exploration with random action
				legal_moves_x, legal_moves_y = GetLegalMoves(environ)
				assert len(legal_moves_x) > 0
				ch = random.randint(0, len(legal_moves_x)-1)
				move1 = legal_moves_x[ch], legal_moves_y[ch]
				action_id = move1[0] * 3 + move1[1]

			else:
				# Panned move (exploitation)
				action_id = np.argmax(q_row)
				move1 = action_id // 3, action_id % 3
				if environ[*move1] != 0:

					# Something went wrong, so fall back to random
					legal_moves_x, legal_moves_y = GetLegalMoves(environ)
					assert len(legal_moves_x) > 0
					ch = random.randint(0, len(legal_moves_x)-1)
					move1 = legal_moves_x[ch], legal_moves_y[ch]
					action_id = move1[0] * 3 + move1[1]

			# Perform action

			assert environ[*move1] == 0
			environ[*move1] = 1

			win = CheckForWin(environ)
			if win == 1:
				print (i, "player 1 wins")
			else:

				legal_moves_x, legal_moves_y = GetLegalMoves(environ)
				if len(legal_moves_x) == 0: # draw
					print (i, "draw")
					win = -1
				else:
					move2 = opponent.get_move(environ)
					if environ[*move2] != 0:
						#AI player attempted illegal move
						movechoice = random.randint(0, len(legal_moves_x)-1)
						move2 = legal_moves_x[movechoice], legal_moves_y[movechoice]
		
					environ[*move2] = 2

					win = CheckForWin(environ)
					if win == opponent.player_num:
						print (i, "player 2 wins")

					else:
						legal_moves_x, legal_moves_y = GetLegalMoves(environ)
						if len(legal_moves_x) == 0: # draw
							print (i, "draw")
							win = -1

			# Measure reward
			rewardDict = {-1:0, 0:0, 1:1, 2:-1}
			reward = rewardDict[win]

			# Ensure this state is in q table, grow if necessary
			environ_flat = tuple(environ.reshape((9,)))
			if environ_flat not in q_state_dict:
				q_row = np.random.rand(9) * 2 - 1
				
				q_state_keys.append(environ_flat)
				q_state_dict[environ_flat] = q_row
				q.append(q_row)
			else:
				q_row = q_state_dict[environ_flat]

			# Get max reward of available states
			max_future_reward = np.max(q_row)

			# Update q table
			new_q = old_q_row[action_id] + alpha * (reward + gamma * max_future_reward - old_q_row[action_id])
			old_q_row[action_id] = new_q

			if win != 0:
				break #Game over

		#if i % 1000 == 0:
		#	print ("q table")
		#	for k, q_row in zip(q_state_keys, q):
		#		print (k, q_row)
		print ("q table len", len(q), ", epsilon", epsilon)

	with open('q_table.npy', 'wb') as f:
		np.save(f, np.array(q))
	with open('q_table_keys.npy', 'wb') as f:
		np.save(f, np.array(q_state_keys))

