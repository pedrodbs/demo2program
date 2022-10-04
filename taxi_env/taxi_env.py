import gym
import numpy as np
from typing import List, Dict
from dijkstra import Graph, DijkstraSPF
from gym.envs.toy_text.taxi import TaxiEnv as TaxiGym
from gym.wrappers import TimeLimit

# see description at: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
CLR_RED_SYMB = 'Red'
CLR_GREEN_SYMB = 'Green'
CLR_YELLOW_SYMB = 'Yellow'
CLR_BLUE_SYMB = 'Blue'
TAXI_SYMB = 'Taxi'

COLORS = [CLR_RED_SYMB, CLR_GREEN_SYMB, CLR_YELLOW_SYMB, CLR_BLUE_SYMB]
PASS_COLORS = COLORS + [TAXI_SYMB]

ACT_MOVE_SYMB = 'Move'
ACT_PICK_SYMB = 'PickUp'
ACT_DROP_SYMB = 'DropOff'
ACTION_LIST = [f'{ACT_MOVE_SYMB}_{c}' for c in COLORS] + [ACT_PICK_SYMB, ACT_DROP_SYMB]
ACT2INT = {v: i for i, v in enumerate(ACTION_LIST)}

MAP_COLORS = ['R', 'G', 'Y', 'B']
MAP_PASS = [f'{c}_P' for c in MAP_COLORS]
MAP_PASS_TAXI = [f'{c}_T_P' for c in MAP_COLORS]
MAP_DEST = [f'{c}_D' for c in MAP_COLORS]
MAP_DEST_TAXI = [f'{c}_D_T' for c in MAP_COLORS]
MAP_DEST_TAXI_PASS = [f'{c}_D_T_P' for c in MAP_COLORS]
MAP_DEST_PASS_TAXI = [f'{c}_D_P_T' for c in MAP_COLORS]
MAP_TAXI = [f'{c}_T' for c in MAP_COLORS] + [f'{c}_T' for c in MAP_PASS]
MAP_GRID = ['+', '-', ':', '|', ' ', ' _T', ' _T_P']
INT2MAP: List[str] = MAP_GRID + MAP_COLORS + MAP_PASS + MAP_TAXI + MAP_PASS_TAXI + \
                     MAP_DEST + MAP_DEST_TAXI + MAP_DEST_TAXI_PASS + MAP_DEST_PASS_TAXI
MAP2INT: Dict[str, int] = {v: i for i, v in enumerate(INT2MAP)}


class TaxiEnv(object):

    def __init__(self, seed: int = 17, max_steps=50):
        self.seed = seed

        self.s_h: List[np.ndarray] = []
        self.a_h: List[int] = []
        self.p_v_h: List[np.ndarray] = []
        self.s: int = -1

        self.env: TaxiGym = TimeLimit(gym.make('Taxi-v3', render_mode='ansi'), max_episode_steps=max_steps)

        # get shortest paths to color locations and compute move action from every other location
        g = self._get_map_graph()
        self.paths = {}
        for loc in self.env.locs:
            self.paths[loc] = {}
            dijkstra = DijkstraSPF(g, loc)
            for other_loc in g.get_nodes():
                path = dijkstra.get_path(other_loc)
                act = -1
                if len(path) == 1:
                    # same pos, compute action to remain in same pos
                    if other_loc[1] == 0:
                        act = 3  # move west
                    elif other_loc[1] == 4:
                        act = 2  # move east
                    if other_loc[0] == 0:
                        act = 1  # move north
                    elif other_loc[0] == 4:
                        act = 0  # move south
                else:
                    next_loc = path[-2]
                    if other_loc[1] != next_loc[1]:
                        if other_loc[1] < next_loc[1]:
                            act = 2  # move east
                        else:
                            act = 3  # move west
                    elif other_loc[0] != next_loc[0]:
                        if other_loc[0] < next_loc[0]:
                            act = 0  # move south
                        else:
                            act = 1  # move north

                assert act != -1
                self.paths[loc][other_loc] = act

    def init_game(self):
        # reset env
        self.s, *_ = self.env.reset(seed=self.seed)
        self.seed = None

        # initialize histories
        self.s_h = [self._get_state_array()]
        self.a_h = []
        self.p_v_h = [self._get_percept_vector()]  # perception vector

    def _get_map_graph(self) -> Graph:
        desc = self.env.desc.astype(str)
        g = Graph()
        for i in range(1, desc.shape[0] - 1):
            r = i - 1
            for j in range(1, desc.shape[1] - 2, 2):
                c = int((j - 1) / 2)
                if i + 1 < desc.shape[0] - 1:
                    g.add_edge((r, c), (r + 1, c), 1)  # add vertical edges
                    g.add_edge((r + 1, c), (r, c), 1)
                if desc[i, j + 1] != '|':
                    g.add_edge((r, c), (r, c + 1), 1)  # add horizontal edges if no wall
                    g.add_edge((r, c + 1), (r, c), 1)
        return g

    def _get_state_array(self) -> np.array:
        # based on gym.envs.toy_text.taxi.TaxiEnv._render_text
        desc = self.env.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.decode(self.s)

        # destination
        di, dj = self.env.locs[dest_idx]
        out[1 + di][2 * dj + 1] += '_D'

        if pass_idx < 4:
            # passenger in pick-up location
            pi, pj = self.env.locs[pass_idx]
            out[1 + pi][2 * pj + 1] += f'_P'
            out[1 + taxi_row][2 * taxi_col + 1] += '_T'
        else:
            # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] += '_T_P'

        # convert to indexes and return array
        return np.array([[MAP2INT[c] for c in row] for row in out], dtype=np.uint8)

    def passenger_at(self, loc: str):
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.decode(self.s)
        return PASS_COLORS[pass_idx] == loc

    def is_destination(self, loc: str):
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.decode(self.s)
        return COLORS[dest_idx] == loc

    def taxi_in(self, loc: str):
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.decode(self.s)
        taxi_loc = (taxi_row, taxi_col)
        color_idx = COLORS.index(loc)
        return self.env.locs[color_idx] == taxi_loc

    def _get_percept_vector(self) -> np.ndarray:
        out = np.full(len(COLORS) * 3 + 1, fill_value=False, dtype=bool)
        idx = 0
        for c in PASS_COLORS:
            out[idx] = self.passenger_at(c)
            idx += 1
        for c in COLORS:
            out[idx] = self.is_destination(c)
            idx += 1
        for c in COLORS:
            out[idx] = self.taxi_in(c)
            idx += 1
        assert 2 <= sum(out) <= 3
        return out

    def state_transition(self, action: str) -> bool:

        if action == ACT_PICK_SYMB:
            act = 4
        elif action == ACT_DROP_SYMB:
            act = 5
        else:
            # move action selected from shortest path to dest color loc
            dest_color = action.replace(f'{ACT_MOVE_SYMB}_', '')
            color_idx = COLORS.index(dest_color)
            dest_loc = self.env.locs[color_idx]
            taxi_row, taxi_col, pass_idx, dest_idx = self.env.decode(self.s)
            cur_loc = (taxi_row, taxi_col)
            act = self.paths[dest_loc][cur_loc]

        # step
        self.s, reward, done, trunc, info = self.env.step(act)

        # update histories
        self.s_h.append(self._get_state_array())
        self.a_h.append(ACT2INT[action])
        self.p_v_h.append(self._get_percept_vector())

        return done or trunc  # end of episode
