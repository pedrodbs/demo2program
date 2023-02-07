import gym
import numpy as np
from typing import List, Dict, Optional
from dijkstra import Graph, DijkstraSPF
from gym.envs.toy_text.taxi import TaxiEnv as TaxiGym
from gym.wrappers import TimeLimit

# see description at: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
RENDER_MODE = 'ansi'  # 'rgb_array'

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
MAP_COLORS_TAXI = [f'{c}_T' for c in MAP_COLORS] + [f'{c}_T' for c in MAP_PASS]
MAP_TAXI = [' _T', ' _T_P']
# MAP_GRID = ['+', '-', ':', '|', ' ']
INT2MAP: List[str] = [' '] + MAP_COLORS + MAP_PASS + MAP_COLORS_TAXI + MAP_PASS_TAXI + \
                     MAP_DEST + MAP_DEST_TAXI + MAP_DEST_TAXI_PASS + MAP_DEST_PASS_TAXI + MAP_TAXI
MAP2INT: Dict[str, int] = {v: i for i, v in enumerate(INT2MAP)}


class TaxiEnv(object):

    def __init__(self, seed: int = 17, max_steps=50):
        self.seed = seed

        self.s_h: List[np.ndarray] = []
        self.a_h: List[int] = []
        self.p_v_h: List[np.ndarray] = []
        self.s: int = -1

        self.env: TaxiGym = TimeLimit(gym.make('Taxi-v3', render_mode=RENDER_MODE), max_episode_steps=max_steps)

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

    def init_game(self, s: Optional[int] = None, s_array: Optional[np.array] = None):
        if s is not None:
            self.s = self.env.s = s
            self.env.reset()
            self.env.lastaction = None
            self.env.taxi_orientation = 0
            if self.env.render_mode == 'human':
                self.env.render()
        elif s_array is not None:
            s = self._state_from_array(s_array)
            self.init_game(s=s)
            return
        else:
            # reset env
            self.s, *_ = self.env.reset(seed=self.seed)
            self.seed = None

        # initialize histories
        self.s_h = [self._array_from_state()]
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

    def _array_from_state(self) -> np.array:
        if RENDER_MODE in {'human', 'rgb_array'}:
            return self.env.render()

        # get symbol-based representation
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

        # convert to indexes, get one-hot encoding and return array
        out = np.array(out)
        out = out[1:-1, np.arange(1, out.shape[1], 2)]  # remove walls since they're static across environments
        out = np.array([[MAP2INT[c] for c in row] for row in out], dtype=np.uint8)
        res = np.eye(len(INT2MAP), dtype=bool)[out.reshape(-1)]
        res = res.reshape(list(out.shape) + [len(INT2MAP)])
        return res

    def _state_from_array(self, s_array: np.ndarray) -> int:
        taxi_symbs = [i for symb, i in MAP2INT.items() if '_T' in symb]
        taxi_locs = np.array(np.where(s_array[..., taxi_symbs])).flatten()

        assert taxi_locs.size == 3  # only one location possible for taxi
        taxi_row, taxi_col, _ = taxi_locs
        pass_symbs = [i for symb, i in MAP2INT.items() if '_P' in symb]
        pass_locs = np.array(np.where(s_array[..., pass_symbs])).flatten()
        assert pass_locs.size == 3  # only one location possible for passenger
        pass_row, pass_col, pass_symb_idx = pass_locs
        if '_T_P' in INT2MAP[pass_symbs[pass_symb_idx]]:
            pass_loc = PASS_COLORS.index(TAXI_SYMB)
        else:
            pass_loc = -1
            for i, loc in enumerate(self.env.locs):
                if pass_row == loc[0] and pass_col == loc[1]:
                    pass_loc = i
                    break
            assert pass_loc != -1

        dest_symbs = [i for symb, i in MAP2INT.items() if '_D' in symb]
        dest_locs = np.array(np.where(s_array[..., dest_symbs])).flatten()
        assert dest_locs.size == 3  # only one location possible for destination
        dest_row, dest_col, _ = dest_locs
        dest_idx = -1
        for i, loc in enumerate(self.env.locs):
            if dest_row == loc[0] and dest_col == loc[1]:
                dest_idx = i
                break
        assert dest_idx != -1
        s = self.env.encode(taxi_row, taxi_col, pass_loc, dest_idx)
        return s

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

    @property
    def num_percepts(self) -> int:
        return len(COLORS) * 3 + 1

    @property
    def percepts(self) -> List[str]:
        return [f'PassengerAt {c}' for c in PASS_COLORS] + \
               [f'IsDestination {c}' for c in COLORS] + \
               [f'TaxiIn {c}' for c in COLORS]

    def _get_percept_vector(self) -> np.ndarray:
        out = np.full(self.num_percepts, fill_value=False, dtype=bool)
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
        self.s_h.append(self._array_from_state())
        self.a_h.append(ACT2INT[action])
        self.p_v_h.append(self._get_percept_vector())

        return done or trunc  # end of episode
