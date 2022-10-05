import numpy as np
from typing import List, Sequence, Callable, Optional, Tuple, Dict, Union
from taxi_env.taxi_env import COLORS, TaxiEnv, TAXI_SYMB, PASS_COLORS, ACTION_LIST

MAX_RECURSION = 100

# SYMBOLS =================================================
PERC_PASS_SYMB = 'PASSENGER_LOC'
PERC_DEST_SYMB = 'DESTINATION'
PERC_TAXI_SYMB = 'TAXI_LOC'

OP_LOG_NOT_SYMB = 'NOT'
OP_LOG_AND_SYMB = 'AND'
OP_LOG_OR_SYMB = 'OR'

OP_CTRL_IF_SYMB = 'IF'
OP_CTRL_ELSE_SYMB = 'ELSE'

LANG_L_PAREN_SYMB = '('
LANG_R_PAREN_SYMB = ')'

PROGRAM_TOKENS = [LANG_L_PAREN_SYMB, LANG_R_PAREN_SYMB,
                  OP_CTRL_IF_SYMB, OP_CTRL_ELSE_SYMB,
                  OP_LOG_NOT_SYMB, OP_LOG_AND_SYMB, OP_LOG_OR_SYMB,
                  PERC_PASS_SYMB, PERC_DEST_SYMB, PERC_TAXI_SYMB]

INT2TOKEN = PROGRAM_TOKENS + ACTION_LIST + PASS_COLORS
TOKEN2INT = {v: i for i, v in enumerate(INT2TOKEN)}


def str2int_seq(string: str) -> List[int]:
    return [TOKEN2INT[t] for t in string.split()]


def int_seq2str(int_seq: Sequence[int]) -> str:
    return ' '.join([INT2TOKEN[i] for i in int_seq])


def token_dim() -> int:
    return len(INT2TOKEN)


# def action_str2int_seq(string: str) -> List[int]:
#     return [ACT_TOKEN2INT[t] for t in string.split()]
#
#
# def action_int_seq2str(int_seq: Sequence[int]) -> str:
#     return ' '.join([ACT_INT2TOKEN[i] for i in int_seq])
#
#
# def action_token_dim() -> int:
#     return len(ACT_INT2TOKEN)


# SYNTAX ==================================================

Action = str
Color = str
RetType = Optional[Union[Action, Color, bool]]
ExecFunc = Callable[[TaxiEnv, int], RetType]
Token = Optional[ExecFunc]
Tokens = List[Token]
Rule = List[Tuple[str, ExecFunc]]

FN_STMT = 'stmt'
FN_IF_ELSE = 'if_else_stmt'
FN_ACT = 'action_stmt'
FN_COND = 'cond_stmt'
FN_PERCEPT = 'percept_stmt'
FN_COLOR_LOC = 'color_loc_stmt'
FN_TAXI_AS_LOC = 'loc_taxi_stmt'

PARSE_RULES: List[Tuple[str, Callable[[Tokens], Rule]]] = []
GEN_RULES: Dict[str, List[str]] = {}


def check_and_apply(queue: List[Tuple[str, Optional[ExecFunc]]], rule: Tuple[str, Callable[[Tokens], Rule]]):
    r = rule[0].split()
    l = len(r)
    if len(queue) >= l:
        t = queue[-l:]
        if list(list(zip(*t))[0]) == r:
            new_t = rule[1](list(list(zip(*t))[1]))
            del queue[-l:]
            queue.extend(new_t)
            return True
    return False


def parse(program: str) -> Optional[ExecFunc]:
    p_tokens = program.split()[::-1]
    queue = []
    applied = False
    while len(p_tokens) > 0 or len(queue) != 1:
        if applied:
            applied = False
        else:
            queue.append((p_tokens.pop(), None))
        for rule in PARSE_RULES:
            applied = check_and_apply(queue, rule)
            if applied:
                break
        if not applied and len(p_tokens) == 0:  # error parsing
            return None
    return queue[0][1]


def r_stmt(t: Tokens) -> Rule:
    stmt = t[0]

    def fn(world: TaxiEnv, n: int) -> RetType:
        if n > MAX_RECURSION:
            return None
        return stmt(world, n + 1)

    return [(FN_STMT, fn)]


# general statements can be either an action (terminal) or control expression
PARSE_RULES.append((FN_ACT, r_stmt))
PARSE_RULES.append((FN_IF_ELSE, r_stmt))
GEN_RULES[FN_STMT] = [FN_ACT, FN_IF_ELSE]


def create_r_const(val: RetType, stmt_key: str):
    def r_value(t: Tokens) -> Rule:
        def fn(world: TaxiEnv, n: int) -> Optional[Color]:
            if n > MAX_RECURSION:
                return None
            return val

        return [(stmt_key, fn)]

    return r_value


GEN_RULES[FN_ACT] = []
for action in ACTION_LIST:
    PARSE_RULES.append((action, create_r_const(action, FN_ACT)))
    GEN_RULES[FN_ACT].append(action)


def r_if_else(t: Tokens) -> Rule:
    cond, stmt1, stmt2 = t[2], t[5], t[9]  # IF ( cond ) ( stmt ) ELSE ( stmt )

    def fn(world: TaxiEnv, n: int) -> RetType:
        if n > MAX_RECURSION:
            return None
        c = cond(world, n + 1)
        if c is None:
            return None
        if c:
            return stmt1(world, n + 1)
        else:
            return stmt2(world, n + 1)

    return [(FN_IF_ELSE, fn)]


if_stmt = f'{OP_CTRL_IF_SYMB} {LANG_L_PAREN_SYMB} {FN_COND} {LANG_R_PAREN_SYMB} ' \
          f'{LANG_L_PAREN_SYMB} {FN_STMT} {LANG_R_PAREN_SYMB} ' \
          f'{OP_CTRL_ELSE_SYMB} {LANG_L_PAREN_SYMB} {FN_STMT} {LANG_R_PAREN_SYMB}'
PARSE_RULES.append((if_stmt, r_if_else))
GEN_RULES[FN_IF_ELSE] = [if_stmt]


def r_cond(t: Tokens) -> Rule:
    cond = t[0]

    def fn(world: TaxiEnv, n: int) -> Optional[bool]:
        if n > MAX_RECURSION:
            return None
        return cond(world, n + 1)

    return [(FN_COND, fn)]


PARSE_RULES.append((FN_PERCEPT, r_cond))
GEN_RULES[FN_COND] = [FN_PERCEPT]


def r_not_cond(t: Tokens) -> Rule:
    cond = t[2]  # not ( cond )

    def fn(world: TaxiEnv, n: int) -> Optional[bool]:
        if n > MAX_RECURSION:
            return None
        c = cond(world, n + 1)
        if c is None:
            return None
        return not c

    return [(FN_COND, fn)]


not_cond_stmt = f'{OP_LOG_NOT_SYMB} {LANG_L_PAREN_SYMB} {FN_COND} {LANG_R_PAREN_SYMB}'
PARSE_RULES.append((not_cond_stmt, r_not_cond))
GEN_RULES[FN_COND].append(not_cond_stmt)


def r_and_cond(t: Tokens) -> Rule:
    cond1, cond2 = t[1], t[3]  # ( cond and cond )

    def fn(world: TaxiEnv, n: int) -> Optional[bool]:
        if n > MAX_RECURSION:
            return None
        c = cond1(world, n + 1)
        if c is None:
            return None
        if not c:
            return False
        return cond2(world, n + 1)

    return [(FN_COND, fn)]


and_stmt = f'{LANG_L_PAREN_SYMB} {FN_COND} {OP_LOG_AND_SYMB} {FN_COND} {LANG_R_PAREN_SYMB}'
PARSE_RULES.append((and_stmt, r_and_cond))
GEN_RULES[FN_COND].append(and_stmt)


def r_or_cond(t: Tokens) -> Rule:
    cond1, cond2 = t[1], t[3]  # ( cond or cond )

    def fn(world: TaxiEnv, n: int) -> Optional[bool]:
        if n > MAX_RECURSION:
            return None
        c = cond1(world, n + 1)
        if c is None:
            return None
        if c:
            return True
        return cond2(world, n + 1)

    return [(FN_COND, fn)]


or_stmt = f'{LANG_L_PAREN_SYMB} {FN_COND} {OP_LOG_OR_SYMB} {FN_COND} {LANG_R_PAREN_SYMB}'
PARSE_RULES.append((or_stmt, r_or_cond))
GEN_RULES[FN_COND].append(or_stmt)


def r_percept_pass_loc(t: Tokens) -> Rule:
    loc = t[1]  # PASSENGER_AT pass_loc

    def fn(world: TaxiEnv, n: int) -> Optional[bool]:
        if n > MAX_RECURSION:
            return None
        return world.passenger_at(loc(world, n + 1))

    return [(FN_PERCEPT, fn)]


GEN_RULES[FN_PERCEPT] = []
pass_loc_stmt = f'{PERC_PASS_SYMB} {FN_COLOR_LOC}'
PARSE_RULES.append((pass_loc_stmt, r_percept_pass_loc))
GEN_RULES[FN_PERCEPT].append(pass_loc_stmt)

pass_loc_stmt = f'{PERC_PASS_SYMB} {FN_TAXI_AS_LOC}'
PARSE_RULES.append((pass_loc_stmt, r_percept_pass_loc))
GEN_RULES[FN_PERCEPT].append(pass_loc_stmt)


def r_percept_dest_loc(t: Tokens) -> Rule:
    loc = t[1]  # IS_DESTINATION loc

    def fn(world: TaxiEnv, n: int) -> Optional[bool]:
        if n > MAX_RECURSION:
            return None
        return world.is_destination(loc(world, n + 1))

    return [(FN_PERCEPT, fn)]


dest_stmt = f'{PERC_DEST_SYMB} {FN_COLOR_LOC}'
PARSE_RULES.append((dest_stmt, r_percept_dest_loc))
GEN_RULES[FN_PERCEPT].append(dest_stmt)


def r_percept_taxi_loc(t: Tokens) -> Rule:
    loc = t[1]  # TAXI_IN loc

    def fn(world: TaxiEnv, n: int) -> Optional[bool]:
        if n > MAX_RECURSION:
            return None
        return world.taxi_in(loc(world, n + 1))

    return [(FN_PERCEPT, fn)]


taxi_loc_stmt = f'{PERC_TAXI_SYMB} {FN_COLOR_LOC}'
PARSE_RULES.append((taxi_loc_stmt, r_percept_taxi_loc))
GEN_RULES[FN_PERCEPT].append(taxi_loc_stmt)

GEN_RULES[FN_COLOR_LOC] = []
for color in COLORS:
    PARSE_RULES.append((color, create_r_const(color, FN_COLOR_LOC)))
    GEN_RULES[FN_COLOR_LOC].append(color)

PARSE_RULES.append((TAXI_SYMB, create_r_const(TAXI_SYMB, FN_TAXI_AS_LOC)))
GEN_RULES[FN_TAXI_AS_LOC] = [TAXI_SYMB]


# PROG GENERATION =========================================

class TaxiProgramGenerator(object):

    def __init__(self, max_depth: int, max_length: int, seed: int = 123):
        self.max_depth = max_depth
        self.max_length = max_length
        self.rng = np.random.RandomState(seed)

    def random_code(self, ) -> str:
        codes = self.random_expand_token(FN_STMT, 0, 0)
        return ' '.join(codes)

    def random_expand_token(self, token: str, depth: int, length: int) -> List[str]:
        if token == FN_STMT and (length + 1 >= self.max_length or depth + 1 >= self.max_depth):
            candidates = [FN_ACT]  # only space for an action (terminal) statement
        else:
            candidates = GEN_RULES[token]
        probs = np.array([len(candidate.split()) for candidate in candidates])
        probs = np.max(probs) - probs
        prob_sum = np.sum(probs)
        probs = (probs / prob_sum) if prob_sum > 0 else (probs + 1 / len(probs))
        sample = self.rng.choice(candidates, p=probs)
        expansion = sample.split()
        codes = []
        for t in expansion:
            if t in GEN_RULES:
                is_control = t == FN_IF_ELSE
                sub_codes = self.random_expand_token(t, depth + 1 if is_control else depth, length)
                length += len(sub_codes)
                codes.extend(sub_codes)
            else:
                length += 1
                codes.append(t)

        return codes
