from typing import List, Sequence, Callable, Optional, Tuple
from taxi_env.taxi_env import COLORS, TaxiEnv, TAXI_SYMB, PASS_COLORS, ACTION_LIST

MAX_FUNC_CALL = 100

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


def action_token_dim() -> int:
    return len(ACT_INT2TOKEN)


def check_and_apply(queue: List[Tuple[str, Callable]], rule):
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


def parse(program: str) -> Tuple[Optional[Callable], bool]:
    p_tokens = program.split()[::-1]
    queue = []
    applied = False
    while len(p_tokens) > 0 or len(queue) != 1:
        if applied:
            applied = False
        else:
            queue.append((p_tokens.pop(), None))
        for rule in RULES:
            applied = check_and_apply(queue, rule)
            if applied:
                break
        if not applied and len(p_tokens) == 0:  # error parsing
            return None, False
    return queue[0][1], True


# SYNTAX ==================================================
FN_STMT = 'stmt'
FN_IF_ELSE = 'if_else_stmt'
FN_ACT = 'action'
FN_COND = 'cond'
FN_PERCEPT = 'percept'
FN_COLOR_LOC = 'color_loc'
FN_TAXI_AS_LOC = 'loc_taxi'

RULES: List[Tuple[str, Callable]] = []


def r_stmt(t: List[Optional[Callable]]):
    stmt = t[0]

    def fn(world: TaxiEnv, n: int):
        if n > MAX_FUNC_CALL:
            return world, n, False
        return stmt(world, n + 1)

    return [(FN_STMT, fn)]


# general statements can be either an action (terminal) or control expression
RULES.append((FN_ACT, r_stmt))
RULES.append((FN_IF_ELSE, r_stmt))


def create_r_action(action):
    def r_action(t):
        def fn(world: TaxiEnv, n: int):
            if n > MAX_FUNC_CALL:
                return world, n, False
            try:
                world.state_transition(action)
            except:
                return world, n, False
            else:
                return world, n, True

        return [(FN_ACT, fn)]

    return r_action


for action in ACTION_LIST:
    RULES.append((action, create_r_action(action)))


def r_if_else(t):
    cond, stmt1, stmt2 = t[2], t[5], t[9]  # IF ( cond ) ( stmt ) ELSE ( stmt )

    def fn(world: TaxiEnv, n: int):
        if n > MAX_FUNC_CALL:
            return world, n, False
        world, n, s, c = cond(world, n + 1)
        if not s:
            return world, n, s
        if c:
            return stmt1(world, n)
        else:
            return stmt2(world, n)

    return [(FN_IF_ELSE, fn)]


RULES.append((f'{OP_CTRL_IF_SYMB} {LANG_L_PAREN_SYMB} {FN_COND} {LANG_R_PAREN_SYMB} ' \
              f'{LANG_L_PAREN_SYMB} {FN_STMT} {LANG_R_PAREN_SYMB} ' \
              f'{OP_CTRL_ELSE_SYMB} {LANG_L_PAREN_SYMB} {FN_STMT} {LANG_R_PAREN_SYMB}', r_if_else))


def r_cond(t):
    cond = t[0]

    def fn(world: TaxiEnv, n: int):
        if n > MAX_FUNC_CALL:
            return world, n, False, False
        return cond(world, n)

    return [(FN_COND, fn)]


RULES.append((FN_PERCEPT, r_cond))


def r_not_cond(t):
    cond = t[2]  # not ( cond )

    def fn(world: TaxiEnv, n: int):
        if n > MAX_FUNC_CALL:
            return world, n, False, False
        world, n, s, c = cond(world, n)
        return world, n, s, not c

    return [(FN_COND, fn)]


RULES.append((f'{OP_LOG_NOT_SYMB} {LANG_L_PAREN_SYMB} {FN_COND} {LANG_R_PAREN_SYMB}', r_not_cond))


def r_and_cond(t):
    cond1, cond2 = t[1], t[3]  # ( cond and cond )

    def fn(world: TaxiEnv, n: int):
        if n > MAX_FUNC_CALL:
            return world, n, False, False
        world, n, s, c = cond1(world, n)
        if not c:
            return world, n, s, False
        return cond2(world, n)

    return [(FN_COND, fn)]


RULES.append((f'{LANG_L_PAREN_SYMB} {FN_COND} {OP_LOG_AND_SYMB} {FN_COND} {LANG_R_PAREN_SYMB}', r_and_cond))


def r_or_cond(t):
    cond1, cond2 = t[1], t[3]  # ( cond or cond )

    def fn(world: TaxiEnv, n: int):
        if n > MAX_FUNC_CALL:
            return world, n, False, False
        world, n, s, c = cond1(world, n)
        if c:
            return world, n, s, True
        return cond2(world, n)

    return [(FN_COND, fn)]


RULES.append((f'{LANG_L_PAREN_SYMB} {FN_COND} {OP_LOG_OR_SYMB} {FN_COND} {LANG_R_PAREN_SYMB}', r_or_cond))


def r_percept_pass_loc(t):
    loc = t[1]  # PASSENGER_AT pass_loc

    def fn(world: TaxiEnv, n: int):
        if n > MAX_FUNC_CALL:
            return world, n, False, False
        c = world.passenger_at(loc())
        return world, n, True, c

    return [(FN_PERCEPT, fn)]


RULES.append((f'{PERC_PASS_SYMB} {FN_COLOR_LOC}', r_percept_pass_loc))
RULES.append((f'{PERC_PASS_SYMB} {FN_TAXI_AS_LOC}', r_percept_pass_loc))


def r_percept_dest_loc(t):
    loc = t[1]  # IS_DESTINATION loc

    def fn(world: TaxiEnv, n: int):
        if n > MAX_FUNC_CALL:
            return world, n, False, False
        c = world.is_destination(loc())
        return world, n, True, c

    return [(FN_PERCEPT, fn)]


RULES.append((f'{PERC_DEST_SYMB} {FN_COLOR_LOC}', r_percept_dest_loc))


def r_percept_taxi_loc(t):
    loc = t[1]  # TAXI_IN loc

    def fn(world: TaxiEnv, n: int):
        if n > MAX_FUNC_CALL:
            return world, n, False, False
        c = world.taxi_in(loc())
        return world, n, True, c

    return [(FN_PERCEPT, fn)]


RULES.append((f'{PERC_TAXI_SYMB} {FN_COLOR_LOC}', r_percept_taxi_loc))

for color in COLORS:
    RULES.append((color, lambda t: [(FN_COLOR_LOC, lambda: color)]))

RULES.append((TAXI_SYMB, lambda t: [(FN_TAXI_AS_LOC, lambda: TAXI_SYMB)]))
