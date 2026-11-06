"""Test-day SGF sanity check: makes every game safe for the KataGo extraction and the move
tokeniser.

The extractors read moves with the regex ``;([BW])\\[([a-s]{0,2})\\]`` and send them to
KataGo, which rejects a whole query when a move is played on an occupied point or is suicide, and
whose policy marks an immediate ko recapture as illegal (extract_rank_features.py then raises).
One such game stops its shard (with many games in flight the extraction even hangs), so
``sanitize`` returns, for every SGF:

* the original string unchanged when the regex already sees exactly the main line of a legal
  19x19 game without setup stones (all training and rehearsal data);
* otherwise a canonical ``(;SZ[19];B[..];W[..]...)`` rebuilt from a real SGF parse of the main
  line: setup stones (AB / AW, e.g. handicap) become leading moves, ``tt`` passes become ``[]``,
  whitespace / lower-case / comment oddities disappear, and the game is cut before its first
  illegal move (occupied point, suicide, immediate ko recapture, off-board coordinate);
* ``usable = False`` when nothing sensible can be extracted (no SGF, board size not 19).
"""
import dataclasses
import re

BOARD = 19
MOVE_PATTERN = re.compile(r";([BW])\[([a-s]{0,2})\]")  # the extractors' own pattern
Move = tuple[str, tuple[int, int] | None]


@dataclasses.dataclass
class Checked:
    """Result of ``sanitize``."""

    sgf: str
    usable: bool
    issues: list[str]
    n_moves: int


def _value(sgf: str, start: int) -> tuple[str, int]:
    """The property value opening at sgf[start] == '[' (escapes resolved) and the index after its
    closing bracket."""
    j, value = start + 1, []
    while j < len(sgf) and sgf[j] != "]":
        if sgf[j] == "\\" and j + 1 < len(sgf):
            j += 1
        value.append(sgf[j])
        j += 1
    return "".join(value), j + 1


@dataclasses.dataclass
class _Variations:
    """Which parentheses belong to the main line (the first branch at every depth)."""

    depth: int = 0
    skip_from: int | None = None  # depth of the side variation being skipped
    opened: set[int] = dataclasses.field(default_factory=set)

    def open(self) -> None:
        """'('"""
        self.depth += 1
        if self.skip_from is None and self.depth in self.opened:
            self.skip_from = self.depth
        self.opened.add(self.depth)
        self.opened.discard(self.depth + 1)

    def close(self) -> None:
        """')'"""
        if self.skip_from == self.depth:
            self.skip_from = None
        self.depth -= 1


def parse_main_line(sgf: str) -> tuple[dict[str, list[str]], list[dict[str, list[str]]]]:
    """Root properties and the following nodes of the main variation (first branch everywhere).
    Property names keep their upper-case letters only (FF3 'AddBlack' style -> 'AB')."""
    nodes: list[dict[str, list[str]]] = []
    branches = _Variations()
    ident, i = "", 0
    while i < len(sgf):
        ch = sgf[i]
        if ch == "[":
            value, i = _value(sgf, i)
            if branches.skip_from is None and nodes:
                key = "".join(c for c in ident if c.isupper()) or ident.upper()
                nodes[-1].setdefault(key, []).append(value)
            continue
        if ch == "(":
            branches.open()
        elif ch == ")":
            branches.close()
        elif ch == ";":
            if branches.skip_from is None:
                nodes.append({})
            ident = ""
        elif ch.isalpha():
            ident = ident + ch if (i > 0 and sgf[i - 1].isalpha()) else ch
        i += 1
    if not nodes:
        return {}, []
    return nodes[0], nodes[1:]


def _point(value: str) -> tuple[int, int] | None | str:
    """(x, y), None for a pass ('' or 'tt'), or 'bad' for an off-board coordinate."""
    value = value.strip().lower()
    if value in ("", "tt"):
        return None
    if len(value) == 2 and all("a" <= c <= "s" for c in value):
        return ord(value[0]) - 97, ord(value[1]) - 97
    return "bad"


def _group(board: dict, start: tuple[int, int]) -> tuple[set, int]:
    """Stones of the group at start and its number of liberties."""
    colour = board[start]
    stones, liberties, stack = {start}, set(), [start]
    while stack:
        x, y = stack.pop()
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nb[0] < BOARD and 0 <= nb[1] < BOARD):
                continue
            stone = board.get(nb)
            if stone is None:
                liberties.add(nb)
            elif stone == colour and nb not in stones:
                stones.add(nb)
                stack.append(nb)
    return stones, len(liberties)


def legal_prefix(moves: list[Move]) -> tuple[int, str]:
    """Number of leading moves that are legal under KataGo's Japanese rules (simple ko, no
    suicide), and the reason the next one is not ('' if all are legal)."""
    board: dict[tuple[int, int], str] = {}
    ko = None
    for n, (colour, point) in enumerate(moves):
        if point is None:
            ko = None
            continue
        if point in board:
            return n, f"move {n + 1} on an occupied point"
        if point == ko:
            return n, f"move {n + 1} retakes a ko immediately"
        board[point] = colour
        captured: list[tuple[int, int]] = []
        x, y = point
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if board.get(nb, colour) != colour:
                stones, libs = _group(board, nb)
                if libs == 0:
                    captured.extend(stones)
                    for s in stones:
                        del board[s]
        own, libs = _group(board, point)
        if libs == 0:
            return n, f"move {n + 1} is suicide"
        ko = captured[0] if len(captured) == 1 and len(own) == 1 and libs == 1 else None
    return len(moves), ""


def canonical(moves: list[Move]) -> str:
    """A minimal SGF the extractors read exactly as ``moves``."""
    body = "".join(f";{c}[{'' if p is None else chr(97 + p[0]) + chr(97 + p[1])}]"
                   for c, p in moves)
    return f"(;SZ[{BOARD}]{body})"


def _moves(root: dict[str, list[str]], nodes: list[dict[str, list[str]]],
           issues: list[str]) -> list[Move]:
    """Setup stones as leading moves (handicap as consecutive black moves), then the main line;
    the game ends before an off-board coordinate."""
    moves: list[Move] = []
    for colour in ("B", "W"):
        moves += [(colour, p) for p in map(_point, root.get("A" + colour, []))
                  if isinstance(p, tuple)]
    if moves:
        issues.append(f"{len(moves)} setup stones turned into leading moves")
    for node in [root] + nodes:
        for colour in ("B", "W"):
            for value in node.get(colour, [])[:1]:
                point = _point(value)
                if isinstance(point, str):
                    issues.append(f"off-board coordinate {value!r}")
                    return moves
                moves.append((colour, point))
    return moves


def sanitize(sgf: object) -> Checked:
    """See the module docstring."""
    if not isinstance(sgf, str) or not sgf.strip():
        return Checked("", False, ["empty SGF"], 0)
    root, nodes = parse_main_line(sgf.lstrip("\ufeff"))
    size = (root.get("SZ") or [str(BOARD)])[0].split(":")[0].strip()
    if size != str(BOARD):
        return Checked(sgf, False, [f"board size {size}"], 0)
    issues: list[str] = []
    moves = _moves(root, nodes, issues)
    n_legal, reason = legal_prefix(moves)
    if n_legal < len(moves):
        issues.append(f"cut after {n_legal} of {len(moves)} moves: {reason}")
        moves = moves[:n_legal]
    if not moves:
        issues.append("no moves (kept: no decisions to extract)")
    seen = [(c, None if not p else (ord(p[0]) - 97, ord(p[1]) - 97))
            for c, p in MOVE_PATTERN.findall(sgf)]
    if not issues and seen == moves:
        return Checked(sgf, True, [], len(moves))
    if not issues:
        issues.append("non-standard SGF text normalised")
    return Checked(canonical(moves), True, issues, len(moves))
