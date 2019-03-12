import random
import argparse
import itertools

DEFAULT_RC = 10
STR_PAD = 3

LEVEL_PS = (0.1, 0.25, 0.5)

I_MINE = 0
I_REVEALED = 1
I_EDGE = 2
I_SCORE = 3
I_HOOD = 4


class Validators(object):
    @staticmethod
    def input(s, n, label):
        try:
            i = int(s) - 1
        except ValueError:
            err = f'Please provide a numeric value for {label}'
            print(err)
            return None

        if i < 0 or i >= n:
            err = f'{label} number must be between 1 and {n}'
            print(err)
            return None

        return i


class Masks(object):
    @staticmethod
    def square(x):
        if x[I_REVEALED] == 1:
            s = 'x' if x[I_MINE] else ' '
        elif x[I_EDGE] == 1:
            s = str(x[I_SCORE])
        else:
            s = '_'

        return s.center(STR_PAD)

    @staticmethod
    def row(x):
        i, row = x
        i_ = str(i + 1).center(STR_PAD)
        row_ = ''.join(map(Masks.square, row))

        return f"{i_}| {row_} |{i_}"

    @staticmethod
    def field(x):
        n = len(x[0])

        header = f"   |{''.join(map(lambda x: str(x).center(STR_PAD), range(1, n+1)))}  |   "
        divider = '-' * len(header)
        rows = map(Masks.row, enumerate(x))

        s = '\n'.join([header, divider] + list(rows) + [divider, header])

        return s


class MineSweeper(object):
    def _mine(self):
        return int(random.random() < self._p)

    def _square(self, x):
        return [self._mine(), 0, 0, 0, None]

    def _field(self):
        m = self._m
        n = self._n

        field = itertools.product(range(m), range(n))
        field = map(self._square, field)
        field = list(itertools.zip_longest(*[field] * m))

        self.field = field

    def _hoods(self):
        m = self._m
        n = self._n

        for r in range(0, m):
            rs = range(max(0, r - 1), min(r + 1, m))

            for c in range(0, n):
                cs = range(max(0, c - 1), min(c + 1, n))
                square = self.field[r][c]
                square[I_HOOD] = tuple(itertools.product(rs, cs))

    def _scores(self):
        field = self.field

        for row in field:
            for square in filter(lambda x: x[I_MINE], row):
                homies = [field[r][c] for r, c in square[I_HOOD]]
                for homie in homies:
                    homie[I_SCORE] += 1

    def __init__(self, m, n, level=1):
        self._m = m
        self._n = n

        self._level = level
        self._p = LEVEL_PS[level - 1]

        self._field()
        self._hoods()
        self._scores()

    def _print(self):
        s = Masks.field(self.field)

        print(s)

    def reveal(self, r, c):
        m = self._m
        n = self._n

        square = self.field[r][c]

        if square[I_SCORE] != 0:
            square[I_EDGE] = 1
            return

        if square[I_REVEALED] == 1:
            return

        square[I_REVEALED] = 1

        for i in (max(0, r - 1), r, min(r + 1, m - 1)):
            for j in (max(0, c - 1), c, min(c + 1, n - 1)):
                if i == r and j == c:
                    continue

                self.reveal(i, j)

    def round(self):
        m = self._m
        n = self._n
        field = self.field

        r = input(f'row [1 to {m}]: ')
        c = input(f'column [1 to {n}]: ')

        r = Validators.input(r, m, 'row')
        c = Validators.input(c, n, 'column')

        if r is None or c is None:
            return True

        square = field[r][c]

        self.reveal(r, c)

        square[I_REVEALED] = 1

        self._print()

        if square[I_MINE] == 1:
            print('You LOST!')
            return False

        return True

    def play(self):
        self._print()

        while True:
            if not self.round():
                break


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rows", help="number of rows", type=int, default=DEFAULT_RC)
parser.add_argument("-c", "--cols", help="number of columns", type=int, default=DEFAULT_RC)
parser.add_argument("-l", "--level", help="game level (1-3)", type=int, default=1)


if __name__ == '__main__':
    args = parser.parse_args()

    m = args.rows
    n = args.cols
    v = args.level

    game = MineSweeper(m, n, v)
    game.play()
