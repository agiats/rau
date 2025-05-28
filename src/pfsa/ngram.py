from fsa import PFSA


class NGram(PFSA):
    def __init__(self, n_symbols: int, n: int):
        self.n = n
        n_states = sum(n_symbols**i for i in range(self.n))
        self.EOS, self.BOS = n_states, n_states + 1

        super().__init__(n_states, n_symbols)
