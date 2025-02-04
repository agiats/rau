import rayuela
from rayuela.base.semiring import Boolean, ProductSemiring, Real, Semiring
from rayuela.base.state import PairState, State
from rayuela.base.symbol import Expr, Sym, ε, ε_1, ε_2, φ
from rayuela.cfg.nonterminal import NT, S
from rayuela.fsa.pathsum import Pathsum, Strategy
from rayuela.fsa.fsa import FSA


class LocalShuffledNgramModel:
    def __init__(self, ngram, perturbation_func):
        self.ngram = ngram
        self.permutation_func = perturbation_func

