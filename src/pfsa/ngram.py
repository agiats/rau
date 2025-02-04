import rayuela
from rayuela.base.semiring import Boolean, ProductSemiring, Real, Semiring
from rayuela.base.state import PairState, State
from rayuela.base.symbol import Expr, Sym, ε, ε_1, ε_2, φ
from rayuela.cfg.nonterminal import NT, S
from rayuela.fsa.pathsum import Pathsum, Strategy
from rayuela.fsa.fsa import FSA


class RandomNGramModel:
    def __init__(self, n: int, alpha: float, bos: Sym, eos: Sym):
        self.alphabet = set()
        self.n = n
        self.alpha = alpha
        self.bos = bos
        self.eos = eos
        self.model: FSA = FSA(R=Real)
        self.__build_model()


    def __build_model(ngram_model, n, alphabet, bos='[BOS]', eos='[EOS]'):
        """
        Build a Weighted FSA (Real semiring) capturing the distribution
        of an n-gram model, one-symbol-per-transition style.

        ngram_model: dict => ngram_model[context][symbol] = probability
        n: the 'n' in n-gram
        alphabet: list of symbols
        bos, eos: special tokens
        """
        fsa = FSA(R=Real)

        # define final "absorbing" state for after EOS
        q_final = State("<<FINAL>>")
        fsa.add_state(q_final)
        fsa.set_F(q_final, Real.one)

        # gather contexts
        contexts = list(ngram_model.keys())  # each is (n-1)-tuple or () if n=1
        context2state = {}
        for ctx in contexts:
            ctxName = str(ctx) if ctx else "()"
            q = State(ctxName)
            context2state[ctx] = q
            fsa.add_state(q)

        # start context
        if n>1:
            start_ctx = tuple([bos]*(n-1))
        else:
            start_ctx = ()

        if start_ctx not in context2state:
            qstart = State(str(start_ctx))
            context2state[start_ctx] = qstart
            fsa.add_state(qstart)
        else:
            qstart = context2state[start_ctx]
        fsa.set_I(qstart, Real.one)

        # define transitions
        for ctx in contexts:
            s_from = context2state[ctx]
            dist_dict = ngram_model[ctx]
            for symbol, prob in dist_dict.items():
                if prob <= 1e-15:
                    continue
                w = Real(prob)
                if symbol == eos:
                    # go to final
                    fsa.add_arc(s_from, Sym(symbol), q_final, w)
                else:
                    # next context
                    if n>1:
                        new_ctx = tuple(list(ctx[1:]) + [symbol]) if len(ctx)==(n-1) else (symbol,)
                    else:
                        new_ctx = ()
                    if new_ctx not in context2state:
                        qq = State(str(new_ctx))
                        context2state[new_ctx] = qq
                        fsa.add_state(qq)
                    s_to = context2state[new_ctx]
                    fsa.add_arc(s_from, Sym(symbol), s_to, w)

        return fsa
