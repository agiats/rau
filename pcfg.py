import random


class PCFG:
    """
    PCFG to sample sentences from
    """
    def __init__(self, grammar_file):
        self.rules = None
        self.change_rules = None
        self.load_rules(grammar_file)

    def load_rules(self, grammar_file):
        new_rules = {}
        change = {}
        g_file = open(grammar_file, 'r')
        lines = g_file.readlines()
        for l in lines:
            if l.startswith(('#', " ", "\t", "\n")) or len(l) < 1:
                continue
            else:
                if l.find("#") != -1:
                    l = l[:l.find("#")]
                idx = -1
                if len(l.rstrip().split("\t")) == 3:
                    weight, lhs, rhs = l.rstrip().split("\t")
                elif len(l.rstrip().split("\t")) == 4:
                    weight, lhs, rhs, idx = l.rstrip().split("\t")
                if lhs not in new_rules.keys():
                    new_rules[lhs] = []
                poss_rhs = new_rules[lhs]
                poss_rhs.append([rhs, float(weight)])
                if idx != -1:
                    change[lhs + "\t" + rhs] = idx
        for lhs, poss in new_rules.items():
            total = 0
            for rhs in poss:
                total += rhs[1]
            for rhs in poss:
                rhs[1] /= total
        self.rules = new_rules
        self.change_rules = change

    def sample_sentence(self, max_expansions, bracketing):
        self.expansions = 0
        done = False
        sent = ["ROOT"]
        idx = 0
        while not done:
            if sent[idx] not in self.rules.keys():
                idx += 1
                if idx >= len(sent):
                    done = True
                continue
            else:
                replace, change_idx = self.expand(sent[idx])
                if bracketing:
                    if change_idx == -1:
                        sent = (sent[:idx]
                            + ["(", sent[idx]] + replace + [")"]
                            + sent[idx + 1:])
                    else:
                        sent = (sent[:idx]
                            + ["(", change_idx + sent[idx]] + replace + [")"]
                            + sent[idx + 1:])
                else:
                    sent = sent[:idx] + replace  + sent[idx + 1:]
                self.expansions += 1
                if bracketing:
                    idx += 2
                if self.expansions > max_expansions:
                    done = True
                if idx >= len(sent):
                    done = True
        if self.expansions > max_expansions:
            print("Max expansions reached")
            return None
            # for idx in range(len(sent)):
            #     if not bracketing:
            #         if sent[idx] in self.rules.keys():
            #             sent[idx] = "..."
            #     else:
            #         if sent[idx] in self.rules.keys() and sent[idx - 1] != "(":
            #             sent[idx] = "..."
        return ' '.join(sent)

    def expand(self, symbol):
        poss = self.rules[symbol]
        sample = random.random()
        val = 0.0
        rhs = ""
        idx = -1
        for p in poss:
            val += p[1]
            if sample <= val:
                if symbol + "\t" + p[0] in self.change_rules.keys():
                    idx = self.change_rules[symbol + "\t" + p[0]]
                rhs = p[0]
                break
        return rhs.split(" "), idx


class PCFGDeterministicShuffle(PCFG):
    def __init__(self, grammar_file, seed=42):
        super().__init__(grammar_file)
        self.seed = seed

    def sample_sentence(self, max_expansions, bracketing):
        sent = super().sample_sentence(max_expansions, bracketing)
        # Save current random state
        state = random.getstate()

        tokens = sent.split(' ')
        eos = tokens.pop() if tokens[-1] == '[eos]' else None

        # Use seed for shuffling
        random.seed(self.seed)
        random.shuffle(tokens)

        # Restore random state after shuffling
        random.setstate(state)

        if eos:
            tokens.append(eos)
        return ' '.join(tokens)

class PCFGNonDeterministicShuffle(PCFG):
    def __init__(self, grammar_file):
        super().__init__(grammar_file)

    def sample_sentence(self, max_expansions, bracketing):
        sent = super().sample_sentence(max_expansions, bracketing)

        tokens = sent.split(' ')
        eos = tokens.pop() if tokens[-1] == '[eos]' else None

        random.shuffle(tokens)

        if eos:
            tokens.append(eos)
        return ' '.join(tokens)

class PCFGLocalShuffle(PCFG):
    def __init__(self, grammar_file, window=5, seed=42):
        super().__init__(grammar_file)
        self.window = window
        self.seed = seed

    def sample_sentence(self, max_expansions, bracketing):
        sent = super().sample_sentence(max_expansions, bracketing)

        tokens = sent.split(' ')
        eos = tokens.pop() if tokens[-1] == '[eos]' else None

        # Save current random state
        state = random.getstate()

        # Use seed for shuffling
        random.seed(self.seed)
        shuffled_tokens = []
        for i in range(0, len(tokens), self.window):
            batch = tokens[i:min(i+self.window, len(tokens))].copy()
            random.shuffle(batch)
            shuffled_tokens.extend(batch)

        # Restore random state after shuffling
        random.setstate(state)

        if eos:
            shuffled_tokens.append(eos)
        return ' '.join(shuffled_tokens)

class PCFGEvenOddShuffle(PCFG):
    def __init__(self, grammar_file):
        super().__init__(grammar_file)

    def sample_sentence(self, max_expansions, bracketing):
        sent = super().sample_sentence(max_expansions, bracketing)

        tokens = sent.split(' ')
        eos = tokens.pop() if tokens[-1] == '[eos]' else None

        even = [tok for i, tok in enumerate(tokens) if i % 2 == 0]
        odd = [tok for i, tok in enumerate(tokens) if i % 2 != 0]
        shuffled = even + odd

        if eos:
            shuffled.append(eos)
        return ' '.join(shuffled)
