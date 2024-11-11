import math
import collections
from .util import group_by
from dataclasses import dataclass


class NoParses(ValueError):
    pass


def compute_parse_grammar_log_probability(parses):
    rules_by_left = group_by(parses.rules, key=lambda r: r.left)
    visiting = set()
    cache = collections.defaultdict(float)

    def recurse(symbol):
        if symbol.is_terminal:
            # This "terminal" is a PCFG rule. Its probability is the
            # probability of the rule.
            return symbol.value.log_probability
        else:
            if symbol in visiting:
                # TODO This can probably be handled in the general case.
                raise ValueError("parse grammar has recursion")
            result = cache.get(symbol)
            if result is None:
                rules = rules_by_left.get(symbol)
                if rules is None:
                    raise NoParses("a nonterminal produces no parses")
                else:
                    visiting.add(symbol)
                    result = logspace_sum(
                        logspace_product(recurse(X) for X in rule.right)
                        for rule in rules
                    )
                    visiting.remove(symbol)
                cache[symbol] = result
            return result

    return recurse(parses.start)


logspace_product = sum


def logspace_sum(values):
    return math.log(sum(math.exp(x) for x in values))


def string_log_probability(parser, string):
    parses = parser.to_parse_grammar(string)
    try:
        ll = compute_parse_grammar_log_probability(parses)
    except NoParses:
        return -math.inf
    else:
        return ll


def compute_lower_bound_perplexity(sampler, num_valid_lengths, samples):
    """
    Compute the lower bound perplexity of a specific set of strings generated
    by a PCFG subject to length constraints.
    """
    # Let p_G be the probability distribution over strings defined by the PCFG.
    # For a given string w with length l, we compute p(w) as
    # p(w) = p(l) p_G(w) / \sum_{w' where |w'| = l } p_G(w').
    parts = compute_lower_bound_parts(sampler, samples)
    return parts_to_perplexity(parts, num_valid_lengths)


@dataclass
class Parts:
    total_neg_log_prob: int
    total_len: int
    num_samples: int


def compute_lower_bound_parts(sampler, samples):
    r"""
    Compute \sum_w log p(w) / \sum_{w' where |w'| = |w|} p_G(w') and
    \sum_w |w| for w in samples.
    """
    total_neg_log_prob = 0.0
    total_len = 0
    num_samples = 0
    for sample in samples:
        total_neg_log_prob -= sampler.log_probability_given_length(sample)
        # Remember to include EOS in the denominator for consistency with
        # model perplexity.
        total_len += len(sample) + 1
        num_samples += 1
    return Parts(total_neg_log_prob, total_len, num_samples)


def parts_to_perplexity(parts, num_valid_lengths):
    length_neg_log_prob = math.log(num_valid_lengths)
    neg_log_prob = parts.total_neg_log_prob + length_neg_log_prob * parts.num_samples
    return math.exp(neg_log_prob / parts.total_len)


def compute_cross_entropy_diff(perplexity, lower_bound_perplexity):
    return math.log(perplexity) - math.log(lower_bound_perplexity)
