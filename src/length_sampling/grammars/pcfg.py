import math

from .cfg import Grammar, Rule, Terminal, Nonterminal
from ..util import group_by


class Rule(Rule):
    def __init__(self, left, right, probability=1.0):
        super().__init__(left, right)
        if probability < 0:
            raise ValueError("probability cannot be negative")
        self.probability = probability

    @property
    def log_probability(self):
        return math.log(self.probability)

    def _get_kwargs(self):
        return dict(left=self.left, right=self.right, probability=self.probability)

    def __str__(self):
        return "{} [{}]".format(super().__str__(), self.probability)

    def __repr__(self):
        return "Rule({!r}, {!r}, {!r})".format(self.left, self.right, self.probability)


class Grammar(Grammar):
    rule_type = Rule

    def __init__(self, start, rules, normalize=True):
        super().__init__(start, rules)
        if normalize:
            self.normalize_rule_probabilities()

    def normalize_rule_probabilities(self):
        grouped_rules = group_by(self.rules, key=lambda r: r.left)
        for A, rules in grouped_rules.items():
            denom = sum(r.probability for r in rules)
            for rule in rules:
                rule.probability /= denom

    @classmethod
    def from_file(cls, file_path: str, start: Nonterminal, normalize=True):
        """
        Create a Grammar instance from a grammar file.

        Args:
            file_path: Path to the grammar file
            start: Start symbol (must be a Nonterminal)
            normalize: Whether to normalize rule probabilities

        Returns:
            Grammar instance
        """
        with open(file_path, "r") as f:
            content = f.read()
        return cls.from_string(content, start, normalize)

    @classmethod
    def from_string(cls, content: str, start: Nonterminal, normalize=True):
        """
        Create a Grammar instance from a grammar string.

        Args:
            content: Grammar rules as a string
            start: Start symbol (must be a Nonterminal)
            normalize: Whether to normalize rule probabilities

        Returns:
            Grammar instance
        """
        if not isinstance(start, Nonterminal):
            raise TypeError("start symbol must be a Nonterminal")

        rules = []

        for line in content.split("\n"):
            # Skip empty lines and comments
            if line.startswith(("#", " ", "\t", "\n")) or len(line.strip()) < 1:
                continue

            # Remove inline comments
            if line.find("#") != -1:
                line = line[: line.find("#")]

            parts = line.rstrip().split("\t")

            # Parse weight, left-hand side, right-hand side
            if len(parts) >= 3:
                weight = float(parts[0])
                lhs = Nonterminal(parts[1])
                rhs_symbols = []

                # Convert right-hand side symbols to Terminal/Nonterminal
                for symbol in parts[2].split():
                    # Assume uppercase symbols are nonterminals
                    if symbol[0].isupper():
                        rhs_symbols.append(Nonterminal(symbol))
                    else:
                        rhs_symbols.append(Terminal(symbol))

                # Create rule with probability
                rule = cls.rule_type(lhs, rhs_symbols, weight)
                rules.append(rule)

        if not rules:
            raise ValueError("No valid rules found in grammar")

        return cls(start=start, rules=rules, normalize=normalize)
