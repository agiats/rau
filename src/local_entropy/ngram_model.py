import numpy as np
from itertools import product
import pickle

class NGramModel:
    def __init__(self, alphabet, n, alpha=1.0, bos='[BOS]', eos='[EOS]'):
        """
        Initialize n-gram model
        """
        self.alphabet = alphabet
        self.n = n
        self.alpha = alpha
        self.bos = bos
        self.eos = eos
        self.model = self._build_random_model()

    def _is_valid_context(self, ctx):
        """
        Internal method to validate context.
        Rules:
        - EOS cannot be in context
        - BOS must appear continuously from the left edge, followed by normal characters only
        """
        if self.eos in ctx:
            return False
        saw_normal = False
        for token in ctx:
            if token == self.bos:
                if saw_normal:
                    return False
            else:
                saw_normal = True
        return True

    def _build_random_model(self):
        """
        Internal method to build random n-gram model using Dirichlet distribution.
        Only uses valid contexts and constrains BOS to appear only at the beginning.
        """
        if self.n < 1:
            raise ValueError("n must be >= 1")

        context_alphabet = [a for a in self.alphabet if a != self.eos]

        if self.n == 1:
            probs = np.random.dirichlet([self.alpha] * len(self.alphabet))
            return {(): {a: p for a, p in zip(self.alphabet, probs)}}

        all_contexts = product(context_alphabet, repeat=self.n-1)
        valid_contexts = [ctx for ctx in all_contexts if self._is_valid_context(ctx)]

        model = {}
        for ctx in valid_contexts:
            # Exclude BOS from output symbols as it should only appear at the beginning
            output_alphabet = [a for a in self.alphabet if a != self.bos]
            probs = np.random.dirichlet([self.alpha] * len(output_alphabet))
            model[ctx] = {a: p for a, p in zip(output_alphabet, probs)}

        return model


    def sample(self, max_length=-1):
        """
        Generate a sequence from the model.
        Starts with (n-1) BOS tokens and ends when either:
        - EOS is generated
        - max_length is reached (if max_length > 0)
        - no valid context is found

        Args:
            max_length: Maximum length of the generated sequence
                    If -1, generates until EOS or no valid context is found

        Returns:
            list: Generated sequence (excluding EOS)
        """
        start_context = tuple([self.bos] * (self.n-1))
        context = list(start_context)
        sequence = []

        while True:
            # Break if max_length is reached (when max_length > 0)
            if max_length > 0 and len(sequence) >= max_length:
                break

            dist = self.model.get(tuple(context))
            if dist is None:
                break
            symbols = list(dist.keys())
            probs = list(dist.values())
            next_symbol = np.random.choice(symbols, p=probs)
            if next_symbol == self.eos:
                break
            sequence.append(next_symbol)
            context = (context + [next_symbol])[-(self.n-1):]

        return sequence


    def save(self, filepath):
        """
        Save model to file using pickle

        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'alphabet': self.alphabet,
                'n': self.n,
                'alpha': self.alpha,
                'bos': self.bos,
                'eos': self.eos,
                'model': self.model
            }, f)

    @classmethod
    def load(cls, filepath):
        """
        Load model from file

        Args:
            filepath: Path to the saved model

        Returns:
            NGramModel: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            alphabet=data['alphabet'],
            n=data['n'],
            alpha=data['alpha'],
            bos=data['bos'],
            eos=data['eos']
        )
        model.model = data['model']  # Override the randomly initialized model
        return model

    def get_probability(self, context, token):
        """
        Get probability of token given context

        Args:
            context: Tuple of tokens representing the context
            token: Token to get probability for

        Returns:
            float: Probability of token given context

        Raises:
            ValueError: If context or token is not found in the model
        """
        dist = self.model.get(context)
        if dist is None:
            raise ValueError(f"Context {context} not found in the model")

        prob = dist.get(token)
        if prob is None:
            raise ValueError(f"Token {token} not found for context {context}")

        return prob

