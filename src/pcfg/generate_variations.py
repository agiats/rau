import random
import argparse
from itertools import product
from pathlib import Path


def flip_rhs(rhs):
    """Reverse the order of elements in the RHS."""
    parts = rhs.split()
    return " ".join(parts[::-1])


def process_rule(line, switch_value):
    """Process a rule and generate variations."""
    parts = line.strip().split("\t")

    # Return the original rule if no switch is defined
    if len(parts) < 4:
        return [line.strip()]

    prob, lhs, rhs, switch_idx = parts

    if switch_value == 0:
        return [f"{prob}\t{lhs}\t{rhs}\t{switch_idx}"]
    elif switch_value == 1:
        return [f"{prob}\t{lhs}\t{flip_rhs(rhs)}\t{switch_idx}"]
    else:  # switch_value == 2
        return [
            f"{prob}\t{lhs}\t{rhs}\t{switch_idx}",
            f"{prob}\t{lhs}\t{flip_rhs(rhs)}\t{switch_idx}",
        ]


def generate_grammar_variation(
    input_file,
    output_dir,
    num_grammar=-1,
    num_switches=7,
    include_all_deterministic_grammar=False,
):
    """Generate grammar variations.

    Args:
        input_file: Path to input grammar file
        output_dir: Directory to output generated grammars
        num_grammar: Number of grammars to generate. If -1, generate all possible combinations
        num_switches: Number of switches in the grammar
        include_all_deterministic_grammar: Include all deterministic grammars
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)

    with input_path.open("r") as f:
        lines = f.readlines()

    # Generate all possible combinations
    all_combinations = list(product(range(3), repeat=num_switches))

    # If num_grammar is specified and valid, randomly sample from all combinations
    if num_grammar > 0 and num_grammar < len(all_combinations):
        if include_all_deterministic_grammar:
            selected_combinations = [c for c in all_combinations if 2 not in c]
            num_deterministic = len(selected_combinations)
            selected_combinations.extend(
                random.sample(
                    [c for c in all_combinations if 2 in c],
                    num_grammar - num_deterministic,
                )
            )
            print(f"Generated {num_deterministic} deterministic grammars and {num_grammar - num_deterministic} non-deterministic grammars")
        else:
            selected_combinations = random.sample(all_combinations, num_grammar)
            print(f"Generated {num_grammar} non-deterministic grammars")
    else:
        print("num_grammar > len(all_combinations), generating all combinations")
        selected_combinations = all_combinations

    for values in selected_combinations:
        print("Generating grammar with switch values:", values)
        output_rules = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 4:  # Rule with a defined switch
                switch_idx = int(parts[3]) - 1  # Convert to 0-based index
                rules = process_rule(line, values[switch_idx])
                output_rules.extend(rules)
            else:  # Rule without a defined switch
                output_rules.append(line)

        # Generate filename based on switch values
        grammar_name = "".join(str(v) for v in values)
        grammar_path = output_path / grammar_name / "grammar.gr"
        grammar_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the output rules to the file
        grammar_path.write_text("\n".join(output_rules) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate grammar variations based on switch values"
    )
    parser.add_argument("--input_file", type=Path, help="Path to input grammar file")
    parser.add_argument("--output_dir", type=Path, help="Directory to output generated grammars")
    parser.add_argument(
        "-n",
        "--num_grammar",
        type=int,
        default=-1,
        help="Number of grammars to generate. If -1, generate all possible combinations",
    )
    parser.add_argument(
        "--num_switches", type=int, default=7, help="Number of switches in the grammar"
    )
    parser.add_argument(
        "--include_all_deterministic_grammar",
        action="store_true",
        help="Include all deterministic grammars",
    )

    args = parser.parse_args()

    generate_grammar_variation(
        args.input_file,
        args.output_dir,
        args.num_grammar,
        args.num_switches,
        args.include_all_deterministic_grammar,
    )


if __name__ == "__main__":
    main()
