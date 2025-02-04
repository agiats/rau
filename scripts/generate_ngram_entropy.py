import string
import argparse
import numpy as np
from src.local_entropy.ngram_model import NGramModel

def calculate_sequence_entropy(model, sequence):
    """一つのシーケンスのエントロピーを計算

    シーケンス全体の確率の対数を長さで割ることで、
    1トークンあたりの平均エントロピーを計算します。
    """
    if not sequence:  # 空行の場合は[EOS]のみ
        sequence = ['[EOS]']
    else:
        sequence = sequence + ['[EOS]']  # EOSを追加

    # 先頭にBOSを追加
    sequence = ['[BOS]'] * (model.n - 1) + sequence

    # シーケンス全体の対数確率を計算
    log_sequence_prob = 0.0
    sequence_length = len(sequence) - (model.n - 1)  # BOSを除いた実際の長さ

    try:
        for i in range(model.n - 1, len(sequence)):
            context = tuple(sequence[i - model.n + 1:i])
            token = sequence[i]
            prob = model.get_probability(context, token)
            log_sequence_prob += np.log2(prob)

        # シーケンスの長さで割って、1トークンあたりの平均エントロピーを計算
        return -log_sequence_prob / sequence_length

    except ValueError as e:
        print(f"Warning: {e}")
        return float('inf')  # エラーが発生した場合は無限大のエントロピーを返す

def calculate_samples_entropy(model, samples_file):
    """全サンプルのエントロピーを計算"""
    with open(samples_file, 'r') as f:
        lines = f.readlines()

    entropies = []
    for line in lines:
        line = line.strip()
        if line:  # 空行でない場合
            sequence = line.split()
        else:  # 空行の場合
            sequence = []

        entropy = calculate_sequence_entropy(model, sequence)
        entropies.append(entropy)

    return np.array(entropies)

def main():
    parser = argparse.ArgumentParser(description='Calculate entropy of samples using n-gram model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained n-gram model file')
    parser.add_argument('--input-samples', type=str, required=True,
                       help='Path to input samples file')
    parser.add_argument('--output-entropies', type=str, required=True,
                       help='Path to save calculated entropies')

    args = parser.parse_args()

    # モデルを読み込む
    print(f"Loading model from {args.model_path}")
    model = NGramModel.load(args.model_path)

    # エントロピーを計算
    print("Calculating entropies...")
    entropies = calculate_samples_entropy(model, args.input_samples)

    # 結果を保存
    print(f"Saving results to {args.output_entropies}")
    np.savetxt(args.output_entropies, entropies)

    # 統計情報を表示
    print("\nEntropy Statistics:")
    print(f"Mean: {np.mean(entropies):.4f}")
    print(f"Std: {np.std(entropies):.4f}")
    print(f"Min: {np.min(entropies):.4f}")
    print(f"Max: {np.max(entropies):.4f}")

if __name__ == "__main__":
    main()
