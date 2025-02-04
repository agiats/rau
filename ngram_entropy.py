import gzip
import math
import kenlm
import argparse
import os
import subprocess
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

def load_data(data_dir: Path, add_eos: bool = True) -> dict:
    """
    指定されたディレクトリからデータセットを読み込む

    Args:
        data_dir (Path): データディレクトリのパス
        add_eos (bool): 文末に[eos]を追加するかどうか
    """
    datasets = {}
    all_sentences = []

    # 各データセットを読み込む
    for filename in ['train.txt', 'dev.txt', 'test.txt']:
        file_path = data_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                sentences = f.read().splitlines()
                if add_eos:
                    sentences = [f"{s} [eos]".strip() for s in sentences]
                dataset_name = filename.split('.')[0]
                datasets[dataset_name] = sentences
                all_sentences.extend(sentences)

    # allデータセットを追加（train + dev + test）
    datasets['all'] = all_sentences

    return datasets

def calculate_entropy_with_kenlm(model, text, add_eos: bool = True):
    """KenLMモデルを使用してテキストのエントロピーを計算する"""
    need_eos_for_model = not add_eos
    log_prob_sum = 0
    word_count = 0

    for line in text:
        log_prob_sum += model.score(line, eos=need_eos_for_model) * math.log2(10)
        word_count += len(line.split()) + 1

    return -1 * (log_prob_sum / word_count)

def process_single_n(args):
    """
    単一のn-gramモデルを処理する関数
    """
    work_file_path, n, lmplz_path, datasets, memory, add_eos = args
    arpa_path = work_file_path.with_suffix(f'.{n}.arpa')

    try:
        # allデータ（train+dev+test）でモデルを学習
        with open(work_file_path, 'w') as f:
            for line in datasets['all']:
                f.write(line + '\n')

        # モデル作成と学習
        subprocess.run([
            str(lmplz_path),
            '-o', str(n),
            '--skip_symbols',
            '--discount_fallback',
            '--memory', memory,
            '--text', str(work_file_path),
            '--arpa', str(arpa_path)
        ], check=True)

        model = kenlm.Model(str(arpa_path))

        # 各データセットのエントロピーを計算
        results = {}
        for dataset_name, sentences in datasets.items():
            results[dataset_name] = calculate_entropy_with_kenlm(model, sentences, add_eos=add_eos)

        # arpa_path.unlink(missing_ok=True)
        return n, results
    except Exception as e:
        print(f"Error processing {n}-gram: {e}")
        arpa_path.unlink(missing_ok=True)
        return n, None

def main():
    parser = argparse.ArgumentParser(description='Calculate n-gram entropy using KenLM')
    parser.add_argument('input_dir', help='Input directory containing train.txt, dev.txt, test.txt', type=Path)
    parser.add_argument('--n', type=int, nargs='+', default=[2, 3, 4, 5],
                       help='n-gram sizes (default: 2 3 4 5)')
    parser.add_argument('--kenlm-path', type=str, default='../kenlm',
                       help='Path to KenLM directory')
    parser.add_argument('--num-processes', type=int, default=None,
                       help='Number of parallel processes (default: number of CPU cores)')
    parser.add_argument('--memory', type=str, default='8G',
                       help='Memory limit for KenLM (default: 4G)')
    parser.add_argument('--work-dir', type=str, default='work',
                       help='Working directory for intermediate files')
    parser.add_argument('--no-eos', action='store_false', dest='add_eos',
                       help='Do not add [eos] token at the end of sentences')
    args = parser.parse_args()

    # データの読み込み
    print("Loading datasets...")
    datasets = load_data(args.input_dir, add_eos=args.add_eos)

    # 作業ディレクトリの作成
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 作業ファイルのパスを生成
    work_file_path = work_dir / f"{args.input_dir.name}.txt"

    # KenLMのパスを設定
    kenlm_build_dir = Path(args.kenlm_path) / 'build'
    lmplz_path = kenlm_build_dir / 'bin' / 'lmplz'

    if not lmplz_path.exists():
        raise FileNotFoundError(f"lmplz not found at {lmplz_path}")

    # プロセス数の設定
    num_processes = args.num_processes or cpu_count()
    print(f"Using {num_processes} processes")

    # 並列処理の実行
    process_args = [(work_file_path, n, lmplz_path, datasets, args.memory, args.add_eos)
                   for n in args.n]
    with Pool(num_processes) as pool:
        results = pool.map(process_single_n, process_args)

    # 結果をDataFrameに変換
    data = []
    for n, dataset_results in results:
        if dataset_results is not None:
            for dataset_name, entropy in dataset_results.items():
                data.append({
                    'n_gram': n,
                    'dataset': dataset_name,
                    'entropy': entropy
                })

    df = pd.DataFrame(data)

    # CSVファイルとして保存
    output_file = args.input_dir / 'entropy.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # 作業ファイルの削除
    if work_file_path.exists():
        work_file_path.unlink()

if __name__ == "__main__":
    main()
