from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple

def load_entropy_results(results_dir: str | Path) -> pd.DataFrame:
    """
    全てのentropy.txtファイルを読み込んでDataFrameにまとめる

    Args:
        results_dir: 結果ディレクトリのパス

    Returns:
        DataFrame with columns:
        - experiment_name: 実験名（ディレクトリ名から抽出）
        - n_gram: n-gramのサイズ
        - entropy: エントロピー値
    """
    results = []
    results_dir = Path(results_dir)

    # 全てのentropy.txtファイルを検索
    for entropy_file in results_dir.glob('**/*_entropy.txt'):
        # 実験名を取得（親ディレクトリの名前）
        experiment_name = entropy_file.parent.name

        # ファイルの内容を解析
        with open(entropy_file) as f:
            lines = f.readlines()

        # エントロピー値を抽出
        for line in lines:
            if 'gram:' in line:
                # 例: "3-gram: 4.5678 bits" から値を抽出
                match = re.match(r'(\d+)-gram: ([\d.]+) bits', line.strip())
                if match:
                    n_gram = int(match.group(1))
                    entropy = float(match.group(2))
                    results.append({
                        'experiment_name': experiment_name,
                        'n_gram': n_gram,
                        'entropy': entropy
                    })

    return pd.DataFrame(results)

def analyze_by_experiment(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    実験ごとの統計量を計算

    Returns:
        Dict with experiment statistics:
        - mean_entropy
        - std_entropy
        - min_entropy
        - max_entropy
    """
    stats = {}
    for name, group in df.groupby('experiment_name'):
        stats[name] = {
            'mean_entropy': group['entropy'].mean(),
            'std_entropy': group['entropy'].std(),
            'min_entropy': group['entropy'].min(),
            'max_entropy': group['entropy'].max()
        }
    return stats

def compare_n_grams(df: pd.DataFrame) -> pd.DataFrame:
    """
    n-gramサイズごとのエントロピーの比較
    """
    return df.pivot_table(
        values='entropy',
        index='experiment_name',
        columns='n_gram',
        aggfunc='mean'
    )

def find_optimal_n(df: pd.DataFrame) -> Dict[str, Tuple[int, float]]:
    """
    各実験で最も高いエントロピーを示すn-gramサイズを特定

    Returns:
        Dict[experiment_name, (optimal_n, max_entropy)]
    """
    optimal = {}
    for name, group in df.groupby('experiment_name'):
        max_idx = group['entropy'].idxmax()
        optimal_n = group.loc[max_idx, 'n_gram']
        max_entropy = group.loc[max_idx, 'entropy']
        optimal[name] = (optimal_n, max_entropy)
    return optimal

def extract_parameters(experiment_name: str) -> Dict[str, str]:
    """
    実験名から実験パラメータを抽出
    例: "1M_samples_eos_min1_max20_LocalShuffle_seed1_window3"
    -> {'seed': '1', 'window': '3'}
    """
    params = {}
    parts = experiment_name.split('_')

    # seedとwindowのパラメータを抽出
    for part in parts:
        if part.startswith('seed'):
            params['seed'] = part[4:]  # 'seed1' -> '1'
        elif part.startswith('window'):
            params['window'] = part[6:]  # 'window3' -> '3'

    return params

def group_by_parameter(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """
    特定のパラメータでグループ化して統計を計算
    """
    # 実験名からパラメータを抽出
    df = df.copy()
    df['params'] = df['experiment_name'].apply(extract_parameters)
    df[param] = df['params'].apply(lambda x: x.get(param, 'unknown'))

    return df.groupby(param)['entropy'].agg(['mean', 'std', 'count'])
