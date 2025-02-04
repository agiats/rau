import os
import re


def remove_last_token(input_file, output_file):
    # ファイルパスからディレクトリ名を取得
    dir_name = os.path.basename(os.path.dirname(input_file))

    # Q*_S数字_s数字 のパターンから数字部分を抽出
    match = re.match(r"Q\d+_S(\d+)_s\d+", dir_name)
    number_to_remove = match.group(1) if match else None
    print(f"Removing last token: {number_to_remove}")

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            # スペースで分割
            tokens = line.strip().split()
            if tokens:  # 空行でない場合
                # 最後のトークンが抽出した数字と一致する場合は削除
                if number_to_remove and tokens[-1] == number_to_remove:
                    new_line = " ".join(tokens[:-1])
                else:
                    new_line = " ".join(tokens)
                f_out.write(new_line + "\n")
            else:  # 空行の場合はそのまま書き込む
                f_out.write("\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python remove_last_token.py input_file output_file")
        sys.exit(1)

    remove_last_token(sys.argv[1], sys.argv[2])
