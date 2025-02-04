import tempfile
import os
import shutil
from remove_last_token import remove_last_token


def test_remove_last_token():
    # テスト用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        # Q8_S16_s12312 ディレクトリを作成
        test_dir = os.path.join(temp_dir, "Q8_S16_s12312")
        os.makedirs(test_dir)

        # テストケース
        test_input = """token1 token2 16
token3 token4 token5 16
normal1 normal2 normal3
token6 16

token7 15
token8 16"""

        expected_output = """token1 token2
token3 token4 token5
normal1 normal2 normal3
token6

token7 15
token8"""

        # 入力ファイルと出力ファイルのパスを設定
        input_file = os.path.join(test_dir, "dev.txt")
        output_file = os.path.join(test_dir, "dev.txt.tmp")

        # テストファイルを作成
        with open(input_file, "w") as f:
            f.write(test_input)

        # 関数を実行
        remove_last_token(input_file, output_file)

        # 結果を確認
        with open(output_file, "r") as f:
            result = f.read().strip()

        # テスト結果を表示
        print("Test passed:" if result == expected_output else "Test failed:")
        if result != expected_output:
            print("\nExpected:")
            print(expected_output)
            print("\nGot:")
            print(result)


if __name__ == "__main__":
    test_remove_last_token()
