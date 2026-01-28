#!/usr/bin/env python3
"""在无法使用 git 时，从 GitHub 下载 bridge_data_v2 仓库的 main 分支。"""
import os
import zipfile
import urllib.request

def main():
    url = "https://github.com/rail-berkeley/bridge_data_v2/archive/refs/heads/main.zip"
    target_dir = "/workspace"
    zip_path = os.path.join(target_dir, "bridge_data_v2.zip")
    extract_name = "bridge_data_v2-main"
    final_name = "bridge_data_v2"

    print("正在下载 bridge_data_v2 (main) ...")
    urllib.request.urlretrieve(url, zip_path)

    print("正在解压 ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)

    src = os.path.join(target_dir, extract_name)
    dst = os.path.join(target_dir, final_name)
    if os.path.exists(dst):
        import shutil
        shutil.rmtree(dst)
    os.rename(src, dst)
    os.remove(zip_path)
    print(f"已解压到 {dst}")

if __name__ == "__main__":
    main()
