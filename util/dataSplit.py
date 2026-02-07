#!/usr/bin/env python3
"""
从TSV文件中提取前若干行并保存到新文件
"""

import sys

# ============================
# 用户配置部分 - 修改这些参数
# ============================

# 源TSV文件路径
SOURCE_FILE = "/tmp/pycharm_project_647/data/valid.tsv"

# 新TSV文件保存路径
OUTPUT_FILE = "/tmp/pycharm_project_647/data/mini/valid.tsv"

# 要提取的行数（包括表头）
EXTRACT_ROWS = 18000  # 例如：提取前1000行（包括表头）


# ============================
# 主程序
# ============================

def extract_rows_from_tsv(source_file, output_file, num_rows):
    """
    从TSV文件中提取前num_rows行并保存到新文件

    参数:
        source_file: 源TSV文件路径
        output_file: 输出TSV文件路径
        num_rows: 要提取的行数
    """
    try:
        # 打开源文件进行读取
        with open(source_file, 'r', encoding='utf-8') as f_in:
            # 读取指定行数
            lines = []
            for i, line in enumerate(f_in):
                if i >= num_rows:
                    break
                lines.append(line)

        if not lines:
            print(f"警告: 源文件 '{source_file}' 为空或行数不足")
            return False

        # 写入新文件
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.writelines(lines)

        # 输出结果信息
        actual_rows = len(lines)
        print(f"成功提取并保存文件!")
        print(f"源文件: {source_file}")
        print(f"输出文件: {output_file}")
        print(f"提取行数: {actual_rows} 行")
        print(f"目标行数: {num_rows} 行")

        if actual_rows < num_rows:
            print(f"注意: 源文件只有 {actual_rows} 行，少于要求的 {num_rows} 行")

        return True

    except FileNotFoundError:
        print(f"错误: 找不到源文件 '{source_file}'")
        return False
    except PermissionError:
        print(f"错误: 没有权限读取/写入文件")
        return False
    except Exception as e:
        print(f"错误: 处理文件时发生异常 - {str(e)}")
        return False


def main():
    """主函数"""
    print(f"开始提取TSV文件...")
    print(f"配置:")
    print(f"  源文件: {SOURCE_FILE}")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"  提取行数: {EXTRACT_ROWS}")
    print("-" * 40)

    success = extract_rows_from_tsv(SOURCE_FILE, OUTPUT_FILE, EXTRACT_ROWS)

    if success:
        print("-" * 40)
        print("操作完成!")
    else:
        print("-" * 40)
        print("操作失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()