#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_exif_gps.py

用途：讀入 ExifTool -ee -gps* -a -u -g1 的輸出文字檔，
解析所有 GPS Speed、Speed 3D、Latitude、Longitude、Altitude…欄位，
並寫成 CSV。
"""

import sys
import csv
from pathlib import Path

def parse_exif_gps(input_path):
    """
    解析 ExifTool 輸出，回傳 list of dict，每筆 dict 包含一組 GPS 欄位。
    假設欄位順序為：
      GPS Speed → GPS Speed 3D → GPS Latitude → GPS Longitude → GPS Altitude → (再重複)
    遇到同樣的 "GPS Speed" 就視為新的紀錄開始。
    """
    records = []
    current = {}
    with open(input_path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # 遇到 composite section 就結束解析
            if line.startswith('---- Composite'):
                break
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            # 每次遇到 GPS Speed，就視為新紀錄
            if key == 'GPS Speed':
                if current:
                    records.append(current)
                current = {'GPS Speed': try_float(val)}
            elif key == 'GPS Speed 3D':
                current['GPS Speed 3D'] = try_float(val)
            elif key == 'GPS Latitude':
                current['GPS Latitude'] = val
            elif key == 'GPS Longitude':
                current['GPS Longitude'] = val
            elif key == 'GPS Altitude':
                # 去掉尾端單位 m
                current['GPS Altitude'] = val.rstrip(' m')
            else:
                # 如果還要收其他 GPS 欄位，可以在這裡加
                if key.startswith('GPS '):
                    current[key] = val
        # 把最後一筆也加進去
        if current:
            records.append(current)
    return records

def try_float(s):
    """嘗試把字串轉成 float，失敗就原樣回傳"""
    try:
        return float(s)
    except ValueError:
        return s

def write_csv(records, output_path):
    """將 records 寫成 CSV，欄位自動收集並排序"""
    # 收所有可能的欄位
    all_keys = set().union(*(r.keys() for r in records))
    # 排序：把 GPS Speed 放最前面，再其他欄位按字母
    fieldnames = ['GPS Speed'] + sorted(k for k in all_keys if k != 'GPS Speed')
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

def main():
    if len(sys.argv) < 2:
        print(f"用法：{sys.argv[0]} <gps_output.txt> [output.csv]", file=sys.stderr)
        sys.exit(1)

    input_txt = Path(sys.argv[1])
    if not input_txt.exists():
        print(f"找不到輸入檔：{input_txt}", file=sys.stderr)
        sys.exit(1)

    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else input_txt.with_suffix('.csv')
    records = parse_exif_gps(input_txt)
    if not records:
        print("❗ 沒有解析到任何 GPS 紀錄", file=sys.stderr)
        sys.exit(1)

    write_csv(records, out_csv)
    print(f"已寫入 {len(records)} 筆紀錄到 {out_csv}")

if __name__ == '__main__':
    main()
