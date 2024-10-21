import os
import shutil
import hashlib
from datetime import datetime, timedelta

def organize_samples(processed_data_dir, positive_dir, negative_dir):
    # 确保目标文件夹存在，并清空已有文件
    for dir_path in [positive_dir, negative_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # 处理纯正样本
    pure_pos_dir = os.path.join(processed_data_dir, "pure_pos_data")
    process_samples(pure_pos_dir, positive_dir, "positive")

    # 处理纯负样本
    pure_neg_dir = os.path.join(processed_data_dir, "pure_neg_data")
    process_samples(pure_neg_dir, negative_dir, "negative")

    # 去重
    remove_duplicates(positive_dir)
    remove_duplicates(negative_dir)

    print("样本组织和去重完成。")
    print(f"正样本数量：{len(os.listdir(positive_dir))}")
    print(f"负样本数量：{len(os.listdir(negative_dir))}")

def process_samples(source_dir, target_dir, sample_type):
    for time_period in os.listdir(source_dir):
        time_period_path = os.path.join(source_dir, time_period)
        if os.path.isdir(time_period_path):
            for file in os.listdir(time_period_path):
                file_path = os.path.join(time_period_path, file)
                new_file_name = f"{sample_type}_{time_period}_{file}"
                shutil.copy(file_path, os.path.join(target_dir, new_file_name))

def remove_duplicates(directory):
    seen_hashes = set()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash in seen_hashes:
            os.remove(file_path)
        else:
            seen_hashes.add(file_hash)

def split_samples_into_time_periods(pos_data_path, neg_data_path, days_step):
    # 1. 读取所有样本并获取时间戳
    all_timestamps = []
    all_files = []

    for folder in [pos_data_path, neg_data_path]:
        for file in os.listdir(folder):
            # .DS_Store跳过
            if '.DS_Store' in file:
                continue
            timestamp = int(file.split('_')[-1].split('.')[0])
            all_timestamps.append(timestamp)
            all_files.append((folder, file, timestamp))

    # 2. 计算时间范围
    min_timestamp = min(all_timestamps)
    max_timestamp = max(all_timestamps)
    start_date = datetime.fromtimestamp(min_timestamp).date()
    end_date = datetime.fromtimestamp(max_timestamp).date()

    # 3. 创建时间段并分配样本
    current_date = start_date
    period_count = 0
    while current_date <= end_date:
        period_start_date = current_date
        period_end_date = min(current_date + timedelta(days=days_step), end_date + timedelta(days=1))

        period_start = int(datetime.combine(period_start_date, datetime.min.time()).timestamp())
        period_end = int(datetime.combine(period_end_date, datetime.min.time()).timestamp())

        # 创建时间段文件夹
        period_folder = f"{period_start_date.strftime('%m%d')}_{period_end_date.strftime('%m%d')}"
        os.makedirs(os.path.join(pos_data_path, period_folder), exist_ok=True)
        os.makedirs(os.path.join(neg_data_path, period_folder), exist_ok=True)

        # 分配样本到对应的时间段文件夹
        for src_folder, file, timestamp in all_files:
            if period_start <= timestamp < period_end:
                src_path = os.path.join(src_folder, file)
                dst_folder = os.path.join(src_folder, period_folder)
                shutil.copy(src_path, dst_folder)

        current_date = period_end_date
        period_count += 1

    print(f"Samples have been split into {period_count} time periods with {days_step} days step from {start_date} to {end_date}.")

if __name__ == "__main__":
    processed_data_dir = "./data/raw/"
    positive_dir = "./data/raw/processed/pos/"
    negative_dir = "./data/raw/processed/neg/"

    # 清空所有子文件夹
    for dir_path in [positive_dir, negative_dir]:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))

    organize_samples(processed_data_dir, positive_dir, negative_dir)
    # split_samples_into_time_periods(positive_dir, negative_dir, days_step=15)