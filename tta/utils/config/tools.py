import os
import argparse
import logging

from datetime import datetime
from tabulate import tabulate
from tta.utils.config import cifar10c


# CLI 인자 파싱 함수
def parse_args():
    parser = argparse.ArgumentParser(description="Run TTA with custom config")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config YAML file"
    )

    # 추가적인 설정값을 키-값 쌍으로 입력받음 (MODEL.ADAPTATION TENT TEST.BATCH_SIZE 64 등)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command line",
    )

    return parser.parse_args()


def load_config(cfg_path, opts):
    # 기본 설정을 클론하여 사용
    cfg_path = os.path.abspath(cfg_path)
    benchmark = cfg_path.split("/")[-2]
    if benchmark == "cifar10c":
        config = cifar10c
    else:
        raise NotImplementedError(
            f"Your selected benchmark '{benchmark}' is not supported."
        )

    # YAML 파일을 적용
    config.merge_from_file(cfg_path)

    # CLI에서 추가 설정 덮어쓰기
    config.merge_from_list(opts)

    # 설정을 동결하여 변경되지 않도록 함
    config.freeze()

    return config


def setup_logger(config_name):
    # Get the current date and time in a nice format
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the log directory path
    log_dir = os.path.join("logs", config_name, date_time)
    os.makedirs(log_dir, exist_ok=True)

    # Set up the file handler
    log_file = os.path.join(log_dir, "log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # or another level you prefer
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Get the main logger and add the file handler
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)  # or another level you prefer

    return logger


def format_experiment_results(results):
    # 노이즈 유형과 심각도 레벨 추출
    noise_types = set(key.rsplit("_", 1)[0] for key in results.keys())
    severities = sorted(set(int(key.rsplit("_", 1)[1]) for key in results.keys()))

    # 테이블 데이터 생성
    table_data = [[""] + [f"Severity {i}" for i in severities]]
    for noise_type in sorted(noise_types):
        row = [noise_type]
        for severity in severities:
            key = f"{noise_type}_{severity}"
            row.append(results.get(key, ""))
        table_data.append(row)

    # tabulate로 표 생성
    return tabulate(table_data, headers="firstrow", tablefmt="grid")
