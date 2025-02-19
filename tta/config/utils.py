import os
import argparse

from ..config import cifar10c

# CLI 인자 파싱 함수
def parse_args():
    parser = argparse.ArgumentParser(description="Run TTA with custom config")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")

    # 추가적인 설정값을 키-값 쌍으로 입력받음 (MODEL.ADAPTATION TENT TEST.BATCH_SIZE 64 등)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command line")

    return parser.parse_args()


def load_config(cfg_path, opts):
    # 기본 설정을 클론하여 사용
    cfg_path = os.path.abspath(cfg_path)
    benchmark = cfg_path.split('/')[-2]
    if benchmark == 'cifar10c':
        config = cifar10c
    else:
        raise NotImplementedError(f"Your selected benchmark '{benchmark}' is not supported.")

    # YAML 파일을 적용
    config.merge_from_file(cfg_path)

    # CLI에서 추가 설정 덮어쓰기
    config.merge_from_list(opts)

    # 설정을 동결하여 변경되지 않도록 함
    config.freeze()

    return config
