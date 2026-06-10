import sys
import argparse


def build_base_args(mode: str) -> list[str]:
    """Return the argument list common to all AMST modes for TCGA."""
    return [
        '--dataset',         'TCGA',
        '--batch_size',      '32',
        '--epochs',          '100',
        '--learning_rate',   '0.001',
        '--lr_decay_step',   '70',
        '--lr_decay_ratio',  '0.1',
        '--random_seed',     '0',
        '--save_path',       '../ckpt',
        '--parallel_method', 'single',
        '--no_using_ploader',  
        '--no_test',
    ]


def run_alt(extra_args: list[str]) -> None:
    from amst_alt import AMST_A_Trainer
    args = build_base_args('alt') + [
        '--prefix',         'AMST-ALT-TCGA',
        '--fusion_method',  'msum',
        '--a_skip_factor',  '3',
        '--v_skip_factor',  '1',
        '--t_skip_factor',  '1',
    ] + extra_args
    trainer = AMST_A_Trainer(args_str=args)
    trainer.train_validate()


def run_joint(extra_args: list[str]) -> None:
    from amst_joint import AMST_J_Trainer
    args = build_base_args('joint') + [
        '--prefix',         'AMST-JOINT-TCGA',
        '--fusion_method',  'concat',
        '--a_skip_factor',  '3',
        '--v_skip_factor',  '1',
        '--t_skip_factor',  '1',
    ] + extra_args
    trainer = AMST_J_Trainer(args_str=args)
    trainer.train_validate()


def run_full(extra_args: list[str]) -> None:
    from amst_full import AMST_F_Trainer
    args = build_base_args('full') + [
        '--prefix',         'AMST-FULL-TCGA',
        '--a_skip_factor',  '3',
        '--v_skip_factor',  '1',
        '--t_skip_factor',  '1',
    ] + extra_args
    trainer = AMST_F_Trainer(args_str=args)
    trainer.train_validate()


# ------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train AMST on the TCGA dummy dataset')
    parser.add_argument('--mode', default='full',
                        choices=['alt', 'joint', 'full'],
                        help='Which AMST variant to train')
    known, extra = parser.parse_known_args()

    dispatch = {'alt': run_alt, 'joint': run_joint, 'full': run_full}
    dispatch[known.mode](extra)
