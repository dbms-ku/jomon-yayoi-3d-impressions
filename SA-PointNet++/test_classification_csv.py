"""Evaluation script for the baseline (non-size-aware) impression classifier.

Adapted from test_classification.py in the PyTorch PointNet/PointNet++
implementation by Xu Yan (yanx27/Pointnet_Pointnet2_pytorch, MIT License;
original author credit: Benny, Nov 2019). This script implements the
paper's evaluation protocol (Section 5.3): the full inference pass is
repeated --repeat times (default 30) and the mean and standard deviation
of Accuracy and Macro-F1 are reported, together with the element-wise
mean confusion matrix (Figure 9). Run-to-run variation stems from the
random initialization of Farthest Point Sampling inside PointNet++ and
from the voting procedure.

Modifications for this project:

  * --dataset selects between the Jomon-Yayoi evaluation set
    (JomonYayoiDatasetPlain; split files under --data_path) and the
    Experimental test split (ImpressionDatasetPlain; requires --CSV_FILE),
  * per-run Macro-F1 is collected so that mean +/- std can be reported
    alongside Accuracy (the original only aggregated Accuracy),
  * --feature_out optionally exports the 1024-dim global features of the
    first run as an .npz with keys 'features'/'labels', in the format
    consumed by visualize_tsne.py (valid for pointnet2 models, whose
    forward returns the global feature as its second output), and
  * execution time and memory statistics are retained (first run excluded
    from timing, as it includes dataset caching).

Run from the root of the PointNet/PointNet++ framework so that `models/`
and `data_utils/` are importable.

Example (Jomon-Yayoi evaluation, paper configuration):
    python test_classification_baseline.py \
        --num_category 5 --num_point 8192 \
        --log_dir 5class_pointnet2_8192 \
        --check_point log/classification/5class_pointnet2_8192/checkpoints/best_model.pth \
        --dataset jomon --data_path data/jomon_yayoi
"""

import argparse
import importlib
import logging
import os
import sys
import time

import numpy as np
import psutil
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from data_utils.impression_dataset_plain import ImpressionDatasetPlain
from data_utils.jomon_yayoi_dataset_plain import JomonYayoiDatasetPlain

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--num_category', default=5, type=int, choices=[5, 10, 40], help='number of classes (5 for impressions)')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline (effectively always on)')
    parser.add_argument('--dataset', type=str, default='jomon', choices=['jomon', 'experimental'],
                        help="evaluation set: 'jomon' (zero-shot Jomon-Yayoi) or 'experimental' (test split)")
    parser.add_argument('--data_path', type=str, default='data', help='dataset root (split files / cache directory)')
    parser.add_argument('--CSV_FILE', type=str, default=None, help='split CSV (required when --dataset experimental)')
    parser.add_argument('--check_point', type=str, required=True, help='checkpoint path (best_model.pth)')
    parser.add_argument('--repeat', type=int, default=30, help='number of independent inference trials')
    parser.add_argument('--feature_out', type=str, default=None,
                        help='optional .npz path to export first-run global features (keys: features, labels)')
    parser.add_argument('--predictions_out', type=str, default=None,
                        help='optional CSV path to export per-sample class probabilities of the last run')
    return parser.parse_args()


def test(model, loader, args, num_class=5, vote_num=1, feature_out=None):
    """Run one full inference pass and return metrics.

    Returns (instance_acc, class_acc, macro_f1, report_str, cm).
    """
    all_features = []
    all_labels = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    all_preds_prob = []
    all_preds_choice = []
    all_targets = []
    all_names = []

    for j, (points, target, names) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class)
        if not args.use_cpu:
            vote_pool = vote_pool.cuda()

        l3_points = None
        for _ in range(vote_num):
            pred, l3_points = classifier(points)
            vote_pool += pred

        if feature_out is not None:
            # Global feature of the last vote; shape (B, 1024, 1) -> (B, 1024).
            # Valid for pointnet2 models whose forward returns the global
            # feature as its second output.
            feat = l3_points.view(points.size(0), 1024).cpu().numpy()
            all_features.append(feat)
            all_labels.append(target.cpu().numpy())

        pred = vote_pool / vote_num
        pred_prob = torch.nn.functional.softmax(pred, dim=1)
        pred_choice = pred.data.max(1)[1]

        all_preds_prob.append(pred_prob.cpu().numpy())
        all_preds_choice.extend(pred_choice.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        all_names.extend(names)

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])

    if feature_out is not None:
        features_array = np.concatenate(all_features, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)
        np.savez(feature_out, features=features_array, labels=labels_array)
        print(f"\nSaved global features to {feature_out}")

    # Instance accuracy computed over all samples (not batch-averaged).
    instance_acc = np.sum(
        np.array(all_preds_choice) == np.array(all_targets)
    ) / len(all_targets)
    macro_f1 = f1_score(all_targets, all_preds_choice, average='macro')

    cm = confusion_matrix(all_targets, all_preds_choice,
                          labels=list(range(num_class)))
    report_str = classification_report(all_targets, all_preds_choice,
                                       digits=4)
    print("\nClassification Report:")
    print(report_str)
    print("\nConfusion Matrix:")
    print(np.array2string(cm, separator=', '))

    if args.predictions_out is not None:
        import pandas as pd
        df = pd.DataFrame(
            np.concatenate(all_preds_prob, axis=0),
            columns=[f"class_{i}" for i in range(num_class)],
        )
        df.insert(0, "filename", all_names)
        df.to_csv(args.predictions_out, index=False)
        print(f"Saved predictions to {args.predictions_out}")

    return instance_acc, class_acc, macro_f1, report_str, cm


def run_once(args, logger, feature_out=None):
    """Load data and model, run one evaluation, return (acc, macro_f1, cm)."""
    def log_string(s):
        logger.info(s)
        print(s)

    log_string('Load dataset ...')
    data_path = args.data_path
    log_string(f'data path is {data_path}')

    if args.dataset == 'experimental':
        if args.CSV_FILE is None:
            raise ValueError("--CSV_FILE is required when --dataset experimental")
        test_dataset = ImpressionDatasetPlain(
            root=data_path, args=args, split='test',
            process_data=args.process_data)
    else:
        test_dataset = JomonYayoiDatasetPlain(
            root=data_path, args=args, split='test',
            process_data=args.process_data)
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=10)

    num_class = args.num_category
    experiment_dir = 'log/classification/' + args.log_dir
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    map_location = 'cpu' if args.use_cpu else None
    checkpoint = torch.load(args.check_point, map_location=map_location)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc, macro_f1, report_str, cm = test(
            classifier.eval(), testDataLoader, args,
            num_class=num_class, vote_num=args.num_votes,
            feature_out=feature_out)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f, '
                   'Macro-F1: %f' % (instance_acc, class_acc, macro_f1))
        log_string('\n--- Detailed Classification Report ---')
        log_string('\n' + report_str)
        log_string('\n--- Confusion Matrix ---')
        log_string('\n' + np.array2string(cm, separator=', '))

    return instance_acc, macro_f1, cm


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = 'log/classification/' + args.log_dir
    log_file_path = f'{experiment_dir}/eval.txt'
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.info('PARAMETER ...')
    logger.info(args)
    print(args)

    repeat = args.repeat
    times, cpu_memories, gpu_memories = [], [], []
    accuracies, macro_f1s, all_cms = [], [], []

    for i in range(repeat):
        print(f"Run {i + 1}/{repeat}...")
        process = psutil.Process()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        cpu_before = process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        # Features (if requested) are exported on the first run only.
        feature_out = args.feature_out if i == 0 else None
        instance_acc, macro_f1, cm = run_once(args, logger,
                                              feature_out=feature_out)

        end_time = time.time()
        cpu_after = process.memory_info().rss / 1024 / 1024

        times.append(end_time - start_time)
        cpu_memories.append(max(cpu_after - cpu_before, 0))
        gpu_memories.append(
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available() else 0)
        accuracies.append(instance_acc)
        macro_f1s.append(macro_f1)
        all_cms.append(cm)

    mean_cm = np.mean(all_cms, axis=0)

    # Accuracy/F1 statistics use all runs; time/memory statistics exclude
    # the first run, which includes one-time dataset caching.
    t_target = times[1:] if len(times) > 1 else times
    c_target = cpu_memories[1:] if len(cpu_memories) > 1 else cpu_memories
    g_target = gpu_memories[1:] if len(gpu_memories) > 1 else gpu_memories

    def mean_std(values):
        # Population standard deviation (ddof=0), matching the aggregation
        # used to produce the paper's reported +/- values.
        if len(values) > 1:
            return float(np.mean(values)), float(np.std(values))
        return values[0], 0.0

    acc_m, acc_s = mean_std(accuracies)
    f1_m, f1_s = mean_std(macro_f1s)
    t_m, t_s = mean_std(t_target)
    c_m, c_s = mean_std(c_target)
    g_m, g_s = mean_std(g_target)

    stats_text = [
        "\n" + "=" * 50,
        f"Final statistics (n={repeat})",
        f"Instance Accuracy (all runs): {acc_m:.4f} +/- {acc_s:.4f}",
        f"Macro-F1 (all runs):          {f1_m:.4f} +/- {f1_s:.4f}",
        f"Wall time (excl. first run):  {t_m:.4f} +/- {t_s:.4f} s",
        f"CPU memory delta:             {c_m:.2f} +/- {c_s:.2f} MB",
        f"GPU memory peak:              {g_m:.2f} +/- {g_s:.2f} MB",
        "\nMean confusion matrix over runs:",
        np.array2string(mean_cm, separator=', '),
        "=" * 50 + "\n",
    ]

    with open(log_file_path, "a", encoding="utf-8") as f:
        for line in stats_text:
            f.write(line + "\n")
            print(line)

    print(f"Statistics appended to {log_file_path}")


if __name__ == '__main__':
    main()