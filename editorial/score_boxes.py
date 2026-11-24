"""Re-score SO-101 VLM box predictions on CPU, using only Python's standard library.

From the release root:
  python editorial/score_boxes.py site/evidence/real-box-pairs.csv

The optional --pairs flag exports de-identified numeric inputs when reading the
original stage2/best/test_evaluation.json. Coordinates are normalized XYXY on
[0, 1000]. kind=0 denotes object_bbox and kind=1 denotes target_bbox. The
reference boxes in this evaluation are reviewed pseudo-labels.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import fmean


TYPES = ('object_bbox', 'target_bbox')
AXES = ('x1', 'y1', 'x2', 'y2')
PAIR_HEADER = ['kind', *('pred_' + a for a in AXES), *('ref_' + a for a in AXES)]
THRESHOLDS = (0.25, 0.50, 0.75)


def as_box(coords):
    if len(coords) != 4:
        raise ValueError('Expected four box coordinates')
    x1, y1, x2, y2 = map(int, coords)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError('Invalid normalized XYXY box')
    return x1, y1, x2, y2


def iou(pred, ref):
    px, py, qx, qy = pred
    rx, ry, sx, sy = ref
    overlap = max(0, min(qx, sx) - max(px, rx)) * max(0, min(qy, sy) - max(py, ry))
    union = (qx - px) * (qy - py) + (sx - rx) * (sy - ry) - overlap
    return overlap / union


def read_pairs(path):
    with path.open(encoding='utf-8', newline='') as stream:
        if path.suffix.lower() == '.json':
            for record in json.load(stream)['predictions']:
                if record['family'] != 'bbox':
                    continue
                pred = as_box(json.loads(record['prediction']))
                ref = as_box(record['answer'])
                if not math.isclose(iou(pred, ref), record['score']['iou'], abs_tol=1e-12):
                    raise ValueError('Recomputed IoU disagrees with the stored prediction score')
                yield TYPES.index(record['type']), pred, ref
        else:
            reader = csv.DictReader(stream)
            if reader.fieldnames != PAIR_HEADER:
                raise ValueError('Unexpected pairs.csv header')
            for record in reader:
                kind = int(record['kind'])
                if kind not in (0, 1):
                    raise ValueError('Unexpected box question type')
                yield kind, as_box([record['pred_' + a] for a in AXES]), as_box([record['ref_' + a] for a in AXES])


def write_summary(stream, pairs):
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['question_type', 'iou_threshold', 'passed', 'total', 'pass_rate', 'mean_iou'])
    for kind, name in [(None, 'all_bbox'), *enumerate(TYPES)]:
        scores = [iou(pred, ref) for label, pred, ref in pairs if kind is None or label == kind]
        if not scores:
            raise ValueError(f'No predictions for {name}')
        for threshold in THRESHOLDS:
            passed = sum(score >= threshold for score in scores)
            writer.writerow([name, f'{threshold:.2f}', passed, len(scores),
                             f'{passed / len(scores):.9f}', f'{fmean(scores):.9f}'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='public pairs CSV or the original evaluation JSON')
    parser.add_argument('--pairs', type=Path, help='save only the question type code and eight coordinates')
    parser.add_argument('--output', type=Path, help='write the grouped CSV here (default: stdout)')
    args = parser.parse_args()
    pairs = list(read_pairs(args.input))
    if args.pairs:
        with args.pairs.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(PAIR_HEADER)
            writer.writerows((label, *pred, *ref) for label, pred, ref in pairs)
    if args.output:
        with args.output.open('w', encoding='utf-8', newline='') as stream:
            write_summary(stream, pairs)
    else:
        write_summary(sys.stdout, pairs)


if __name__ == '__main__':
    main()
