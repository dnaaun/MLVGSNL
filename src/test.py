# pyright: strict
import argparse
from typing import List, Tuple, Union
from typing_extensions import Literal

from evaluation import test_trees


def extract_spans(tree: str) -> List[Tuple[int, int]]:
    answer: List[Tuple[int, int]] = list()
    stack: List[Union[Literal["("], Tuple[int, int]]] = list()
    items = tree.split()
    curr_index = 0
    for item in items:
        if item == ")":
            pos = -1
            right_margin = stack[pos][1]
            left_margin = None
            while stack[pos] != "(":
                left_margin = stack[pos][0]
                pos -= 1
            assert left_margin is not None
            assert right_margin is not None
            assert isinstance(left_margin, int)
            assert isinstance(right_margin, int)
            # Type annotation neeeded for pyright
            to_add: List[Union[Literal["("], Tuple[int, int]]] = [
                (left_margin, right_margin)
            ]
            stack = stack[:pos] + to_add
            answer.append((left_margin, right_margin))
        elif item == "(":
            stack.append(item)
        else:
            stack.append((curr_index, curr_index))
            curr_index += 1
    return answer


def extract_statistics(
    gold_tree_spans: List[Tuple[int, int]], produced_tree_spans: List[Tuple[int, int]]
):
    set_gold_tree_spans = set(gold_tree_spans)
    set_produced_tree_spans = set(produced_tree_spans)
    precision_cnt = sum(
        list(
            map(
                lambda span: 1.0 if span in set_gold_tree_spans else 0.0,
                set_produced_tree_spans,
            )
        )
    )
    recall_cnt = sum(
        list(
            map(
                lambda span: 1.0 if span in set_produced_tree_spans else 0.0,
                set_gold_tree_spans,
            )
        )
    )
    precision_denom = len(set_produced_tree_spans)
    recall_denom = len(set_gold_tree_spans)
    return precision_cnt, precision_denom, recall_cnt, recall_denom


def f1_score(produced_trees: List[str], gold_trees: List[str]):
    gold_trees_spans = list(map(lambda tree: extract_spans(tree), gold_trees))
    produced_trees_spans = list(map(lambda tree: extract_spans(tree), produced_trees))
    assert len(produced_trees_spans) == len(gold_trees_spans)
    precision_cnt, precision_denom, recall_cnt, recall_denom = 0, 0, 0, 0
    for i, item in enumerate(produced_trees_spans):
        pc, pd, rc, rd = extract_statistics(gold_trees_spans[i], item)
        precision_cnt += pc
        precision_denom += pd
        recall_cnt += rc
        recall_denom += rd
    precision = float(precision_cnt) / precision_denom * 100.0
    recall = float(recall_cnt) / recall_denom * 100.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", type=str, required=True, help="model path to evaluate"
    )
    args = parser.parse_args()

    trees, ground_truth = test_trees(args.candidate)
    f1, _, _ = f1_score(trees, ground_truth)
    print("Model:", args.candidate)
    print("F1 score:", f1)
