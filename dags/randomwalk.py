'''Random‑walk bandit recommender for sequential Kindle‑store interactions.

The module constructs a directed temporal graph from a Pandas dataframe where
nodes are products (``parent_asin``) and edges encode successive actions of the
same user. Edge attributes capture time difference, price history, ratings and
content similarity. A contextual multi‑armed bandit guides biased random walks
over the graph. Five traversal policies are evaluated via 5‑fold cross‑
validation, and the best model can be persisted. The public API exposes
``fit`` for offline training, ``recommend`` for online inference on a new user
trace, and ``visualise`` for exploratory graph inspection.
'''

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
#import joblib

K_FOLDS = 5
MAX_WALK_LEN = 50
SEED = 42
BUDGET = 1e6
TOP_K_RECS = 20
EXPLORATION = 0.1
NOISE = 0.05

Traversal = List[str]

def safe_cast(value, to_type, default=0.0):
    try:
        if pd.isna(value) or str(value).lower() in {'none', 'nan', 'nat', ''}:
            return default
        return to_type(value)
    except (ValueError, TypeError):
        return default

def _edge_payload(prev: pd.Series, cur: pd.Series) -> Tuple[float, float, float, int, float, float, float, float, float, int]:
    return (
        safe_cast(cur['timestamp'], float) - safe_cast(prev['timestamp'], float),
        safe_cast(prev['price'], float),
        safe_cast(cur['price'], float),
        safe_cast(cur['average_rating'], float),
        safe_cast(cur['rating_number'], float),
        int(str(prev['categories']) == str(cur['categories']))
    )

class GraphRecommender:
    '''Random‑walk bandit recommender backed by a directed temporal graph.'''

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.rng = random.Random(SEED)

    @staticmethod
    def _build_graph(df: pd.DataFrame) -> nx.DiGraph:
        g = nx.DiGraph()
        df_sorted = df.sort_values(['user_id', 'timestamp'])
        for _, group in df_sorted.groupby('user_id'):
            prev_row = None
            for _, row in group.iterrows():
                if prev_row is not None:
                    src = prev_row['parent_asin']
                    dst = row['parent_asin']
                    if src == dst:
                        prev_row = row
                        continue
                    payload = _edge_payload(prev_row, row)
                    if g.has_edge(src, dst):
                        g[src][dst]['payloads'].append(payload)
                    else:
                        g.add_edge(src, dst, payloads=[payload])
                prev_row = row
        for u, v, d in g.edges(data=True):
            arr = np.array(d['payloads'])
            d['weight'] = 1 / (arr[:, 0].mean() + 1e-9)
            d['context'] = arr.mean(axis=0)
            del d['payloads']
        return g

    def fit(self, df: pd.DataFrame) -> None:
        self.graph = self._build_graph(df)
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, val_idx in kf.split(df):
            train_g = self._build_graph(df.iloc[train_idx])
            val_df = df.iloc[val_idx]
            acc = self._evaluate_fold(train_g, val_df)
            scores.append(acc)
        self.cv_score_ = float(np.mean(scores))

    def _bandit_step(self, node: str, budget: float) -> str | None:
        if node not in self.graph:
            return None
        nbrs = list(self.graph.successors(node))
        if not nbrs:
            return None
        contexts = np.array([self.graph[node][n]['context'] for n in nbrs])
        weights = np.array([self.graph[node][n]['weight'] for n in nbrs])
        probs = 1 / (weights + 1e-9)
        if self.rng.random() < EXPLORATION:
            probs = np.ones_like(probs)
        noise = self.rng.uniform(0, NOISE)
        probs = probs + noise
        probs = probs / probs.sum()
        return self.rng.choices(nbrs, weights=probs)[0]

    def _walk(self, start: str, length: int) -> Traversal:
        path = [start]
        budget = BUDGET
        for _ in range(length):
            nxt = self._bandit_step(path[-1], budget)
            if nxt is None:
                break
            path.append(nxt)
        return path

    def _traverse(self, start: str, policy: str) -> Traversal:
        if policy == 'one':
            return self._walk(start, 1)[1:]
        if policy == 'breadth20':
            frontier = [start]
            seen = set(frontier)
            out = []
            while frontier and len(out) < 20:
                nxt_frontier = []
                for node in frontier:
                    children = self._walk(node, 1)[1:]
                    for c in children:
                        if c not in seen and len(out) < 20:
                            out.append(c)
                            seen.add(c)
                            nxt_frontier.append(c)
                frontier = nxt_frontier
            return out
        if policy == 'breadth10_depth2':
            frontier = [start]
            seen = {start}
            out = []
            for _ in range(2):
                nxt_frontier = []
                for node in frontier:
                    children = self._walk(node, 1)[1:]
                    for c in children:
                        if c not in seen:
                            seen.add(c)
                            nxt_frontier.append(c)
                            if len(out) < 10:
                                out.append(c)
                frontier = nxt_frontier
            return out[:10]
        if policy == 'breadth5_depth4':
            frontier = [start]
            seen = {start}
            out = []
            for _ in range(4):
                nxt_frontier = []
                for node in frontier:
                    children = self._walk(node, 1)[1:]
                    for c in children:
                        if c not in seen:
                            seen.add(c)
                            nxt_frontier.append(c)
                            if len(out) < 5:
                                out.append(c)
                frontier = nxt_frontier
            return out[:5]
        if policy == 'depth20':
            return self._walk(start, 20)[1:]
        raise ValueError('unknown policy')

    def recommend(self, user_trace: pd.DataFrame, policy: str = 'breadth20', store: bool = False) -> List[str]:
        if user_trace.empty:
            return []
        user_trace_sorted = user_trace.sort_values('timestamp')
        start = user_trace_sorted.iloc[-1]['parent_asin']
        recs = self._traverse(start, policy)[:TOP_K_RECS]
        if store:
            self.graph = nx.compose(self.graph, self._build_graph(user_trace))
        return recs

    def _evaluate_fold(self, g: nx.DiGraph, val_df: pd.DataFrame) -> float:
        hits = 0
        total = 0
        for _, group in val_df.groupby('user_id'):
            if len(group) < 2:
                continue
            group = group.sort_values('timestamp')
            start = group.iloc[0]['parent_asin']
            true_next = group.iloc[1]['parent_asin']
            self.graph = g
            preds = self._traverse(start, 'breadth20')
            hits += int(true_next in preds)
            total += 1
        return hits / total if total else 0.0

    def save(self, path: str | Path) -> None:
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> 'GraphRecommender':
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    def evaluate_metrics(self, df: pd.DataFrame, policies: List[str] = None, k: int = 5) -> None:
        if policies is None:
            policies = ['one', 'breadth20', 'breadth10_depth2', 'breadth5_depth4', 'depth20']
    
        results = {}
        df_sorted = df.sort_values(['user_id', 'timestamp'])
    
        for policy in policies:
            y_true = []
            y_pred = []
            y_score = []
    
            for _, group in df_sorted.groupby('user_id'):
                if len(group) < 2:
                    continue
                group = group.sort_values("timestamp")
                user_hist = group.iloc[:-1]
                true_next = group.iloc[-1]["parent_asin"]
    
                recs = self.recommend(user_hist, policy=policy, store=False)
                y_true.append(true_next)
                y_pred.append(recs[0] if recs else None)
                y_score.append(recs)
    
            # Filter out None predictions
            valid_idx = [i for i, p in enumerate(y_pred) if p is not None]
            y_true = [y_true[i] for i in valid_idx]
            y_pred = [y_pred[i] for i in valid_idx]
            y_score = [y_score[i] for i in valid_idx]
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            hit_rate = sum(t in s for t, s in zip(y_true, y_score)) / len(y_true)
            acc_at_k = sum(t in s[:k] for t, s in zip(y_true, y_score)) / len(y_true)
            prec_at_k = np.mean([1.0 if t in s[:k] else 0.0 for t, s in zip(y_true, y_score)])
    
            # NDCG@k
            def dcg(r):
                return sum([1 / np.log2(i + 2) for i, val in enumerate(r) if val > 0])
            ndcg = np.mean([
                dcg([1 if t == s else 0 for s in topk])
                for t, topk in zip(y_true, y_score)
            ])
            results[policy] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Accuracy@K": acc_at_k,
                "Precision@K": prec_at_k,
                "NDCG": ndcg,
                "HitRate": hit_rate,
            }
    
        # Print results
        df_result = pd.DataFrame(results).T
        print(df_result)
    
        # Visualise
        df_result.plot(kind='bar', figsize=(12, 6))
        plt.title("Evaluation Metrics by Policy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def visualise(self, num_nodes: int = 100) -> None:
        sub = self.graph.copy()
        if sub.number_of_nodes() > num_nodes:
            nodes = list(sub.nodes)[:num_nodes]
            sub = sub.subgraph(nodes)
        pos = nx.spring_layout(sub, seed=SEED)
        nx.draw_networkx_nodes(sub, pos, node_size=50)
        nx.draw_networkx_edges(sub, pos, arrowstyle='->', arrowsize=5)
        plt.axis('off')
        plt.show()
