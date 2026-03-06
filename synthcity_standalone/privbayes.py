"""
Standalone PrivBayes (SynthCity algorithm, no synthcity/torch).
Reference: PrivBayes: Private Data Release via Bayesian Networks. (2017),
Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X.
"""
from collections import namedtuple
from itertools import combinations, product
import logging
from math import ceil
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from pgmpy.factors.discrete.CPD import TabularCPD
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pydantic import validate_arguments
from scipy.optimize import fsolve
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

log = logging.getLogger(__name__)

network_edge = namedtuple("network_edge", ["feature", "parents"])


def usefulness_minus_target(
    k: int,
    num_attributes: int,
    num_tuples: int,
    target_usefulness: int = 5,
    epsilon: float = 0.1,
) -> int:
    """Usefulness function in PrivBayes (Lemma 3)."""
    if k == num_attributes:
        usefulness = target_usefulness
    else:
        usefulness = (
            num_tuples * epsilon / ((num_attributes - k) * (2 ** (k + 3)))
        )
    return usefulness - target_usefulness


class PrivBayes:
    """PrivBayes: DP data release via Bayesian networks. Standalone (no synthcity/torch)."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        epsilon: float = 1.0,
        K: int = 0,
        n_bins: int = 100,
        mi_thresh: float = 0.01,
        target_usefulness: int = 5,
    ) -> None:
        self.epsilon = epsilon / 2
        self.K = K
        self.n_bins = n_bins
        self.target_usefulness = target_usefulness
        self.mi_thresh = mi_thresh
        self.default_k = 3
        self.mi_cache: dict = {}

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, data: pd.DataFrame) -> Any:
        self.n_columns = len(data.columns)
        self.n_records_fit = len(data)
        self.mi_cache = {}

        data, self.encoders = self._encode(data)
        log.debug("[Privbayes] Run greedy Bayes")
        self.dag = self._greedy_bayes(data)
        self.ordered_nodes = [attr for attr, _ in self.dag]
        self.display_network()

        log.debug("[Privbayes] Compute noisy cond")
        cpds = self._compute_noisy_conditional_distributions(data)

        log.debug("[Privbayes] Create net")
        self.network = BayesianNetwork()
        for child, parents in self.dag:
            self.network.add_node(child)
            for parent in parents:
                self.network.add_edge(parent, child)
        self.network.add_cpds(*cpds)

        log.info(f"[PrivBayes] network is valid = {self.network.check_model()}")
        self.model = BayesianModelSampling(self.network)
        log.info("[PrivBayes] done training")
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample(self, count: int) -> pd.DataFrame:
        log.debug(f"[PrivBayes] sample {count} examples")
        samples = self.model.forward_sample(size=count, show_progress=True)
        log.debug(f"[PrivBayes] decode {count} examples")
        return self._decode(samples)

    def _encode(self, data: pd.DataFrame) -> Any:
        data = data.copy()
        encoders = {}
        for col in data.columns:
            if len(data[col].unique()) < self.n_bins or data[col].dtype.name not in [
                "object",
                "category",
            ]:
                encoders[col] = {
                    "type": "categorical",
                    "model": LabelEncoder().fit(data[col]),
                }
                data[col] = encoders[col]["model"].transform(data[col])
            else:
                col_data = pd.cut(data[col], bins=self.n_bins)
                encoders[col] = {
                    "type": "continuous",
                    "model": LabelEncoder().fit(col_data),
                }
                data[col] = encoders[col]["model"].transform(col_data)
        return data, encoders

    def _decode(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in data.columns:
            if col not in self.encoders:
                continue
            inversed = self.encoders[col]["model"].inverse_transform(data[col])
            if self.encoders[col]["type"] == "categorical":
                data[col] = inversed
            elif self.encoders[col]["type"] == "continuous":
                output = [
                    np.random.uniform(interval.left, interval.right)
                    for interval in inversed
                ]
                data[col] = output
            else:
                raise RuntimeError(f"Invalid encoder {self.encoders[col]}")
        return data

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _greedy_bayes(self, data: pd.DataFrame) -> List:
        if self.K == 0:
            self.K = self._compute_K(data)
        log.info(f"[PrivBayes] Using K = {self.K}")
        data = data.copy()
        data.columns = data.columns.astype(str)
        num_tuples, num_attributes = data.shape
        nodes = set(data.columns)
        nodes_selected = set()
        network = []
        root = np.random.choice(data.columns)
        network.append(network_edge(feature=root, parents=[]))
        nodes_selected.add(root)
        nodes_remaining = nodes - nodes_selected

        for i in tqdm(range(len(nodes_remaining))):
            if len(nodes_remaining) == 0:
                break
            parents_pair_list = []
            mutual_info_list = []
            num_parents = min(len(nodes_selected), self.K)
            for candidate, split in product(
                nodes_remaining, range(len(nodes_selected) - num_parents + 1)
            ):
                candidate_pairs, candidate_mi = self._evaluate_parent_mutual_information(
                    data,
                    candidate=candidate,
                    parent_candidates=nodes_selected,
                    parent_limit=num_parents,
                    split=split,
                )
                parents_pair_list.extend(candidate_pairs)
                mutual_info_list.extend(candidate_mi)
            sampling_distribution = self._exponential_mechanism(
                data, parents_pair_list, mutual_info_list
            )
            candidate_idx = np.random.choice(
                list(range(len(mutual_info_list))), p=sampling_distribution
            )
            sampled_pair = parents_pair_list[candidate_idx]
            if self.mi_thresh >= mutual_info_list[candidate_idx]:
                log.info("[PrivBayes] Weak MI score, using empty parent")
                sampled_pair = network_edge(sampled_pair.feature, parents=[])
            log.info(
                f"[PrivBayes] Sampled {sampled_pair} with score {mutual_info_list[candidate_idx]}"
            )
            nodes_selected.add(sampled_pair.feature)
            network.append(sampled_pair)
            nodes_remaining = nodes - nodes_selected
        return network

    def _laplace_noise_parameter(self, n_items: int, n_features: int) -> float:
        return 2 * (n_features - self.K) / (n_items * self.epsilon)

    def _get_noisy_counts_for_attributes(
        self, raw_data: pd.DataFrame, attributes: list
    ) -> pd.DataFrame:
        data = raw_data.copy().loc[:, attributes]
        data = data.sort_values(attributes)
        stats = (
            data.groupby(attributes).size().reset_index().rename(columns={0: "count"})
        )
        noise_para = self._laplace_noise_parameter(*raw_data.shape)
        laplace_noises = np.random.laplace(0, scale=noise_para, size=stats.index.size)
        stats["count"] += laplace_noises
        stats.loc[stats["count"] < 0, "count"] = 0
        return stats

    def _get_noisy_distribution_from_counts(
        self, stats: pd.DataFrame, attribute: str, parents: list
    ) -> pd.DataFrame:
        if len(parents) > 0:
            plist = [stats[pkey].T for pkey in parents]
            output = pd.crosstab(
                stats[attribute],
                plist,
                values=stats["count"],
                aggfunc="sum",
                dropna=False,
            )
            output = output.fillna(0)
            output += 1
            output = output.values
            return output / (output.sum(axis=0) + 1e-8)
        else:
            output = stats[["count"]].values
            return output / (output.sum() + 1e-8)

    def _get_noisy_distribution_for_attribute(
        self,
        data: pd.DataFrame,
        attribute: str,
        parents: list,
        counts: Any = None,
    ) -> pd.DataFrame:
        attributes = parents + [attribute]
        if counts is None:
            counts = self._get_noisy_counts_for_attributes(data, attributes)
        else:
            counts = counts[attributes + ["count"]]
            counts = counts.sort_values(attributes)
            counts = (
                counts.groupby(attributes)
                .sum()
                .reset_index()
                .rename(columns={0: "count"})
            )
        return self._get_noisy_distribution_from_counts(counts, attribute, parents)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _compute_noisy_conditional_distributions(
        self, data: pd.DataFrame
    ) -> List:
        conditional_distributions = []
        card = data.nunique()
        first_K_attr_counts = self._get_noisy_counts_for_attributes(
            data, self.ordered_nodes[0 : self.K]
        )
        for idx in range(len(self.dag)):
            attribute, parents = self.dag[idx]
            if idx < self.K:
                node_values = self._get_noisy_distribution_for_attribute(
                    data, attribute, parents, counts=first_K_attr_counts
                )
            else:
                node_values = self._get_noisy_distribution_for_attribute(
                    data, attribute, parents
                )
            if len(parents) == 0:
                assert np.allclose(node_values.sum().sum(), 1), f"Invalid node_values"
            else:
                assert np.allclose(node_values.sum(axis=0), 1), f"Invalid node_values"
            assert np.isnan(node_values).sum() == 0, f"Invalid node_values"
            node_cpd = TabularCPD(
                variable=attribute,
                variable_card=card[attribute],
                values=node_values,
                evidence=parents,
                evidence_card=card[parents].values,
            )
            conditional_distributions.append(node_cpd)
        return conditional_distributions

    def _normalize_given_distribution(self, frequencies: List[float]) -> np.ndarray:
        distribution = np.array(frequencies, dtype=float).clip(0)
        summation = distribution.sum()
        if summation <= 0:
            return np.full_like(distribution, 1 / distribution.size)
        if np.isinf(summation):
            return self._normalize_given_distribution(np.isinf(distribution).astype(float))
        return distribution / summation

    def _calculate_sensitivity(
        self, data: pd.DataFrame, child: str, parents: List[str]
    ) -> float:
        num_tuples = len(data)
        attr_to_is_binary = {attr: data[attr].unique().size <= 2 for attr in data}
        if attr_to_is_binary[child] or (
            len(parents) == 1 and attr_to_is_binary[parents[0]]
        ):
            a = np.log(num_tuples) / num_tuples
            b = (num_tuples - 1) / num_tuples
            b_inv = num_tuples / (num_tuples - 1)
            return a + b * np.log(b_inv)
        else:
            a = (2 / num_tuples) * np.log((num_tuples + 1) / 2)
            b = (1 - 1 / num_tuples) * np.log(1 + 2 / (num_tuples - 1))
            return a + b

    def _calculate_delta(self, data: pd.DataFrame, sensitivity: float) -> float:
        num_attributes = len(data.columns)
        return (num_attributes - 1) * sensitivity / self.epsilon

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_parent_mutual_information(
        self,
        data: pd.DataFrame,
        candidate: str,
        parent_candidates: List[str],
        parent_limit: int,
        split: int,
    ) -> Tuple[List, List[float]]:
        if candidate in parent_candidates:
            raise RuntimeError(f"Candidate {candidate} already in {parent_candidates}")
        if split + parent_limit > len(parent_candidates):
            return [], []
        parents_pair_list = []
        mutual_info_list = []
        if candidate not in self.mi_cache:
            self.mi_cache[candidate] = {}
        for other_parents in combinations(parent_candidates[split:], parent_limit):
            parents = list(other_parents)
            parents_key = "_".join(sorted(parents))
            if parents_key in self.mi_cache[candidate]:
                score = self.mi_cache[candidate][parents_key]
            else:
                score = self.mutual_info_score(data, parents, candidate)
                self.mi_cache[candidate][parents_key] = score
            parents_pair_list.append(network_edge(candidate, parents=parents))
            mutual_info_list.append(score)
        return parents_pair_list, mutual_info_list

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def mutual_info_score(
        self, data: pd.DataFrame, parents: List[str], candidate: str
    ) -> float:
        if len(parents) == 0:
            return 0
        src = data[parents]
        n_clusters = min(10, len(src))
        if n_clusters < 2:
            return 0
        src_cluster = KMeans(n_clusters=n_clusters).fit(src)
        src_bins = src_cluster.predict(src)
        target = data[candidate]
        try:
            target_bins, _ = pd.cut(target, bins=self.n_bins, retbins=True)
        except Exception:
            target_bins = target
        target_bins = LabelEncoder().fit_transform(target_bins.astype(str))
        return normalized_mutual_info_score(src_bins, target_bins)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _exponential_mechanism(
        self,
        data: pd.DataFrame,
        parents_pair_list: List,
        mutual_info_list: List[float],
    ) -> List:
        delta_array = [
            self._calculate_delta(data, self._calculate_sensitivity(data, p.feature, p.parents))
            for p in parents_pair_list
        ]
        mi_array = np.array(mutual_info_list) / (2 * np.array(delta_array))
        mi_array = np.exp(mi_array)
        return self._normalize_given_distribution(mi_array).tolist()

    def _compute_K(self, data: pd.DataFrame) -> int:
        num_tuples, num_attributes = data.shape
        initial_usefulness = usefulness_minus_target(
            self.default_k, num_attributes, num_tuples, 0, self.epsilon
        )
        log.info(
            f"[PrivBayes] initial_usefulness = {initial_usefulness} target_usefulness = {self.target_usefulness}"
        )
        if initial_usefulness > self.target_usefulness:
            return self.default_k
        arguments = (num_attributes, num_tuples, self.target_usefulness, self.epsilon)
        try:
            ans = fsolve(
                usefulness_minus_target,
                np.array([int(num_attributes / 2)]),
                args=arguments,
            )[0]
            ans = ceil(float(ans))
        except RuntimeWarning:
            ans = self.default_k
        if ans < 1 or ans > num_attributes:
            ans = self.default_k
        return int(ans)

    def display_network(self) -> None:
        length = max(len(child) for child, _ in self.dag) if self.dag else 0
        log.info("Constructed Bayesian network:")
        for child, parents in self.dag:
            log.info(f"    {child:{length}} has parents {parents}.")
