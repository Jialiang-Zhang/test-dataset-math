#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

@dataclass(frozen=True)
class Rule:
    key: str
    pattern: str
    zh: str

RULES = (
    Rule("foci",r"\bfoci\b","焦点"), Rule("holomorphic_function",r"\bholomorphic\s+function(?:s)?\b","全纯函数"),
    Rule("holomorphic",r"\bholomorphic\b","全纯"), Rule("meromorphic",r"\bmeromorphic\b","亚纯"),
    Rule("residue",r"\bresidue(?:s)?\b","留数"), Rule("essential_singularity",r"\bessential\s+singularit(?:y|ies)\b","本性奇点"),
    Rule("removable_singularity",r"\bremovable\s+singularit(?:y|ies)\b","可去奇点"), Rule("branch_point",r"\bbranch\s+point(?:s)?\b","分支点"),
    Rule("geodesic",r"\bgeodesic(?:s)?\b","测地线"), Rule("sectional_curvature",r"\bsectional\s+curvature\b","截面曲率"),
    Rule("ricci_curvature",r"\bRicci\s+curvature\b","里奇曲率"), Rule("gaussian_curvature",r"\bGaussian\s+curvature\b","高斯曲率"),
    Rule("diffeomorphism",r"\bdiffeomorphism(?:s)?\b","微分同胚"), Rule("manifold",r"\bmanifold(?:s)?\b","流形"),
    Rule("compact_support",r"\bcompact\s+support\b","紧支集"), Rule("weak_convergence",r"\bweak\s+convergence\b","弱收敛"),
    Rule("converges_weakly",r"\bconverges?\s+weakly\b","弱收敛"), Rule("uniform_convergence",r"\buniform\s+convergence\b","一致收敛"),
    Rule("pointwise_convergence",r"\bpointwise\s+convergence\b","逐点收敛"), Rule("almost_everywhere",r"\balmost\s+everywhere\b","几乎处处"),
    Rule("almost_surely",r"\balmost\s+surely\b","几乎必然"), Rule("absolutely_continuous",r"\babsolutely\s+continuous\b","绝对连续"),
    Rule("bounded_variation",r"\bbounded\s+variation\b","有界变差"), Rule("measurable",r"\bmeasurable\b","可测"),
    Rule("lebesgue_measure",r"\bLebesgue\s+measure\b","勒贝格测度"), Rule("banach_space",r"\bBanach\s+space(?:s)?\b","巴拿赫空间"),
    Rule("hilbert_space",r"\bHilbert\s+space(?:s)?\b","希尔伯特空间"), Rule("normed_space",r"\bnormed\s+(?:linear\s+)?space(?:s)?\b","赋范空间"),
    Rule("bounded_linear_operator",r"\bbounded\s+linear\s+operator(?:s)?\b","有界线性算子"), Rule("compact_operator",r"\bcompact\s+operator(?:s)?\b","紧算子"),
    Rule("self_adjoint",r"\bself[- ]adjoint\b","自伴"), Rule("weak_solution",r"\bweak\s+solution(?:s)?\b","弱解"),
    Rule("positive_semidefinite",r"\bpositive\s+semidefinite\b","半正定"), Rule("positive_definite",r"\bpositive\s+definite\b","正定"),
    Rule("eigenvalue",r"\beigenvalue(?:s)?\b","特征值"), Rule("eigenvector",r"\beigenvector(?:s)?\b","特征向量"),
    Rule("injective",r"\binjective\b","单射"), Rule("surjective",r"\bsurjective\b","满射"), Rule("bijective",r"\bbijective\b","双射"),
    Rule("isomorphism",r"\bisomorphism(?:s)?\b","同构"), Rule("homomorphism",r"\bhomomorphism(?:s)?\b","同态"),
    Rule("automorphism",r"\bautomorphism(?:s)?\b","自同构"), Rule("normal_subgroup",r"\bnormal\s+subgroup(?:s)?\b","正规子群"),
    Rule("quotient_group",r"\bquotient\s+group(?:s)?\b","商群"), Rule("maximal_ideal",r"\bmaximal\s+ideal(?:s)?\b","极大理想"),
    Rule("prime_ideal",r"\bprime\s+ideal(?:s)?\b","素理想"), Rule("field_extension",r"\bfield\s+extension(?:s)?\b","域扩张"),
    Rule("algebraically_closed",r"\balgebraically\s+closed\b","代数闭"), Rule("irreducible_polynomial",r"\birreducible\s+polynomial(?:s)?\b","不可约多项式"),
    Rule("homeomorphism",r"\bhomeomorphism(?:s)?\b","同胚"), Rule("homotopy",r"\bhomotop(?:y|ies)\b","同伦"),
    Rule("fundamental_group",r"\bfundamental\s+group(?:s)?\b","基本群"), Rule("path_connected",r"\bpath[- ]connected\b","道路连通"),
    Rule("random_variable",r"\brandom\s+variable(?:s)?\b","随机变量"), Rule("expected_value",r"\bexpected\s+value\b","期望"),
    Rule("distribution_function",r"\bdistribution\s+function(?:s)?\b","分布函数"), Rule("martingale",r"\bmartingale(?:s)?\b","鞅"),
    Rule("stopping_time",r"\bstopping\s+time(?:s)?\b","停时"), Rule("markov_chain",r"\bMarkov\s+chain(?:s)?\b","马尔可夫链"),
    Rule("confidence_interval",r"\bconfidence\s+interval(?:s)?\b","置信区间"), Rule("maximum_likelihood_estimator",r"\bmaximum\s+likelihood\s+estimator(?:s)?\b","最大似然估计量"),
    Rule("unbiased_estimator",r"\bunbiased\s+estimator(?:s)?\b","无偏估计量"), Rule("sufficient_statistic",r"\bsufficient\s+statistic(?:s)?\b","充分统计量"),
    Rule("boundary_condition",r"\bboundary\s+condition(?:s)?\b","边界条件"), Rule("initial_condition",r"\binitial\s+condition(?:s)?\b","初始条件"),
    Rule("condition_number",r"\bcondition\s+number(?:s)?\b","条件数"), Rule("truncation_error",r"\btruncation\s+error(?:s)?\b","截断误差"),
)
PATTERN = re.compile("|".join(f"(?P<R{i}>{r.pattern})" for i,r in enumerate(RULES)),re.IGNORECASE)

def rule_for(m: re.Match[str]) -> Rule:
    return RULES[int(str(m.lastgroup)[1:])]

def inject(text: str) -> str:
    return PATTERN.sub(lambda m: rule_for(m).zh,text)

def required(text: str) -> Counter[str]:
    c: Counter[str] = Counter()
    for m in PATTERN.finditer(text): c[rule_for(m).zh] += 1
    return c

def audit(source: str, translated: str) -> list[str]:
    return [f"{term}:{count}" for term,count in required(source).items() if translated.count(term) < count]

if __name__ == "__main__":
    s="An injective holomorphic function whose foci converge weakly."
    z=inject(s)
    assert all(x in z for x in ("单射","全纯","焦点","弱收敛")) and not audit(s,z)
    print("glossary ok")
