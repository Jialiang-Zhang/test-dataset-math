# Math Test Dataset

用于数学推理智能体与大语言模型评测的高质量数学 Benchmark。项目强调正确性、区分度、覆盖度与可复现的数据规范，而不是单纯追求题量。

## Principles

质量优先级：**正确性 > 区分度 > 覆盖度 > 数量**。

题目应尽量区分真正的数学理解与模式匹配，重点覆盖定理适用条件、充分必要条件、收敛关系、独立性、局部与全局、存在与唯一、有限维与无限维、边界情况、反例以及条件缺失等容易暴露推理漏洞的位置。

## Repository Structure

```text
test-dataset-math/
├── README.md
├── development/
│   ├── multiple-choice/
│   ├── fill-in-the-blank/
│   ├── true-false/
│   ├── calculation/
│   ├── proof/
│   └── short-answer/
├── validation/
└── held-out-test/
```

`development` 用于开发与错误分析；`validation` 用于 Prompt、Agent 策略与 SkillRAG 调整；`held-out-test` 用于最终保留评测。保留测试中的具体题目与答案不应进入智能体的提示词、技能库或训练材料。

## Subjects

长期滚动覆盖 18 个方向：离散数学、数值分析、测度积分、微分几何、概率论、抽象代数、随机过程、复分析、常微分方程、统计推断、泛函分析、线性回归、偏微分方程、高等代数、运筹学、数学分析、拓扑学、非基础及进阶课程。

## Data Format

每道题使用一个独立 JSON 文件保存。每个 JSON 对象**严格且只能包含以下 5 个字段**：

- `idx`：题目唯一编号
- `problem`：完整题目文本；选择题选项直接写入该字段
- `answer`：标准答案
- `subject`：所属数学学科
- `source`：来源及标准化设计信息

禁止加入 `solution`、`difficulty`、`question_type`、`options`、`trap_type`、`tags`、`metadata` 或任何其他字段。

选择题、判断题和填空题的 `answer` 可以只保存最终答案；计算题、证明题和简答题的 `answer` 应给出标准、严谨、可判分的必要推导与结论。

`source` 推荐使用稳定格式，例如：

- `generated | standard`
- `generated | trap: necessary-vs-sufficient`
- `generated | trap: convergence-direction`
- `generated | trap: missing-assumption`
- `generated | trap: finite-vs-infinite-dimensional`
- `generated | trap: counterexample`
- `generated | trap: theorem-condition`

## Quality Control

新增或修改的题目应独立复核。多选题逐项核验；判断题主动寻找反例；计算题独立重算；证明题检查假设、量词、定义域、边界条件与定理适用条件。存在答案不唯一、条件不足、合理歧义或复核冲突的题目不得直接入库。

持续检查 JSON 合法性、五字段约束、LaTeX 转义、`idx` 唯一连续性、语义重复与简单数字换皮。长期保持约 30%-40% 的高区分度陷阱题，但陷阱必须来自真实数学概念，而不是文字歧义。
