# Math Test Dataset

一个用于数学推理智能体与大语言模型评测的高质量数学测试集仓库。

当前仓库首先整理多学科数学选择题，重点覆盖容易暴露模型推理漏洞的题目，包括定理适用条件、充分必要条件、收敛关系、独立性、边界情况、反例判断以及相近选项辨析等。

## Repository Structure

```text
test-dataset-math/
├── README.md
└── multiple-choice/
    ├── 0.json
    ├── 1.json
    ├── 2.json
    └── ...
```

## Data Format

`multiple-choice/` 中每道题使用一个独立的 JSON 文件保存。

每个 JSON 对象 **只允许包含以下 5 个字段**：

- `idx`：题目唯一编号
- `problem`：完整题目文本；选择题选项直接写入该字段
- `answer`：标准答案
- `subject`：所属数学学科
- `source`：题目来源

示例：

```json
{
  "idx": 0,
  "problem": "设$\\mathbb{F}_{81}$为$81$元的有限域。$T=\\{\\alpha\\in\\mathbb{F}_{81}\\mid\\mathbb{F}_{81}=\\mathbb{F}_3(\\alpha)\\}$。求$T$中元素的个数。",
  "answer": "72",
  "subject": "抽象代数",
  "source": "sample"
}
```

## Strict Schema Rules

所有题目数据必须严格遵守以下约束：

1. 只保留 `idx`、`problem`、`answer`、`subject`、`source` 五个字段。
2. 不允许加入 `solution`、`difficulty`、`question_type`、`options`、`trap_type` 或其他字段。
3. 选择题的 A/B/C/D 等选项直接写入 `problem`。
4. `answer` 只保存标准答案，不混入解析。
5. `idx` 在整个测试集中应保持唯一且连续管理。
6. 数学公式使用 LaTeX 表示，并保证 JSON 转义合法。

## Goal

本项目旨在持续构建规模更大、学科覆盖更广、具有区分度的数学评测数据，用于检验数学推理模型在知识理解、条件辨析、严谨推理和抗陷阱能力方面的表现。
