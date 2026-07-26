"""第 1 个语言模型：不依赖第三方库的 Bigram。

运行：
    python3 tiny-gpt/src/bigram.py

Bigram 只根据当前字符预测下一个字符：
    P(next_token | current_token)
"""

import math
import random


TEXT = """
人工智能正在学习语言。
语言模型根据前文预测下一个字符。
人工智能可以学习，语言模型也可以学习。
学习从最小模型开始。
"""

# 【模块讲解】
# import 用于导入模块。math 和 random 都属于 Python 标准库。
# TEXT 是变量，= 表示赋值。全大写表示我们把它当作常量使用。
# 三个引号包围的是多行字符串，其中的换行符也是训练数据的一部分。


def build_vocabulary(text):
    """建立 token 与整数编号之间的双向映射。"""
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    return stoi, itos


# 【build_vocabulary 讲解】
# def 用于定义函数，text 是调用函数时传入的参数。
#
# set(text)
#     去除重复字符。例如 set("人人工") 中只保留一个“人”。
#
# sorted(...)
#     把字符排序并返回列表，使每次运行获得稳定的 token 编号。
#
# enumerate(chars)
#     同时产生字符的位置和字符本身：
#         (0, "人"), (1, "工"), ...
#
# {char: index for ...}
#     这是字典推导式，生成“字符 -> 整数”的 stoi 字典。
#
# stoi.items()
#     产生 stoi 的所有 (键, 值)，将它们反过来就得到 itos。
#
# return stoi, itos
#     返回两个字典。Python 实际上将它们包装成一个元组 tuple。


class BigramLanguageModel:
    """根据当前 token 预测下一个 token 的统计语言模型。"""

    def __init__(self, vocab_size):
        self.counts = [
            [1 for _ in range(vocab_size)]
            for _ in range(vocab_size)
        ]

    # 【__init__ 讲解】
    # class 用于定义类，类可以理解为对象的设计图。
    # __init__ 是初始化方法，创建模型对象时由 Python 自动调用。
    # self 代表当前模型对象，self.counts 表示对象拥有的 counts 数据。
    #
    # 两层列表推导式创建 vocab_size × vocab_size 的二维表：
    #     counts[当前 token][下一个 token]
    #
    # 每个计数从 1 开始，而不是从 0 开始。这叫拉普拉斯平滑，
    # 可以防止没有见过的 token 组合概率直接变成 0。
    #
    # range(vocab_size) 产生从 0 到 vocab_size - 1 的整数。
    # 下划线 _ 是普通变量名，惯例上表示“这个变量不会被使用”。

    def train(self, tokens):
        """统计训练数据中所有相邻 token 对。"""
        for current_token, next_token in zip(tokens, tokens[1:]):
            self.counts[current_token][next_token] += 1

    # 【train 讲解】
    # 类中定义的函数叫方法。调用 model.train(tokens) 时，
    # Python 自动将 model 作为 self 传入。
    #
    # tokens[1:] 是列表切片，表示从索引 1 取到末尾。
    # 假设 tokens 是：
    #     [人, 工, 智, 能]
    #
    # zip(tokens, tokens[1:]) 会形成：
    #     (人, 工)、(工, 智)、(智, 能)
    #
    # for 循环依次取出每一对 token。
    # current_token, next_token 是元组解包。
    # += 1 等价于“原来的计数 = 原来的计数 + 1”。

    def probabilities(self, token):
        """返回给定 token 后所有候选 token 的概率。"""
        row = self.counts[token]
        total = sum(row)
        return [count / total for count in row]

    # 【probabilities 讲解】
    # self.counts[token] 从二维表中取出指定的一行。
    # 这一行记录“给定当前 token 后，各 token 出现过多少次”。
    #
    # sum(row) 将这一行的所有计数相加。
    # [count / total for count in row] 是列表推导式，
    # 它把每个计数除以总数，得到总和为 1 的概率分布。
    #
    # 例如计数 [1, 5, 2, 2] 会变成：
    #     [0.1, 0.5, 0.2, 0.2]

    def loss(self, inputs, targets):
        """计算所有预测的平均交叉熵损失。"""
        total_loss = 0.0
        for current_token, correct_next_token in zip(inputs, targets):
            probability = self.probabilities(current_token)[correct_next_token]
            total_loss += -math.log(probability)
        return total_loss / len(targets)

    # 【loss 讲解】
    # inputs 是输入 token，targets 是对应的正确答案。
    # zip(inputs, targets) 将每个输入与正确答案配成一对。
    #
    # probabilities(current_token) 得到所有候选 token 的概率；
    # 再用 [correct_next_token] 取出正确答案对应的概率。
    #
    # 单个样本的损失是：
    #     loss = -log(正确答案的概率)
    #
    # 正确答案概率越高，loss 越接近 0；概率越低，loss 越大。
    # len(targets) 返回样本数量，因此最后返回的是平均损失。

    def generate(self, start_token, max_new_tokens, rng):
        """从一个 token 开始，连续采样新的 token。"""
        generated = [start_token]
        current_token = start_token

        for _ in range(max_new_tokens):
            probabilities = self.probabilities(current_token)
            current_token = rng.choices(
                range(len(probabilities)),
                weights=probabilities,
                k=1,
            )[0]
            generated.append(current_token)

        return generated

    # 【generate 讲解】
    # generated 是列表，首先放入起始 token。
    # for _ in range(max_new_tokens) 将生成步骤重复指定次数。
    #
    # rng.choices(...) 根据权重进行随机抽样：
    #     range(len(probabilities))：所有候选 token ID；
    #     weights=probabilities：每个 token 的概率权重；
    #     k=1：只抽取一个 token。
    #
    # choices 返回一个列表，即使 k=1 也形如 [token]，
    # 所以使用末尾的 [0] 取出真正的 token。
    #
    # append 将新 token 添加到 generated 列表末尾。
    # 新 token 随后成为 current_token，参与下一轮预测。


def main():
    """组织数据准备、训练、生成和结果展示。"""
    stoi, itos = build_vocabulary(TEXT)
    tokens = [stoi[char] for char in TEXT]

    model = BigramLanguageModel(vocab_size=len(stoi))
    model.train(tokens)

    inputs = tokens[:-1]
    targets = tokens[1:]
    training_loss = model.loss(inputs, targets)

    start_char = "人"
    generated_tokens = model.generate(
        start_token=stoi[start_char],
        max_new_tokens=60,
        rng=random.Random(7),
    )
    generated_text = "".join(itos[token] for token in generated_tokens)

    print(f"词表大小: {len(stoi)}")
    print(f"训练样本数: {len(targets)}")
    print(f"训练损失: {training_loss:.4f}")
    print(f"生成文本: {generated_text}")

    predictions = sorted(
        zip(model.probabilities(stoi["人"]), itos.values()),
        reverse=True,
    )
    print("'人' 后面概率最高的 5 个字符:")
    for probability, char in predictions[:5]:
        print(f"  {repr(char)}: {probability:.3f}")


# 【main 讲解】
# stoi, itos = ...
#     左边两个变量接收函数返回的两个值，这叫元组解包。
#
# [stoi[char] for char in TEXT]
#     遍历每个字符并转换成整数 token，是字符级编码过程。
#
# BigramLanguageModel(vocab_size=len(stoi))
#     创建模型对象。vocab_size=... 是关键字参数；
#     len(stoi) 返回词表中的字符数量。
#
# tokens[:-1] 与 tokens[1:]
#     将同一个序列错开一个位置：
#         tokens：  人 工 智 能
#         inputs：  人 工 智
#         targets：   工 智 能
#
# random.Random(7)
#     7 是固定随机种子，使每次运行得到相同的随机生成结果。
#
# "".join(...)
#     将解码后的字符连接成一个完整字符串。
#
# f"训练损失: {training_loss:.4f}"
#     字符串前的 f 表示格式化字符串；{...} 中可以放表达式；
#     :.4f 表示将浮点数保留四位小数。
#
# sorted(..., reverse=True)
#     reverse=True 表示从大到小排序。
#
# predictions[:5]
#     列表切片，只取前五项。
#
# repr(char)
#     显示字符在 Python 中的明确形式，例如换行显示为 '\n'。


if __name__ == "__main__":
    main()

# 【程序入口讲解】
# __name__ 是 Python 自动提供的特殊变量。
# 直接运行本文件时，__name__ 等于 "__main__"，因此调用 main()。
# 如果其他文件通过 import 导入本文件，则不会自动执行 main()。
