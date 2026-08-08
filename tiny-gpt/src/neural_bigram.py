"""第 2 个语言模型：用梯度下降训练的神经网络 Bigram。

运行：
    python3 tiny-gpt/src/neural_bigram.py

学习链路：
    token -> logits -> softmax -> loss -> gradient -> update
"""

import math
import random


TEXT = """
人工智能正在学习语言。
语言模型根据前文预测下一个字符。
人工智能可以学习，语言模型也可以学习。
学习从最小模型开始。
"""

LEARNING_RATE = 0.2
TRAINING_EPOCHS = 300
RANDOM_SEED = 7

# 【模块讲解】
# 这一课仍把每个字符视为一个 token。
# LEARNING_RATE（学习率）控制每次更新参数的步幅。
# TRAINING_EPOCHS 表示完整学习训练数据多少轮。
# 固定 RANDOM_SEED 可以让参数初始化和生成结果保持一致。


def build_vocabulary(text):
    """建立字符和 token ID 之间的双向映射。"""
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    return stoi, itos


# 【build_vocabulary 讲解】
# 这一部分与统计版 Bigram 相同：
#     stoi：字符 -> token ID，用于编码；
#     itos：token ID -> 字符，用于解码。
# 神经网络只处理数字，所以文本必须先转换成 token ID。


def softmax(logits):
    """把任意实数分数转换成总和为 1 的概率分布。"""
    largest_logit = max(logits)
    exponentials = [math.exp(logit - largest_logit) for logit in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


# 【softmax 讲解】
# logits 是模型给每个候选 token 的原始分数，可以为正数或负数，
# 不要求位于 0 到 1 之间，也不要求总和为 1。
#
# softmax 的数学公式是：
#     probability_i = exp(logit_i) / sum(exp(logit_j))
#
# math.exp(x) 计算 e 的 x 次方，它保证结果大于 0。
# 再除以全部结果之和，就得到总和为 1 的概率。
#
# 计算 exp 前统一减去最大 logit，不会改变 softmax 的结果，
# 却可以防止 exp(很大的数) 溢出。这叫数值稳定技巧。


class NeuralBigramLanguageModel:
    """用一张可学习的权重矩阵预测下一个 token。"""

    def __init__(self, vocab_size, rng):
        self.vocab_size = vocab_size
        self.weights = [
            [rng.uniform(-0.01, 0.01) for _ in range(vocab_size)]
            for _ in range(vocab_size)
        ]

    # 【__init__ 讲解】
    # weights 是 vocab_size × vocab_size 的可学习参数矩阵：
    #     weights[当前 token][候选的下一个 token]
    #
    # 与上一课的 counts 外形相同，但含义变了：
    #     counts  是人工统计出来的次数；
    #     weights 是模型通过梯度下降学习的实数参数。
    #
    # rng.uniform(-0.01, 0.01) 在这个区间随机生成一个小数。
    # 一开始所有权重接近 0，因此模型对各 token 的预测概率接近相等。

    def forward(self, token):
        """前向传播：根据输入 token 计算 logits 和概率。"""
        logits = self.weights[token]
        probabilities = softmax(logits)
        return logits, probabilities

    # 【forward 讲解】
    # forward 表示前向传播，也就是模型“从输入算出预测”的过程。
    #
    # 输入 token 后，weights[token] 取出权重矩阵的一行作为 logits。
    # softmax 再把 logits 转换成 probabilities。
    #
    # 这与 PyTorch 的 nn.Embedding(vocab_size, vocab_size) 本质相同：
    # 都是用 token ID 查询一行可学习参数。

    def loss(self, inputs, targets):
        """计算数据集上的平均交叉熵损失。"""
        total_loss = 0.0
        for current_token, correct_next_token in zip(inputs, targets):
            _, probabilities = self.forward(current_token)
            total_loss += -math.log(probabilities[correct_next_token])
        return total_loss / len(targets)

    # 【loss 讲解】
    # _ 接收我们不需要使用的 logits，probabilities 接收预测概率。
    # 正确答案的交叉熵损失仍然是：
    #     -log(模型分给正确 token 的概率)
    #
    # 与上一课不同的是，这次 loss 不只是评价指标。
    # 我们还会计算“每个权重对 loss 的影响”，即梯度。

    def train_step(self, current_token, correct_next_token, learning_rate):
        """对一个训练样本执行前向传播、反向传播和参数更新。"""
        _, probabilities = self.forward(current_token)
        sample_loss = -math.log(probabilities[correct_next_token])

        gradients = probabilities.copy()
        gradients[correct_next_token] -= 1.0

        for candidate_token in range(self.vocab_size):
            self.weights[current_token][candidate_token] -= (
                learning_rate * gradients[candidate_token]
            )

        return sample_loss

    # 【train_step 讲解】
    # 一个训练步骤包含三部分：
    #
    # 1. 前向传播
    #        probabilities = softmax(logits)
    #
    # 2. 反向传播
    #    softmax 与交叉熵组合后的 logit 梯度恰好是：
    #        gradient = prediction - correct_answer
    #
    #    probabilities.copy() 创建概率列表的副本。
    #    对正确 token 的位置减 1，相当于减去 one-hot 正确答案。
    #
    #    假设预测是 [0.2, 0.5, 0.3]，正确答案是第 0 项：
    #        prediction = [ 0.2,  0.5,  0.3]
    #        answer     = [ 1.0,  0.0,  0.0]
    #        gradient   = [-0.8,  0.5,  0.3]
    #
    # 3. 参数更新
    #        new_weight = old_weight - learning_rate * gradient
    #
    #    正确 token 的梯度为负，减去负数会提高它的权重；
    #    错误 token 的梯度为正，因此它们的权重会降低。

    def train(self, inputs, targets, epochs, learning_rate, rng):
        """多轮遍历训练集并更新参数。"""
        examples = list(zip(inputs, targets))

        for epoch in range(epochs):
            rng.shuffle(examples)
            for current_token, correct_next_token in examples:
                self.train_step(
                    current_token,
                    correct_next_token,
                    learning_rate,
                )

            if epoch == 0 or (epoch + 1) % 50 == 0:
                current_loss = self.loss(inputs, targets)
                print(f"第 {epoch + 1:>3} 轮，loss = {current_loss:.4f}")

    # 【train 讲解】
    # list(zip(inputs, targets)) 创建所有“输入、答案”训练样本。
    # epoch 是完整遍历一次训练集；这里总共遍历 epochs 次。
    #
    # rng.shuffle(examples) 每轮打乱样本顺序。
    # 对每个样本调用 train_step，权重就会被逐步更新。
    #
    # (epoch + 1) % 50 == 0 使用取余运算符 %，
    # 表示每完成 50 轮打印一次 loss。
    # {epoch + 1:>3} 表示输出占 3 个字符宽度并右对齐。

    def generate(self, start_token, max_new_tokens, rng):
        """根据学到的概率连续采样 token。"""
        generated = [start_token]
        current_token = start_token

        for _ in range(max_new_tokens):
            _, probabilities = self.forward(current_token)
            current_token = rng.choices(
                range(self.vocab_size),
                weights=probabilities,
                k=1,
            )[0]
            generated.append(current_token)

        return generated

    # 【generate 讲解】
    # 生成过程和上一课相同，区别在于概率的来源：
    #     上一课：由人工统计的 counts 计算；
    #     这一课：由梯度下降学到的 weights 计算。
    #
    # 每次抽出的 token 都会成为下一轮输入，这叫自回归生成。


def main():
    """准备数据，训练模型并生成文本。"""
    stoi, itos = build_vocabulary(TEXT)
    tokens = [stoi[char] for char in TEXT]
    inputs = tokens[:-1]
    targets = tokens[1:]

    training_rng = random.Random(RANDOM_SEED)
    model = NeuralBigramLanguageModel(len(stoi), training_rng)

    print(f"词表大小: {len(stoi)}")
    print(f"参数数量: {len(stoi) * len(stoi)}")
    print(f"训练前 loss: {model.loss(inputs, targets):.4f}")

    model.train(
        inputs=inputs,
        targets=targets,
        epochs=TRAINING_EPOCHS,
        learning_rate=LEARNING_RATE,
        rng=training_rng,
    )

    generated_tokens = model.generate(
        start_token=stoi["人"],
        max_new_tokens=60,
        rng=random.Random(RANDOM_SEED),
    )
    generated_text = "".join(itos[token] for token in generated_tokens)

    print(f"训练后 loss: {model.loss(inputs, targets):.4f}")
    print(f"生成文本: {generated_text}")


# 【main 讲解】
# tokens[:-1] 和 tokens[1:] 把文本错开一位，构成下一个 token 预测：
#     inputs：  人 工 智
#     targets：   工 智 能
#
# 参数数量等于 vocab_size × vocab_size。
# 当前词表有 34 个 token，所以模型有 34 × 34 = 1156 个参数。
#
# 训练前打印一次 loss，训练中持续打印 loss，训练后再打印一次。
# 如果代码正确，loss 应整体下降，这就是模型确实在学习的证据。


if __name__ == "__main__":
    main()

# 【程序入口讲解】
# 直接运行此文件时 __name__ 等于 "__main__"，因此执行 main()。
# 被其他 Python 文件导入时，模型和函数可用，但 main() 不会自动运行。
