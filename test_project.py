"""
项目功能验证测试脚本
覆盖核心模块：配置、数据加载、数据集、模型、训练函数
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config, RobertaConfig, TextCNNConfig
from src.data.loader import clean_data, load_data, split_data
from src.data.dataset import JigsawDataset, ToxicityDataset, build_vocab, text_to_sequence
from src.models.roberta import JigsawModel
from src.models.textcnn import TextCNN
from src.utils.trainer import train_fn_roberta, eval_fn_roberta, train_epoch_cnn, evaluate_cnn


class TestConfig(unittest.TestCase):
    """测试配置模块"""

    def test_config_basic_attributes(self):
        """测试Config基本属性"""
        config = Config()
        self.assertEqual(config.DATA_DIR, "data")
        self.assertEqual(config.OUTPUT_DIR, "outputs")
        self.assertEqual(len(config.TARGET_COLS), 6)
        self.assertIn("toxic", config.TARGET_COLS)

    def test_roberta_config_inheritance(self):
        """测试RobertaConfig继承和属性"""
        config = RobertaConfig()
        self.assertIsInstance(config, Config)
        self.assertEqual(config.MODEL_NAME, "distilroberta-base")
        self.assertEqual(config.TRAIN_BATCH_SIZE, 128)
        self.assertEqual(config.MAX_LENGTH, 64)

    def test_textcnn_config_inheritance(self):
        """测试TextCNNConfig继承和属性"""
        config = TextCNNConfig()
        self.assertIsInstance(config, Config)
        self.assertEqual(config.EMBEDDING_DIM, 100)
        self.assertEqual(config.MAX_VOCAB_SIZE, 20000)
        self.assertEqual(config.N_FILTERS, 100)


class TestDataLoader(unittest.TestCase):
    """测试数据加载模块"""

    def setUp(self):
        """创建测试数据"""
        self.train_data = pd.DataFrame({
            'id': ['1', '2', '3', '4', '5'],
            'comment_text': ['Hello world', 'Nice day', 'Bad comment', 'Good job', None],
            'toxic': [0, 0, 1, 0, 0],
            'severe_toxic': [0, 0, 0, 0, 0],
            'obscene': [0, 0, 0, 0, 0],
            'threat': [0, 0, 0, 0, 0],
            'insult': [0, 0, 1, 0, 0],
            'identity_hate': [0, 0, 0, 0, 0]
        })
        self.target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def test_clean_data_fillna(self):
        """测试clean_data处理缺失值"""
        result = clean_data(self.train_data.copy(), self.target_cols, is_train=True)
        self.assertFalse(result['comment_text'].isna().any())

    def test_clean_data_filter_invalid_labels(self):
        """测试clean_data过滤无效标签"""
        data_with_invalid = self.train_data.copy()
        data_with_invalid.loc[0, 'toxic'] = -1
        result = clean_data(data_with_invalid, self.target_cols, is_train=True)
        self.assertEqual(len(result), 4)  # 过滤了1行

    def test_clean_data_test_mode(self):
        """测试clean_data测试模式"""
        test_data = pd.DataFrame({
            'id': ['1', '2'],
            'comment_text': ['Test', None],
            'text': ['Hello', None]
        })
        result = clean_data(test_data, self.target_cols, is_train=False)
        self.assertFalse(result['text'].isna().any())


class TestDataset(unittest.TestCase):
    """测试数据集类"""

    def setUp(self):
        """创建模拟tokenizer"""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        self.target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def test_jigsaw_dataset_training_mode(self):
        """测试JigsawDataset训练模式"""
        df = pd.DataFrame({
            'comment_text': ['Hello world', 'Bad comment'],
            'toxic': [0, 1],
            'severe_toxic': [0, 0],
            'obscene': [0, 0],
            'threat': [0, 0],
            'insult': [0, 1],
            'identity_hate': [0, 0]
        })
        dataset = JigsawDataset(df, self.mock_tokenizer, 64, self.target_cols, is_test=False)
        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('targets', item)
        self.assertEqual(len(item['targets']), 6)

    def test_jigsaw_dataset_test_mode(self):
        """测试JigsawDataset测试模式"""
        df = pd.DataFrame({
            'id': ['1', '2'],
            'comment_text': ['Hello', 'World']
        })
        dataset = JigsawDataset(df, self.mock_tokenizer, 64, self.target_cols, is_test=True)
        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('ids', item)
        self.assertNotIn('targets', item)

    def test_jigsaw_dataset_text_column_fallback(self):
        """测试JigsawDataset的text列名兼容"""
        df = pd.DataFrame({
            'text': ['Hello world'],
            'toxic': [0], 'severe_toxic': [0], 'obscene': [0],
            'threat': [0], 'insult': [0], 'identity_hate': [0]
        })
        dataset = JigsawDataset(df, self.mock_tokenizer, 64, self.target_cols, is_test=False)
        self.assertEqual(len(dataset), 1)

    def test_toxicity_dataset_training_mode(self):
        """测试ToxicityDataset训练模式"""
        df = pd.DataFrame({
            'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            'toxic': [0, 1],
            'severe_toxic': [0, 0],
            'obscene': [0, 0],
            'threat': [0, 0],
            'insult': [0, 1],
            'identity_hate': [0, 0]
        })
        dataset = ToxicityDataset(df, self.target_cols, is_test=False)
        self.assertEqual(len(dataset), 2)
        text, label = dataset[0]
        self.assertIsInstance(text, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(len(label), 6)

    def test_toxicity_dataset_test_mode(self):
        """测试ToxicityDataset测试模式"""
        df = pd.DataFrame({
            'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            'toxic': [0, 1],
            'severe_toxic': [0, 0],
            'obscene': [0, 0],
            'threat': [0, 0],
            'insult': [0, 1],
            'identity_hate': [0, 0]
        })
        dataset = ToxicityDataset(df, self.target_cols, is_test=True)
        self.assertEqual(len(dataset), 2)
        text = dataset[0]
        self.assertIsInstance(text, torch.Tensor)


class TestBuildVocab(unittest.TestCase):
    """测试词汇表构建"""

    def test_build_vocab_basic(self):
        """测试基本词汇表构建"""
        texts = ["hello world", "hello python", "world cup"]
        vocab = build_vocab(texts, max_size=10)
        self.assertIn('<PAD>', vocab)
        self.assertIn('<UNK>', vocab)
        self.assertEqual(vocab['<PAD>'], 0)
        self.assertEqual(vocab['<UNK>'], 1)
        self.assertIn('hello', vocab)

    def test_build_vocab_max_size(self):
        """测试词汇表大小限制"""
        texts = ["a b c d e f g h i j k l m n"]
        vocab = build_vocab(texts, max_size=5)
        self.assertLessEqual(len(vocab), 5)

    def test_build_vocab_empty_texts(self):
        """测试空文本列表"""
        vocab = build_vocab([], max_size=10)
        self.assertEqual(len(vocab), 2)  # 只有<PAD>和<UNK>


class TestTextToSequence(unittest.TestCase):
    """测试文本序列化"""

    def test_text_to_sequence_basic(self):
        """测试基本文本序列化"""
        vocab = {'<PAD>': 0, '<UNK>': 1, 'hello': 2, 'world': 3}
        result = text_to_sequence("hello world", vocab, 5)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[:2], [2, 3])

    def test_text_to_sequence_padding(self):
        """测试序列填充"""
        vocab = {'<PAD>': 0, '<UNK>': 1, 'hi': 2}
        result = text_to_sequence("hi", vocab, 5)
        self.assertEqual(result, [2, 0, 0, 0, 0])

    def test_text_to_sequence_truncation(self):
        """测试序列截断"""
        vocab = {'<PAD>': 0, '<UNK>': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5}
        result = text_to_sequence("a b c d e f", vocab, 4)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, [2, 3, 4, 5])

    def test_text_to_sequence_unknown_words(self):
        """测试未知词处理"""
        vocab = {'<PAD>': 0, '<UNK>': 1, 'hello': 2}
        result = text_to_sequence("hello unknown", vocab, 4)
        self.assertEqual(result[1], 1)  # <UNK>


class TestRoBERTaModel(unittest.TestCase):
    """测试RoBERTa模型"""

    def setUp(self):
        self.device = torch.device('cpu')

    @patch('src.models.roberta.AutoModel.from_pretrained')
    def test_model_initialization(self, mock_from_pretrained):
        """测试模型初始化"""
        mock_bert = MagicMock()
        mock_bert.config.hidden_size = 768
        mock_from_pretrained.return_value = mock_bert

        model = JigsawModel("distilroberta-base", 6, dropout=0.2)
        self.assertIsNotNone(model.bert)
        self.assertIsNotNone(model.out)

    @patch('src.models.roberta.AutoModel.from_pretrained')
    def test_model_forward(self, mock_from_pretrained):
        """测试模型前向传播"""
        mock_bert = MagicMock()
        mock_bert.config.hidden_size = 768
        mock_bert.return_value.pooler_output = torch.randn(2, 768)
        mock_from_pretrained.return_value = mock_bert

        model = JigsawModel("distilroberta-base", 6)
        model.eval()

        input_ids = torch.randint(0, 100, (2, 64))
        attention_mask = torch.ones(2, 64)

        with torch.no_grad():
            output = model(input_ids, attention_mask)

        self.assertEqual(output.shape, (2, 6))


class TestTextCNNModel(unittest.TestCase):
    """测试TextCNN模型"""

    def test_model_initialization(self):
        """测试模型初始化"""
        model = TextCNN(
            vocab_size=10000,
            embed_dim=100,
            num_classes=6,
            n_filters=100,
            filter_sizes=[3, 4, 5],
            dropout=0.5
        )
        self.assertIsNotNone(model.embedding)
        self.assertEqual(len(model.convs), 3)

    def test_model_forward(self):
        """测试模型前向传播"""
        model = TextCNN(
            vocab_size=10000,
            embed_dim=100,
            num_classes=6,
            n_filters=100,
            filter_sizes=[3, 4, 5],
            dropout=0.5
        )
        model.eval()

        x = torch.randint(0, 10000, (32, 64))  # batch_size=32, seq_len=64

        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape, (32, 6))
        self.assertTrue((output >= 0).all() and (output <= 1).all())


class TestTrainerFunctions(unittest.TestCase):
    """测试训练函数"""

    def setUp(self):
        self.device = torch.device('cpu')
        self.target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def test_train_epoch_cnn(self):
        """测试CNN训练epoch"""
        model = TextCNN(10000, 100, 6, 100, [3, 4, 5], 0.5)
        model.train()

        # 创建模拟数据
        texts = torch.randint(0, 10000, (16, 64))
        labels = torch.rand(16, 6)

        loader = [(texts, labels)]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        loss = train_epoch_cnn(model, loader, optimizer, criterion, self.device)

        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_evaluate_cnn(self):
        """测试CNN评估"""
        model = TextCNN(10000, 100, 6, 100, [3, 4, 5], 0.5)
        model.eval()

        texts = torch.randint(0, 10000, (16, 64))
        labels = torch.randint(0, 2, (16, 6)).float()

        loader = [(texts, labels)]
        criterion = torch.nn.BCELoss()

        loss, acc = evaluate_cnn(model, loader, criterion, self.device)

        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(loss, 0)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)


class TestSplitData(unittest.TestCase):
    """测试数据分割"""

    def test_split_data_basic(self):
        """测试基本数据分割"""
        df = pd.DataFrame({
            'comment_text': ['a'] * 100,
            'toxic': [0] * 50 + [1] * 50,
            'severe_toxic': [0] * 100,
            'obscene': [0] * 100,
            'threat': [0] * 100,
            'insult': [0] * 100,
            'identity_hate': [0] * 100
        })
        target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        train, val = split_data(df, target_cols, test_size=0.2, random_state=42)

        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 20)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况和异常场景"""

    def test_empty_dataframe_clean_data(self):
        """测试空DataFrame处理"""
        empty_df = pd.DataFrame(columns=['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
        target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        result = clean_data(empty_df, target_cols, is_train=True)
        self.assertEqual(len(result), 0)

    def test_toxicity_dataset_empty_input(self):
        """测试空输入的ToxicityDataset"""
        df = pd.DataFrame({
            'input_ids': [],
            'toxic': [],
            'severe_toxic': [],
            'obscene': [],
            'threat': [],
            'insult': [],
            'identity_hate': []
        })
        target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        dataset = ToxicityDataset(df, target_cols, is_test=False)
        self.assertEqual(len(dataset), 0)

    def test_text_to_sequence_single_word(self):
        """测试单个词的序列"""
        vocab = {'<PAD>': 0, '<UNK>': 1, 'hello': 2}
        result = text_to_sequence("hello", vocab, 10)
        self.assertEqual(result[0], 2)
        self.assertEqual(sum(result[1:]), 0)

    def test_text_to_sequence_all_unknown_words(self):
        """测试全是未知词的序列"""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        result = text_to_sequence("xyz abc def", vocab, 5)
        self.assertTrue(all(token == 1 for token in result))


if __name__ == '__main__':
    # 设置测试环境
    print("=" * 70)
    print("项目功能验证测试")
    print("=" * 70)
    print()

    # 运行测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestConfig,
        TestDataLoader,
        TestDataset,
        TestBuildVocab,
        TestTextToSequence,
        TestRoBERTaModel,
        TestTextCNNModel,
        TestTrainerFunctions,
        TestSplitData,
        TestEdgeCases
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # 运行测试并生成报告
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print()
    print("=" * 70)
    print("测试结果总结")
    print("=" * 70)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print()

    if result.wasSuccessful():
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败，请查看上方详细信息。")
        if result.failures:
            print("\n失败详情:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        if result.errors:
            print("\n错误详情:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")

    print("=" * 70)
