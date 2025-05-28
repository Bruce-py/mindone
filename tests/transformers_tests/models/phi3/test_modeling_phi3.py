# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import inspect
import logging
import unittest

import numpy as np
import pytest
import torch
from parameterized import parameterized
from transformers import Phi3Config, AutoTokenizer

import mindspore as ms
from transformers.testing_utils import slow

from mindone.transformers import Phi3ForCausalLM
from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules, forward_compare,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [0, 1]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phi3ModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.head_dim = self.hidden_size // self.num_attention_heads

    # Copied from tests.models.mistral.test_modeling_mistral.MistralModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = np.tril(np.ones_like(input_ids))

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_numpy([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_numpy([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_numpy([self.batch_size], self.num_choices)

        config = self.get_config()
        # logger.info(f"config: {config}")
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return Phi3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_activation=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            use_cache=False,
        )


class Phi3ModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = Phi3ModelTester()

    @parameterized.expand(
        [(dtype,) + (mode,) for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_forward(self, dtype, mode):
        ms.set_context(mode=mode)
        pt_module = "transformers.Phi3Model"
        ms_module = "mindone.transformers.Phi3Model"
        config, input_ids, _, input_mask = self.model_tester.prepare_config_and_inputs()[:4]
        init_args = (config,)
        init_kwargs = {}
        inputs_args = (input_ids,)
        inputs_kwargs = {"attention_mask": input_mask}
        outputs_map = {"last_hidden_state": 0}  # key: torch attribute, value: mindspore idx

        diffs, pt_dtype, ms_dtype = forward_compare(
            pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
        )

        THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
        self.assertTrue(
            (np.array(diffs) < THRESHOLD).all(),
            f"For Phi3Model forward test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )

    @parameterized.expand(
        [(dtype,) + (mode,) for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_generate(self, dtype, mode):
        ms.set_context(mode=mode)
        pt_module = "transformers.Phi3ForCausalLM"
        ms_module = "mindone.transformers.Phi3ForCausalLM"
        config, input_ids = self.model_tester.prepare_config_and_inputs()[:2]
        init_args = (config,)
        init_kwargs = {}
        inputs_args = (input_ids,)
        inputs_kwargs = {"max_new_tokens": 10, "do_sample": False, "use_cache": False}

        (
            pt_model,
            ms_model,
            pt_dtype,
            ms_dtype,
        ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)

        pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
            pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
        )

        if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
            pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
            ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

        with torch.no_grad():
            pt_outputs = pt_model.generate(*pt_inputs_args, **pt_inputs_kwargs)
        ms_outputs = ms_model.generate(*ms_inputs_args, **ms_inputs_kwargs)
        pt_outputs_np, ms_outputs_np = pt_outputs.numpy(), ms_outputs.asnumpy()

        self.assertTrue(
            ms_outputs_np.shape == pt_outputs_np.shape and (ms_outputs_np == pt_outputs_np).all(),
            f"For Phi3ForCausalLM generate test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"ms_outputs_shape: {ms_outputs_np.shape}, pt_outputs_shape: {pt_outputs_np.shape},"
            f"ms_outputs: {ms_outputs_np}, pt_outputs: {pt_outputs_np}"
        )


class Phi3IntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_mini_4k_instruct_logits(self, mode):
        ms.set_context(mode=mode)
        input_ids = {
            "input_ids": ms.Tensor([[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], ms.int32)
        }
        model_name = "/home/slg/test_mindway/phi-3-mini-4k-instruct"
        # model_name = "microsoft/phi-3-mini-4k-instruct"
        model = Phi3ForCausalLM.from_pretrained(model_name)

        output_logits = model(**input_ids)

        EXPECTED_LOGITS = ms.Tensor([[ 0.9979, -1.9449, -2.5613, -2.2110, -0.9323, -2.2726, -3.2468, -2.0122,-1.0021,
                                       -1.2764, -1.0876, -1.2358,  3.9385,  6.2152, -0.3695, -2.3285,-1.2907, -1.8238,
                                       -1.9941, -2.2098, -0.6923, -1.6793, -1.1660, -2.0469,-0.7369, -1.4101, -1.4091,
                                       -3.1694, -1.8383, -1.1952],
                                     [ 3.0525,  1.9178,  3.7016,  0.9263,  0.3397,  1.9584,  2.1347,  0.3482, 1.3773,
                                       0.2153,  0.2798,  0.8360,  9.0936, 11.4944, -0.3575, -0.9442,-0.1246,  1.3869,
                                       0.9846,  1.7243,  0.9150,  1.0823,  0.4313,  1.5742, 0.2566, -0.1401, -1.3019,
                                       0.4967,  0.6941,  0.7214]])
        np.testing.assert_allclose(EXPECTED_LOGITS, output_logits[0, :2, :30], rtol=1e-4, atol=1e-4)

    @parameterized.expand(MODES)
    @slow
    def test_model_mini_4k_instruct_generate(self, mode):
        ms.set_context(mode=mode)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        model_name = "/home/slg/test_mindway/phi-3-mini-4k-instruct"
        # model_name = "microsoft/phi-3-mini-4k-instruct"
        model = Phi3ForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="ms")

        outputs = model.generate(inputs, max_new_tokens=32)
        output_text = tokenizer.batch_decode(outputs)

        EXPECTED_OUTPUT = [
            "<|system|> You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.<|end|><|user|> Can you provide ways to eat combinations of bananas and dragonfruits?<|end|><|assistant|> Certainly! Bananas and dragonfruits can be combined in various delicious ways. Here are some ideas for incorporating these fruits into your"
        ]
        self.assertListEqual(output_text, EXPECTED_OUTPUT)
