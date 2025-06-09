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
import unittest

import numpy as np
import pytest
import torch
import mindspore as ms
from parameterized import parameterized
from transformers import GemmaConfig, AutoTokenizer

from transformers.testing_utils import slow

from mindone.transformers import GemmaForCausalLM
from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    generalized_parse_args,
    get_modules,
    forward_compare,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [0, 1]


class GemmaModelTester:
    config_class = GemmaConfig

    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
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
        self.num_key_value_heads = num_key_value_heads
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

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return self.config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            head_dim=self.head_dim,
        )


class GemmaModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = GemmaModelTester()

    @parameterized.expand(
        [(dtype,) + (mode,) for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_forward(self, dtype, mode):
        ms.set_context(mode=mode)
        pt_module = "transformers.GemmaModel"
        ms_module = "mindone.transformers.GemmaModel"
        config, input_ids, _, input_mask = self.model_tester.prepare_config_and_inputs()[:4]
        init_args = (config,)
        init_kwargs = {}
        inputs_args = (input_ids,)
        inputs_kwargs = {"attention_mask": input_mask}
        outputs_map = {"last_hidden_state": 0}

        diffs, pt_dtype, ms_dtype = forward_compare(
            pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
        )

        THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
        self.assertTrue(
            (np.array(diffs) < THRESHOLD).all(),
            f"For GemmaModel forward test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )

    @parameterized.expand(
        [(dtype,) + (mode,) for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_generate(self, dtype, mode):
        ms.set_context(mode=mode)
        pt_module = "transformers.GemmaForCausalLM"
        ms_module = "mindone.transformers.GemmaForCausalLM"
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
            f"For GemmaForCausalLM generate test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"ms_outputs_shape: {ms_outputs_np.shape}, pt_outputs_shape: {pt_outputs_np.shape},"
            f"ms_outputs: {ms_outputs_np}, pt_outputs: {pt_outputs_np}"
        )


class GemmaIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_2b_logits(self, mode):
        ms.set_context(mode=mode)
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        # model_name = "google/gemma-2b"
        model_name = "/home/slg/test_mindway/data/gemma-2b"
        model = GemmaForCausalLM.from_pretrained(model_name)
        input_ids = ms.tensor([input_ids], ms.int32)
        model.set_train(False)
        out_logits = model(input_ids, use_cache=False)[0].asnumpy()
        # Expected mean on dim = -1
        EXPECTED_MEAN = np.array(
            [[-10.9505, 5.6840, -2.8602, -1.4515, 2.6672, 8.8554, 13.3415, 1.8176]]).astype(np.float32)
        np.testing.assert_allclose(out_logits.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, :4, :5]
        EXPECTED_SLICE = np.array([[-18.1809, 19.3977, -31.0383, -18.0997, -13.8313],
                                   [7.8680, 21.3193, -6.85355, 7.8942, 12.8942],
                                   [-1.7373, 11.9662, -9.3825, -1.6966, 3.0051],
                                   [3.2289, 10.3501, -1.9382, 3.2792, 3.95485]]).astype(np.float32)
        np.testing.assert_allclose(out_logits[0, :4, :5], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @parameterized.expand(MODES)
    @slow
    def test_model_2b_generate(self, mode):
        ms.set_context(mode=mode)
        # todo EXPECTED
        EXPECTED_TEXT = "Hello I am doing a project on the 1990s and I need to know what the most popular music"
        prompt = "Hello I am doing"
        model_name = "/home/slg/test_mindway/data/gemma-2b"
        # model_name = "google/gemma-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = GemmaForCausalLM.from_pretrained(model_name)
        input_ids = ms.Tensor(tokenizer([prompt], return_tensors="np").input_ids, ms.int32)

        generated_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.assertEqual(EXPECTED_TEXT, text)
