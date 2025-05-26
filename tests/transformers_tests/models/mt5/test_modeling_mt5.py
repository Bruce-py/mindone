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
from parameterized import parameterized
from transformers import MT5Config, AutoTokenizer

import mindspore as ms

from mindone.transformers import MT5ForConditionalGeneration
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


class MT5ModelTester:
    def __init__(
        self,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=7,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
        scope=None,
        decoder_layers=None,
    ):
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.scope = None
        self.decoder_layers = decoder_layers

    def get_large_model_config(self):
        return MT5Config.from_pretrained("google-t5/t5-base")

    def prepare_config_and_inputs(self):
        input_ids = np.maximum(ids_numpy([self.batch_size, self.encoder_seq_length], self.vocab_size), 2)
        input_ids[:, -1] = self.eos_token_id  # Eos Token
        decoder_input_ids = ids_numpy([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_numpy([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_numpy([self.batch_size, self.decoder_seq_length], self.vocab_size)

        config = self.get_config()

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def get_pipeline_config(self):
        return MT5Config(
            vocab_size=166,  # t5 forces 100 extra tokens
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def get_config(self):
        return MT5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )


class MT5ModelTest(unittest.TestCase):
    # 初始化用例参数
    model_tester = MT5ModelTester()
    config, input_ids, decoder_input_ids, attention_mask = model_tester.prepare_config_and_inputs()[:4]

    MT5_CASES = [
        [
            "MT5Model",
            "transformers.MT5Model",
            "mindone.transformers.MT5Model",
            (config,),
            {},
            (),
            {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids},
            {
                "last_hidden_state": 0,
            },
        ],
        [
            "MT5EncoderModel",
            "transformers.MT5EncoderModel",
            "mindone.transformers.MT5EncoderModel",
            (config,),
            {},
            (),
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            {
                "last_hidden_state": 0,
            },
        ],
    ]

    @parameterized.expand(
        [
            case
            + [
                dtype,
            ]
            + [
                mode,
            ]
            for case in MT5_CASES
            for dtype in DTYPE_AND_THRESHOLDS
            for mode in MODES
        ],
    )
    def test_model_forward(
            self,
            name,
            pt_module,
            ms_module,
            init_args,
            init_kwargs,
            inputs_args,
            inputs_kwargs,
            outputs_map,
            dtype,
            mode,
    ):
        ms.set_context(mode=mode)

        diffs, pt_dtype, ms_dtype = forward_compare(
            pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
        )

        THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
        self.assertTrue(
            (np.array(diffs) < THRESHOLD).all(),
            f"For {name} forward test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}")


class MT5IntegrationTest(unittest.TestCase):
        def test_model_inference_logits(self):
            model_name = "google-mt5/mt5-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = MT5ForConditionalGeneration.from_pretrained(model_name)

            input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="np").input_ids
            labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="np").input_ids
            output_logits = model(input_ids=ms.Tensor(input_ids), labels=ms.Tensor(labels, ms.int32))[1]

            # check the logits todo
            EXPECTED_SHAPE = ()
            self.assertEqual(output_logits.shape, EXPECTED_SHAPE)

            EXPECTED_SLICE = ms.Tensor([], ms.float32)
            np.testing.assert_allclose(output_logits[0, :10], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

        def test_model_inference_generate(self):
            model_name = "google/mt5-small"  # 支持多语言版本:ml-citation{ref="4" data="citationList"}
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = MT5ForConditionalGeneration.from_pretrained(model_name)

            input_text = "translate English to German: Hello, how are you?"
            input_ids = ms.Tensor(tokenizer(input_text, return_tensors="np").input_ids, ms.int32)

            generate_ids = model.generate(input_ids, max_length=50, do_sample=False, temperature=0)
            output_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

            # check the text
            EXPECTED_TEXT = ""
            self.assertEqual(output_text, EXPECTED_TEXT)