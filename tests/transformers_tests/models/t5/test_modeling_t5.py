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
from transformers import T5Config, AutoTokenizer

import mindspore as ms
from transformers.testing_utils import slow

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


class T5ModelTester:
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
        return T5Config.from_pretrained("google-t5/t5-base")

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
        return T5Config(
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
        return T5Config(
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


class T5ModelTest(unittest.TestCase):
    # 初始化用例参数
    model_tester = T5ModelTester()
    (
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ) = model_tester.prepare_config_and_inputs()

    T5_CASES = [
        [
            "T5Model",
            "transformers.T5Model",
            "mindone.transformers.T5Model",
            (config,),
            {},
            (),
            {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids},
            {
                "last_hidden_state": 0,
            },
        ],
        [
            "T5EncoderModel",
            "transformers.T5EncoderModel",
            "mindone.transformers.T5EncoderModel",
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
            for case in T5_CASES
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

    @parameterized.expand(
        [(dtype,) + (mode,) for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_generate(self, dtype, mode):
        ms.set_context(mode=mode)
        model_name = "T5ForConditionalGeneration"
        pt_module = f"transformers.{model_name}"
        ms_module = f"mindone.transformers.{model_name}"
        init_args = (self.config,)
        init_kwargs = {}
        inputs_args = (self.input_ids,)
        inputs_kwargs = {"max_new_tokens": 15, "do_sample": False, "use_cache": False}

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
            f"For {model_name} generate test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"ms_outputs_shape: {ms_outputs_np.shape}, pt_outputs_shape: {pt_outputs_np.shape},"
            f"ms_outputs: {ms_outputs_np}, pt_outputs: {pt_outputs_np}"
        )


class T5IntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_inference_logits(self, mode):
        ms.set_context(mode=mode)
        model_name = "/home/slg/test_mindway/data/flan-t5-small"
        # model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MT5ForConditionalGeneration.from_pretrained(model_name)

        input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="np").input_ids
        labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="np").input_ids
        outputs = model(input_ids=ms.Tensor(input_ids), labels=ms.Tensor(labels, ms.int32))
        loss, logits = outputs[:2]

        # check the loss
        EXPECTED_LOSS = ms.Tensor(16.6146, ms.float32)
        np.testing.assert_allclose(loss, EXPECTED_LOSS, rtol=1e-4, atol=1e-4)

        # check the logits
        EXPECTED_SHAPE = (1, 9, 32128)
        self.assertEqual(logits.shape, EXPECTED_SHAPE)

        EXPECTED_SLICE = ms.Tensor([[-41.227314, -3.6791453, -8.1832485, 0.9304836, -12.596826],
                                    [-34.545765, -3.5687943, -1.4550631, 4.510734, -4.293913]], ms.float32)
        np.testing.assert_allclose(logits[0, :2, :5], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @parameterized.expand(MODES)
    @slow
    def test_small_generation(self, mode):
        ms.set_context(mode=mode)
        # model_name = "/home/slg/test_mindway/data/t5-small"
        model_name = "google-t5/t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MT5ForConditionalGeneration.from_pretrained(model_name)

        input_text = "summarize: Hello there"
        input_ids = ms.Tensor(tokenizer(input_text, return_tensors="np").input_ids, ms.int32)

        generate_ids = model.generate(input_ids, max_length=8, num_beams=1, do_sample=False)

        output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        self.assertTrue(output_text == "Hello there!")


    @parameterized.expand(MODES)
    @slow
    def test_model_inference_translate_en_to_de(self, mode):
        ms.set_context(mode=mode)
        model_name = "/home/slg/test_mindway/data/t5-small"
        # model_name = "google-t5/t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MT5ForConditionalGeneration.from_pretrained(model_name)

        input_text = "translate English to German: Hello, how are you?"
        input_ids = ms.Tensor(tokenizer(input_text, return_tensors="np").input_ids, ms.int32)

        generate_ids = model.generate(input_ids)
        output_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

        # check the text
        EXPECTED_TEXT = "Hallo, wie geht es Ihnen?"
        self.assertEqual(output_text, EXPECTED_TEXT)
