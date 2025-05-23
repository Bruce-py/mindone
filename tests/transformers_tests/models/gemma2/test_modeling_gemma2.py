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
from transformers import Gemma2Config, AutoTokenizer

import mindspore as ms
from transformers.testing_utils import slow

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules, forward_compare,
)

from ..gemma.test_modeling_gemma import GemmaModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 1e-2}
MODES = [0, 1]


class Gemma2ModelTester(GemmaModelTester):
    config_class = Gemma2Config


class Gemma2ModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = Gemma2ModelTester()

    @parameterized.expand(
        [(dtype,) + (mode,) for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_forward(self, dtype, mode):
        ms.set_context(mode=mode)
        pt_module = "transformers.Gemma2Model"
        ms_module = "mindone.transformers.Gemma2Model"
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
            f"For Gemma2Model forward test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )

    @parameterized.expand(
        [(dtype,) + (mode,) for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_generate(self, dtype, mode):
        ms.set_context(mode=mode)
        pt_module = "transformers.Gemma2ForCausalLM"
        ms_module = "mindone.transformers.Gemma2ForCausalLM"
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
            f"For Gemma2ForCausalLM generate test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"ms_outputs_shape: {ms_outputs_np.shape}, pt_outputs_shape: {pt_outputs_np.shape},"
            f"ms_outputs: {ms_outputs_np}, pt_outputs: {pt_outputs_np}"
        )


# todo 暂无Gemma2ForCausalLM
class GemmaIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_2b_logits(self, mode):
        ms.set_context(mode=mode)
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model_name = "google/gemma-2-2b"
        model = Gemma2ForCausalLM.from_pretrained(model_name)
        input_ids = ms.tensor([input_ids], ms.int32)
        model.set_train(False)
        out_logits = model(input_ids, use_cache=False)[0].asnumpy()
        # todo add EXPECTED
        # Expected mean on dim = -1
        EXPECTED_MEAN = np.array(
            [[]]).astype(np.float32)
        np.testing.assert_allclose(out_logits.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:20]
        EXPECTED_SLICE = np.array(
            []).astype(np.float32)
        np.testing.assert_allclose(out_logits[0, 0, :20], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @parameterized.expand(MODES)
    @slow
    def test_model_2b_generate(self, mode):
        ms.set_context(mode=mode)
        # todo EXPECTED
        EXPECTED_TEXT = """"""
        prompt = "What is your favorite condiment?"
        model_name = "google/gemma-2-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = Gemma2ForCausalLM.from_pretrained(model_name)
        input_ids = ms.Tensor(tokenizer([prompt], return_tensors="np").input_ids, ms.int32)

        generated_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.assertEqual(EXPECTED_TEXT, text)
