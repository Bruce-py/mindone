# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import unittest

import numpy as np
import pytest
from parameterized import parameterized
from transformers import Blip2VisionConfig, Blip2Processor

import mindspore as ms
from transformers.testing_utils import slow

from mindone.transformers import Blip2ForConditionalGeneration
from tests.modeling_test_utils import forward_compare, prepare_img
from tests.transformers_tests.models.modeling_common import floats_numpy

# fp16 got nan
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "bf16": 5e-3}
MODES = [0, 1]


class Blip2ModelTester:
    def __init__(
        self,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=1e-10,
        scope=None,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return Blip2VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )


class Blip2ModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = Blip2ModelTester()

    @parameterized.expand(
        [[dtype,] + [mode,] for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_forward(self, dtype, mode):
        ms.set_context(mode=mode)
        pt_module = "transformers.Blip2VisionModel"
        ms_module = "mindone.transformers.Blip2VisionModel"
        blip2_text_config, pixel_values = self.model_tester.prepare_config_and_inputs()
        init_args = (blip2_text_config,)
        init_kwargs = {}
        inputs_args = (pixel_values,)
        inputs_kwargs = {}
        outputs_map = {"last_hidden_state": 0, "pooler_output": 1}

        diffs, pt_dtype, ms_dtype = forward_compare(
            pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
        )

        THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
        self.assertTrue(
            (np.array(diffs) < THRESHOLD).all(),
            f"For Blip2VisionModel forward test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )


class Blip2ModelIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_opt_2700m_generate(self, mode):
        ms.set_context(mode=mode)
        model_name = "Salesforce/blip2-opt-2.7b"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, load_in_8bit=True, mindspore_dtype=ms.float16)

        # image_url = "https://huggingface.co/hf-internal-testing/blip-test-image/resolve/main/demo.jpg"
        image_url = "/home/slg/test_mindway/data/images/demo.jpg"
        image = prepare_img(image_url)
        # case1 image
        inputs = ms.Tensor(processor(images=image, return_tensors="np")).to(ms.float16)

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        expected_ids = [50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265,
                        50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265,
                        50265, 50265, 50265, 50265, 50265, 50265, 2, 102, 693, 2828, 15, 5, 4105, 19, 10, 2335,
                        50118]  # fmt: skip
        self.assertEqual(generated_ids[0].tolist(), expected_ids)
        self.assertEqual("a woman sitting on the beach with a dog", generated_text)

        # case2 image and context
        prompt = "Question: which city is this? Answer:"
        inputs = ms.Tensor(processor(images=image, text=prompt, return_tensors="np")).to(ms.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=11)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        expected_ids = [50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265,
                        50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265, 50265,
                        50265, 50265, 50265, 50265, 50265, 50265, 2, 45641, 35, 61, 343, 16, 42, 116, 31652, 35, 24, 18,
                        45, 10, 343, 6, 24, 18, 10, 4105, 50118]  # fmt: skip
        self.assertEqual(generated_ids[0].tolist(), expected_ids)
        self.assertEqual(generated_text, "Question: which city is this? Answer: it's not a city, it's a beach")
