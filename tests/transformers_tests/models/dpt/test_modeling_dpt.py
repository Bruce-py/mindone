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
from transformers import DPTConfig, DPTImageProcessor

import mindspore as ms
from transformers.testing_utils import slow

from mindone.transformers import DPTForDepthEstimation
from tests.modeling_test_utils import forward_compare, prepare_img
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

# fp16 diff too large, resize not support bf16
DTYPE_AND_THRESHOLDS = {"fp32": 1e-3}
MODES = [0, 1]


class DPTModelTester:
    def __init__(
        self,
        batch_size=2,
        image_size=32,
        patch_size=16,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        backbone_out_indices=[0, 1, 2, 3],
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        num_labels=3,
        neck_hidden_sizes=[16, 32],
        is_hybrid=False,
        scope=None,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.backbone_out_indices = backbone_out_indices
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        self.is_hybrid = is_hybrid
        self.neck_hidden_sizes = neck_hidden_sizes
        # sequence length of DPT = num_patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_numpy([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DPTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            fusion_hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            backbone_out_indices=self.backbone_out_indices,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            is_hybrid=self.is_hybrid,
            neck_hidden_sizes=self.neck_hidden_sizes,
        )


class DPTModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = DPTModelTester()

    @parameterized.expand(
        [[dtype,] + [mode,] for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_forward(self, dtype, mode):
        ms.set_context(mode=mode)
        pt_module = "transformers.DPTForDepthEstimation"
        ms_module = "mindone.transformers.DPTForDepthEstimation"
        config, pixel_values, _ = self.model_tester.prepare_config_and_inputs()
        init_args = (config,)
        init_kwargs = {}
        inputs_args = (pixel_values,)
        inputs_kwargs = {}
        outputs_map = {"predicted_depth": 0}

        diffs, pt_dtype, ms_dtype = forward_compare(
            pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
        )

        THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
        self.assertTrue(
            (np.array(diffs) < THRESHOLD).all(),
            f"For DPTForDepthEstimation forward test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )


class DPTModelIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_inference_depth_estimation(self, mode):
        ms.set_context(mode=mode)
        model_name = "Intel/dpt-large"
        image_processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForDepthEstimation.from_pretrained(model_name)

        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = prepare_img(image_url)
        inputs = image_processor(images=image, return_tensors="np")
        inputs["pixel_values"] = ms.Tensor(inputs["pixel_values"])

        outputs = model(**inputs)
        predicted_depth = outputs[0]

        # check the predicted depth
        EXPECTED_SHAPE = (1, 384, 384)
        self.assertEqual(predicted_depth.shape, EXPECTED_SHAPE)

        EXPECTED_SLICE = ms.Tensor(
            [[6.3199, 6.3629, 6.4148], [6.3850, 6.3615, 6.4166], [6.3519, 6.3176, 6.3575]]
        )
        np.testing.assert_allclose(predicted_depth[0, :3, :3], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)
