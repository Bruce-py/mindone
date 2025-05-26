import logging
import unittest

import numpy as np
import pytest
import torch
from parameterized import parameterized
from transformers import GLPNConfig

import mindspore as ms
from transformers.testing_utils import slow

from mindone.transformers import GLPNForDepthEstimation, GLPNImageProcessor, AutoImageProcessor
from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules, forward_compare, prepare_img

# -------------------------------------------------------------
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4}
MODES = [1]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GLPNModelTester:
    def __init__(
        self,
        batch_size=13,
        image_size=64,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[16, 32, 64, 128],
        downsampling_rates=[1, 4, 8, 16],
        num_attention_heads=[1, 2, 4, 8],
        is_training=True,
        use_labels=True,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        decoder_hidden_size=16,
        num_labels=3,
        scope=None,
        type_sequence_label_size=10,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.sr_ratios = sr_ratios
        self.depths = depths
        self.hidden_sizes = hidden_sizes
        self.downsampling_rates = downsampling_rates
        self.num_attention_heads = num_attention_heads
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.decoder_hidden_size = decoder_hidden_size
        self.num_labels = num_labels
        self.scope = scope

        self.type_sequence_label_size = type_sequence_label_size

    def prepare_config_and_inputs(self):
        # Generate pixel values (B, C, H, W) as numpy float arrays
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            # Generate labels if needed for classification task
            labels = ids_numpy([self.batch_size, self.image_size, self.image_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        # Create GLPNConfig using parameters from __init__
        return GLPNConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            num_encoder_blocks=self.num_encoder_blocks,
            depths=self.depths,
            hidden_sizes=self.hidden_sizes,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            decoder_hidden_size=self.decoder_hidden_size,
        )


class GLPNModelTest(unittest.TestCase):
    # 初始化用例参数
    model_tester = GLPNModelTester()
    config, pixel_values, labels = model_tester.prepare_config_and_inputs()

    GLPN_CASES = [
        [
            "GLPNModel",
            "transformers.GLPNModel",
            "mindone.transformers.GLPNModel",
            (config,),
            {},
            (pixel_values,),
            {},
            {
                "last_hidden_state": "last_hidden_state",
            },
        ],
        [
            "GLPNForDepthEstimation",
            "transformers.GLPNForDepthEstimation",
            "mindone.transformers.GLPNForDepthEstimation",
            (config,),
            {},
            (pixel_values,),
            {},
            {
                "predicted_depth": "predicted_depth",
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
            for case in GLPN_CASES
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


class GLPNModelIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_inference_logits(self, mode):
        ms.set_context(mode=mode)
        model_name = "vinvino02/glpn-kitti"
        model = GLPNForDepthEstimation.from_pretrained(model_name)
        # processor = GLPNImageProcessor.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

        # image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image_url = "/home/slg/test_mindway/data/images/000000039769.jpg"
        image = prepare_img(image_url)
        inputs = image_processor(images=image, return_tensors="np")

        pixel_values = ms.Tensor(inputs.pixel_values)
        outputs = model(pixel_values=pixel_values)

        EXPECTED_SHAPE = (1, 480, 640)
        self.assertEqual(outputs.shape, EXPECTED_SHAPE)
        EXPECTED_SLICE = ms.Tensor(
            [[3.4291, 2.7865, 2.5151], [3.2841, 2.7021, 2.3502], [3.1147, 2.4625, 2.2481]]
        )
        np.testing.assert_allclose(outputs[0, :3, :3], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)
