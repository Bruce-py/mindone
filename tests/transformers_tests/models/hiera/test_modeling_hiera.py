import unittest

import numpy as np
import pytest
from parameterized import parameterized
from transformers import HieraConfig

import mindspore as ms
from transformers.testing_utils import slow

from mindone.transformers import HieraForImageClassification, AutoImageProcessor, HieraModel
from tests.modeling_test_utils import forward_compare, prepare_img

from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4}
MODES = [1]


class HieraModelTester:
    def __init__(
        self,
        batch_size=13,
        image_size=[64, 64],
        mlp_ratio=1.0,
        num_channels=3,
        depths=[1, 1, 1, 1],
        patch_stride=[4, 4],
        patch_size=[7, 7],
        patch_padding=[3, 3],
        masked_unit_size=[8, 8],
        num_heads=[1, 1, 1, 1],
        embed_dim_multiplier=2.0,
        is_training=True,
        use_labels=True,
        embed_dim=8,
        hidden_act="gelu",
        decoder_hidden_size=2,
        decoder_depth=1,
        decoder_num_heads=1,
        initializer_range=0.02,
        scope=None,
        type_sequence_label_size=10,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.num_channels = num_channels
        self.depths = depths
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.patch_padding = patch_padding
        self.masked_unit_size = masked_unit_size
        self.num_heads = num_heads
        self.embed_dim_multiplier = embed_dim_multiplier
        self.is_training = is_training
        self.use_labels = use_labels
        self.embed_dim = embed_dim
        self.hidden_act = hidden_act
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.initializer_range = initializer_range
        self.scope = scope
        self.type_sequence_label_size = type_sequence_label_size

    def prepare_config_and_inputs(self):
        # Generate pixel values (B, C, H, W) as numpy float arrays
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])

        labels = None
        if self.use_labels:
            # Generate labels if needed for classification task
            labels = ids_numpy([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        # Create HieraConfig using parameters from __init__
        return HieraConfig(
            embed_dim=self.embed_dim,
            image_size=self.image_size,
            patch_stride=self.patch_stride,
            patch_size=self.patch_size,
            patch_padding=self.patch_padding,
            masked_unit_size=self.masked_unit_size,
            mlp_ratio=self.mlp_ratio,
            num_channels=self.num_channels,
            depths=self.depths,
            num_heads=self.num_heads,
            embed_dim_multiplier=self.embed_dim_multiplier,
            hidden_act=self.hidden_act,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.decoder_num_heads,
            initializer_range=self.initializer_range,
        )


class HieraModelTest(unittest.TestCase):
    # 初始化用例参数
    model_tester = HieraModelTester()
    config, pixel_values, labels = model_tester.prepare_config_and_inputs()

    HIERA_CASES = [
        [
            "HieraModel",
            "transformers.HieraModel",
            "mindone.transformers.HieraModel",
            (config,),
            {},
            (pixel_values,),
            {},
            {
                "last_hidden_state": "last_hidden_state",
            },
        ],
        [
            "HieraForImageClassification_Logits",
            "transformers.HieraForImageClassification",
            "mindone.transformers.HieraForImageClassification",
            (config,),
            {},
            (pixel_values,),
            {},
            {
                "logits": "logits",
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
            for case in HIERA_CASES
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


class HieraModelIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_inference_image_classification_logits(self, mode):
        ms.set_context(mode=mode)
        model_name = "facebook/hiera-tiny-224-in1k-hf"
        model = HieraForImageClassification.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

        # image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image_url = "/home/slg/test_mindway/data/images/000000039769.jpg"
        image = prepare_img(image_url)
        inputs = image_processor(images=image, return_tensors="np")
        pixel_values = ms.Tensor(inputs.pixel_values)

        output_logits = model(pixel_values).logits

        # check the logits
        EXPECTED_SHAPE = (1, 1000)
        self.assertEqual(output_logits.shape, EXPECTED_SHAPE)

        EXPECTED_SLICE = ms.Tensor([[0.8028, 0.2409, -0.2254, -0.3712, -0.2848]], ms.float32)
        np.testing.assert_allclose(output_logits[0, :5], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @parameterized.expand(MODES)
    @slow
    def test_mode_inference_interpolate_pos_encoding_logits(self, mode):
        ms.set_context(mode=mode)
        model_name = "facebook/hiera-tiny-224-in1k-hf"
        model = HieraModel.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(
            model_name, size={"shortest_edge": 448}, crop_size={"height": 448, "width": 448}
        )
        # image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image_url = "/home/slg/test_mindway/data/images/000000039769.jpg"
        image = prepare_img(image_url)
        inputs = image_processor(images=image, return_tensors="np")
        pixel_values = ms.Tensor(inputs.pixel_values)

        output_logits = model(pixel_values, interpolate_pos_encoding=True).logits

        # check the logits
        EXPECTED_SHAPE = (1, 196, 768)
        self.assertEqual(output_logits.shape, EXPECTED_SHAPE)

        EXPECTED_SLICE = ms.Tensor(
            [[1.7853, 0.0690, 0.3177], [2.6853, -0.2334, 0.0889], [1.5445, -0.1515, -0.0300]], ms.float32)
        np.testing.assert_allclose(output_logits[0, :3, :3], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)
