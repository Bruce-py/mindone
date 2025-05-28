import unittest

import numpy as np
import pytest
from parameterized import parameterized
from transformers import IJepaConfig, ViTImageProcessor

import mindspore as ms

from transformers.testing_utils import slow

from mindone.transformers import IJepaModel
from tests.modeling_test_utils import forward_compare, prepare_img

from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 1e-2}
MODES = [1]


class IJepaModelTester:
    def __init__(
        self,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        scope=None,
        encoder_stride=2,
        mask_ratio=0.5,
        attn_implementation="eager",
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.encoder_stride = encoder_stride
        self.attn_implementation = attn_implementation

        # in IJEPA, the seq length equals the number of patches (we don't add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches
        self.mask_ratio = mask_ratio
        self.num_masks = int(mask_ratio * self.seq_length)
        self.mask_length = num_patches

    def prepare_config_and_inputs(self):
        # Generate pixel values (B, C, H, W) as numpy float arrays
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            # Generate labels if needed for classification task
            labels = ids_numpy([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        # Create LevitConfig using parameters from __init__
        return IJepaConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            encoder_stride=self.encoder_stride,
            attn_implementation=self.attn_implementation,
        )


class IjepaModelTest(unittest.TestCase):
    # 初始化用例参数
    model_tester = IJepaModelTester()
    config, pixel_values, labels = model_tester.prepare_config_and_inputs()

    IJEPA_CASES = [
        [
            "IJepaModel",
            "transformers.IJepaModel",
            "mindone.transformers.IJepaModel",
            (config,),
            {},
            (pixel_values,),
            {},
            {
                "last_hidden_state": "last_hidden_state",
            },
        ],
        [
            "IJepaForImageClassification",
            "transformers.IJepaForImageClassification",
            "mindone.transformers.IJepaForImageClassification",
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
            for case in IJEPA_CASES
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


class IjepaModelIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_inference_no_head_logits(self, mode):
        ms.set_context(mode=mode)
        model_name = "/home/slg/test_mindway/data/ijepa_vith14_1k"
        # model_name = "facebook/ijepa_vith14_1k"
        model = IJepaModel.from_pretrained(model_name)
        # image_processor = AutoImageProcessor.from_pretrained(model_name)
        image_processor = ViTImageProcessor.from_pretrained(model_name)

        # image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image_url = "/home/slg/test_mindway/data/images/000000039769.jpg"
        image = prepare_img(image_url)
        inputs = image_processor(images=image, return_tensors="np")
        pixel_values = ms.Tensor(inputs.pixel_values)

        output_logits = model(pixel_values).logits

        # check the logits
        EXPECTED_SHAPE = (1, 256, 1280)
        self.assertEqual(output_logits.shape, EXPECTED_SHAPE)

        EXPECTED_SLICE = ms.Tensor([[-0.0621, -0.0054, -2.7513],
                                    [-0.1952, 0.0909, -3.9536],
                                    [0.0942, -0.0331, -1.2833]], ms.float32)

        np.testing.assert_allclose(output_logits[0, :3:, :3], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)
