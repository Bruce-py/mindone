import unittest

import numpy as np
import pytest
from parameterized import parameterized
from transformers import ImageGPTConfig

import mindspore as ms
from transformers.testing_utils import slow

from mindone.transformers import ImageGPTForCausalImageModeling, ImageGPTImageProcessor
from tests.modeling_test_utils import forward_compare, prepare_img

from tests.transformers_tests.models.modeling_common import ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 1e-3, "fp16": 1e-2, "bf16": 1e-2}
MODES = [0, 1]


class ImageGPTModelTester:
    def __init__(
        self,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_token_type_ids=True,
        use_input_mask=True,
        use_labels=True,
        use_mc_token_ids=True,
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
        scope=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_token_type_ids = use_token_type_ids
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.use_mc_token_ids = use_mc_token_ids
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
        self.scope = None

    def prepare_config_and_inputs(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size - 1)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)

        mc_token_ids = None
        if self.use_mc_token_ids:
            mc_token_ids = ids_numpy([self.batch_size, self.num_choices], self.seq_length)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_numpy([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_numpy([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_numpy([self.batch_size], self.num_choices)

        config = self.get_config(
            gradient_checkpointing=gradient_checkpointing,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )

        head_mask = ids_numpy([self.num_hidden_layers, self.num_attention_heads], 2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        return ImageGPTConfig(
            vocab_size=self.vocab_size,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            n_inner=self.intermediate_size,
            activation_function=self.hidden_act,
            resid_pdrop=self.hidden_dropout_prob,
            attn_pdrop=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            use_cache=True,
            gradient_checkpointing=gradient_checkpointing,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )


class ImageGPTModelTest(unittest.TestCase):
    # 初始化用例参数
    model_tester = ImageGPTModelTester()
    (
        config,
        input_ids,
        input_mask,
        head_mask,
        token_type_ids,
        mc_token_ids,
        sequence_labels,
        token_labels,
        choice_labels,
    ) = model_tester.prepare_config_and_inputs()

    IMAGEGPT_CASES = [
        [
            "ImageGPTModel",
            "transformers.ImageGPTModel",
            "mindone.transformers.ImageGPTModel",
            (config,),
            {},
            (input_ids,),
            {
                "head_mask": head_mask,
            },
            {
                "last_hidden_state": "last_hidden_state",
            },
        ],
        [
            "ImageGPTForImageClassification",
            "transformers.ImageGPTForImageClassification",
            "mindone.transformers.ImageGPTForImageClassification",
            (config,),
            {},
            (input_ids,),
            {
                "head_mask": head_mask,
            },
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
            for case in IMAGEGPT_CASES
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


class ImageGPTModelIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_inference_causal_lm_head_logits(self, mode):
        ms.set_context(mode=mode)
        model_name = "/home/slg/test_mindway/data/imagegpt-small"
        # model_name = "openai/imagegpt-small"
        model = ImageGPTForCausalImageModeling.from_pretrained(model_name)
        image_processor = ImageGPTImageProcessor.from_pretrained(model_name)
        # image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image_url = "/home/slg/test_mindway/data/images/000000039769.jpg"
        image = prepare_img(image_url)
        inputs = image_processor(images=image, return_tensors="np")
        pixel_values = ms.Tensor(inputs.pixel_values)

        output_logits = model(pixel_values).logits

        # check the logits
        EXPECTED_SHAPE = (1, 1024, 512)
        self.assertEqual(output_logits.shape, EXPECTED_SHAPE)

        EXPECTED_SLICE = ms.Tensor([[2.3445, 2.6889, 2.7313],
                                    [1.0530, 1.2416, 0.5699],
                                    [0.2205, 0.7749, 0.3953]], ms.float32)

        np.testing.assert_allclose(output_logits[0, :3, :3], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)