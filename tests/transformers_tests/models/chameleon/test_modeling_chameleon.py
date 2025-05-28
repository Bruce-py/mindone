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
from transformers import ChameleonConfig, ChameleonProcessor

import mindspore as ms
from transformers.testing_utils import slow

from mindone.transformers import ChameleonForConditionalGeneration
from tests.modeling_test_utils import forward_compare, prepare_img
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class ChameleonModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=35,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        image_token_id=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
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
        vq_num_embeds=5,
        vq_embed_dim=5,
        vq_channel_multiplier=[1, 4],
        vq_img_token_start_id=10,  # has to be less than vocab size when added with vq_num_embeds
        scope=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
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
        self.vq_num_embeds = vq_num_embeds
        self.vq_embed_dim = vq_embed_dim
        self.vq_channel_multiplier = vq_channel_multiplier
        self.vq_img_token_start_id = vq_img_token_start_id

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = np.tril(np.ones_like(input_ids))

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_numpy([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_numpy([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_numpy([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        # create dummy vocab map for image2bpe mapping if it needs remapping
        # we assume that vocab size is big enough to account for image tokens somewhere in the beginning
        # same way as in real ckpt, when img tokens are in first half of embeds
        # we will need "vq_num_embeds" amount of tokens

        vocab_map = {i: chr(i) for i in range(self.vocab_size)}
        vocab_map[self.image_token_id] = "<image>"
        start = self.vq_img_token_start_id
        end = self.vq_img_token_start_id + self.vq_num_embeds
        for i in range(start, end):
            image_token_infix = "".join(chr(ord("A") + int(c)) for c in str(i))
            # dummy str for each image token, anything starting with IMGIMG
            vocab_map[i] = f"IMGIMG{image_token_infix}Z"

        return ChameleonConfig(
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
            vocabulary_map={v: k for k, v in vocab_map.items()},
            vq_config=self.get_vq_config(),
        )

    def get_vq_config(self):
        return {
            "embed_dim": self.vq_embed_dim,
            "num_embeddings": self.vq_num_embeds,
            "latent_channels": self.vq_embed_dim,
            "in_channels": 3,
            "base_channels": 32,  # we have a GroupNorm of 32 groups, so can't do less
            "channel_multiplier": self.vq_channel_multiplier,
        }


class ChameleonModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = ChameleonModelTester()

    @parameterized.expand(
        [[dtype,] + [mode,] for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_forward(self, dtype, mode):
        ms.set_context(mode=mode)
        pt_module = "transformers.ChameleonModel"
        ms_module = "mindone.transformers.ChameleonModel"
        config, input_ids, input_mask, _, _, _ = self.model_tester.prepare_config_and_inputs()
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
            f"For ChameleonModel forward test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )


class ChameleonModelIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_7b_generate(self, mode):
        ms.set_context(mode=mode)
        model_name = "/home/slg/test_mindway/data/chameleon-7b"
        # model_name = "facebook/chameleon-7b"
        processor = ChameleonProcessor.from_pretrained(model_name)
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_name, mindspore_dtype=ms.float32, attn_implementation="flash_attention_2"
        )

        # image_url = "https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg"
        image_url = "/home/slg/test_mindway/data/images/the-big-dipper-1.jpg"
        image = prepare_img(image_url)
        prompt = "<image>Describe what do you see here and tell me about the history behind it?"

        inputs = processor(images=image, text=prompt, return_tensors="np")
        for k, v in inputs.items():
            inputs[k] = ms.Tensor(v)
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # check generation outputs
        EXPECTED_TEXT = [
            'Describe what do you see here and tell me about the history behind it?The image depicts a star map, with a bright blue dot in the center representing the star Alpha Centauri. The star map is a representation of the night sky, showing the positions of stars in']  # fmt: skip

        self.assertEqual(EXPECTED_TEXT, generated_text)