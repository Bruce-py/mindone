__version__ = "0.32.2"

from typing import TYPE_CHECKING

from .utils import _LazyModule

# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import diffusers` provides the names in the namespace without actually importing anything (and especially none of the backends).

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "loaders": ["FromOriginalModelMixin"],
    "models": [
        "AllegroTransformer3DModel",
        "AsymmetricAutoencoderKL",
        "AuraFlowTransformer2DModel",
        "AutoencoderDC",
        "AutoencoderKL",
        "AutoencoderKLAllegro",
        "AutoencoderKLCogVideoX",
        "AutoencoderKLHunyuanVideo",
        "AutoencoderKLLTXVideo",
        "AutoencoderKLMochi",
        "AutoencoderKLTemporalDecoder",
        "AutoencoderOobleck",
        "AutoencoderTiny",
        "CogVideoXTransformer3DModel",
        "CogView3PlusTransformer2DModel",
        "CogView4Transformer2DModel",
        "ConsistencyDecoderVAE",
        "ControlNetModel",
        "ControlNetUnionModel",
        "ControlNetXSAdapter",
        "DiTTransformer2DModel",
        "FluxControlNetModel",
        "FluxMultiControlNetModel",
        "FluxTransformer2DModel",
        "HunyuanDiT2DControlNetModel",
        "HunyuanDiT2DModel",
        "HunyuanDiT2DMultiControlNetModel",
        "HunyuanVideoTransformer3DModel",
        "I2VGenXLUNet",
        "Kandinsky3UNet",
        "LatteTransformer3DModel",
        "LTXVideoTransformer3DModel",
        "LuminaNextDiT2DModel",
        "MochiTransformer3DModel",
        "ModelMixin",
        "MotionAdapter",
        "MultiAdapter",
        "MultiControlNetModel",
        "PixArtTransformer2DModel",
        "PriorTransformer",
        "SanaTransformer2DModel",
        "SD3ControlNetModel",
        "SD3MultiControlNetModel",
        "SD3Transformer2DModel",
        "SparseControlNetModel",
        "StableAudioDiTModel",
        "T2IAdapter",
        "T5FilmDecoder",
        "Transformer2DModel",
        "StableCascadeUNet",
        "UNet1DModel",
        "UNet2DConditionModel",
        "UNet2DModel",
        "UNet3DConditionModel",
        "UNetControlNetXSModel",
        "UNetMotionModel",
        "UNetSpatioTemporalConditionModel",
        "UVit2DModel",
        "VQModel",
    ],
    "optimization": [
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ],
    "pipelines": [
        "AllegroPipeline",
        "AmusedImg2ImgPipeline",
        "AmusedInpaintPipeline",
        "AmusedPipeline",
        "AnimateDiffControlNetPipeline",
        "AnimateDiffPAGPipeline",
        "AnimateDiffPipeline",
        "AnimateDiffSDXLPipeline",
        "AnimateDiffSparseControlNetPipeline",
        "AnimateDiffVideoToVideoControlNetPipeline",
        "AnimateDiffVideoToVideoPipeline",
        "AudioLDMPipeline",
        "AuraFlowPipeline",
        "AutoPipelineForImage2Image",
        "AutoPipelineForInpainting",
        "AutoPipelineForText2Image",
        "BlipDiffusionControlNetPipeline",
        "BlipDiffusionPipeline",
        "CLIPImageProjection",
        "CogVideoXFunControlPipeline",
        "CogVideoXImageToVideoPipeline",
        "CogVideoXPipeline",
        "CogVideoXVideoToVideoPipeline",
        "CogView3PlusPipeline",
        "CogView4Pipeline",
        "ConsistencyModelPipeline",
        "DanceDiffusionPipeline",
        "DDIMPipeline",
        "DDPMPipeline",
        "DiffusionPipeline",
        "DiTPipeline",
        "FluxControlImg2ImgPipeline",
        "FluxControlInpaintPipeline",
        "FluxControlNetImg2ImgPipeline",
        "FluxControlNetInpaintPipeline",
        "FluxControlNetPipeline",
        "FluxControlPipeline",
        "FluxFillPipeline",
        "FluxImg2ImgPipeline",
        "FluxInpaintPipeline",
        "FluxPipeline",
        "FluxPriorReduxPipeline",
        "HunyuanDiTControlNetPipeline",
        "HunyuanDiTPAGPipeline",
        "HunyuanDiTPipeline",
        "HunyuanVideoPipeline",
        "I2VGenXLPipeline",
        "IFImg2ImgPipeline",
        "IFImg2ImgSuperResolutionPipeline",
        "IFInpaintingPipeline",
        "IFInpaintingSuperResolutionPipeline",
        "IFPipeline",
        "IFSuperResolutionPipeline",
        "KandinskyCombinedPipeline",
        "KandinskyImg2ImgCombinedPipeline",
        "KandinskyImg2ImgPipeline",
        "KandinskyInpaintCombinedPipeline",
        "KandinskyInpaintPipeline",
        "KandinskyPipeline",
        "KandinskyPriorPipeline",
        "KandinskyV22CombinedPipeline",
        "KandinskyV22ControlnetImg2ImgPipeline",
        "KandinskyV22ControlnetPipeline",
        "KandinskyV22Img2ImgCombinedPipeline",
        "KandinskyV22Img2ImgPipeline",
        "KandinskyV22InpaintCombinedPipeline",
        "KandinskyV22InpaintPipeline",
        "KandinskyV22Pipeline",
        "KandinskyV22PriorEmb2EmbPipeline",
        "KandinskyV22PriorPipeline",
        "Kandinsky3Img2ImgPipeline",
        "Kandinsky3Pipeline",
        "KolorsPAGPipeline",
        "KolorsPipeline",
        "KolorsImg2ImgPipeline",
        "LatentConsistencyModelImg2ImgPipeline",
        "LatentConsistencyModelPipeline",
        "LattePipeline",
        "LDMSuperResolutionPipeline",
        "LDMTextToImagePipeline",
        "LEditsPPPipelineStableDiffusion",
        "LEditsPPPipelineStableDiffusionXL",
        "LTXImageToVideoPipeline",
        "LTXPipeline",
        "LuminaText2ImgPipeline",
        "MarigoldDepthPipeline",
        "MarigoldNormalsPipeline",
        "MochiPipeline",
        "MusicLDMPipeline",
        "PixArtAlphaPipeline",
        "PixArtSigmaPAGPipeline",
        "PixArtSigmaPipeline",
        "SanaPAGPipeline",
        "SanaPipeline",
        "ShapEImg2ImgPipeline",
        "ShapEPipeline",
        "StableAudioPipeline",
        "StableCascadeCombinedPipeline",
        "StableCascadeDecoderPipeline",
        "StableCascadePriorPipeline",
        "StableDiffusion3ControlNetPipeline",
        "StableDiffusion3ControlNetInpaintingPipeline",
        "StableDiffusion3Img2ImgPipeline",
        "StableDiffusion3InpaintPipeline",
        "StableDiffusion3PAGImg2ImgPipeline",
        "StableDiffusion3PAGPipeline",
        "StableDiffusion3Pipeline",
        "StableDiffusionAdapterPipeline",
        "StableDiffusionControlNetImg2ImgPipeline",
        "StableDiffusionControlNetInpaintPipeline",
        "StableDiffusionControlNetPAGInpaintPipeline",
        "StableDiffusionControlNetPAGPipeline",
        "StableDiffusionControlNetPipeline",
        "StableDiffusionControlNetXSPipeline",
        "StableDiffusionDepth2ImgPipeline",
        "StableDiffusionDiffEditPipeline",
        "StableDiffusionGLIGENPipeline",
        "StableDiffusionGLIGENTextImagePipeline",
        "StableDiffusionImageVariationPipeline",
        "StableDiffusionImg2ImgPipeline",
        "StableDiffusionInpaintPipeline",
        "StableDiffusionInstructPix2PixPipeline",
        "StableDiffusionLatentUpscalePipeline",
        "StableDiffusionMixin",
        "StableDiffusionPAGImg2ImgPipeline",
        "StableDiffusionPAGInpaintPipeline",
        "StableDiffusionPAGPipeline",
        "StableDiffusionPipeline",
        "StableDiffusionUpscalePipeline",
        "StableDiffusionXLAdapterPipeline",
        "StableDiffusionXLControlNetImg2ImgPipeline",
        "StableDiffusionXLControlNetInpaintPipeline",
        "StableDiffusionXLControlNetPAGImg2ImgPipeline",
        "StableDiffusionXLControlNetPAGPipeline",
        "StableDiffusionXLControlNetPipeline",
        "StableDiffusionXLControlNetUnionImg2ImgPipeline",
        "StableDiffusionXLControlNetUnionInpaintPipeline",
        "StableDiffusionXLControlNetUnionPipeline",
        "StableDiffusionXLControlNetXSPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusionXLInpaintPipeline",
        "StableDiffusionXLInstructPix2PixPipeline",
        "StableDiffusionXLPAGImg2ImgPipeline",
        "StableDiffusionXLPAGInpaintPipeline",
        "StableDiffusionXLPAGPipeline",
        "StableDiffusionXLPipeline",
        "StableVideoDiffusionPipeline",
        "UnCLIPImageVariationPipeline",
        "UnCLIPPipeline",
        "UniDiffuserModel",
        "UniDiffuserPipeline",
        "UniDiffuserTextDecoder",
        "WuerstchenCombinedPipeline",
        "WuerstchenDecoderPipeline",
        "WuerstchenPriorPipeline",
    ],
    "schedulers": [
        "AmusedScheduler",
        "ConsistencyDecoderScheduler",
        "CosineDPMSolverMultistepScheduler",
        "CMStochasticIterativeScheduler",
        "CogVideoXDDIMScheduler",
        "CogVideoXDPMScheduler",
        "DDIMScheduler",
        "DDIMInverseScheduler",
        "DDIMParallelScheduler",
        "DDPMScheduler",
        "DDPMParallelScheduler",
        "DDPMWuerstchenScheduler",
        "DEISMultistepScheduler",
        "DPMSolverMultistepScheduler",
        "DPMSolverMultistepInverseScheduler",
        "DPMSolverSDEScheduler",
        "DPMSolverSinglestepScheduler",
        "EDMDPMSolverMultistepScheduler",
        "EDMEulerScheduler",
        "EulerAncestralDiscreteScheduler",
        "EulerDiscreteScheduler",
        "FlowMatchEulerDiscreteScheduler",
        "FlowMatchHeunDiscreteScheduler",
        "HeunDiscreteScheduler",
        "IPNDMScheduler",
        "KDPM2AncestralDiscreteScheduler",
        "KDPM2DiscreteScheduler",
        "LCMScheduler",
        "LMSDiscreteScheduler",
        "PNDMScheduler",
        "RePaintScheduler",
        "SASolverScheduler",
        "UnCLIPScheduler",
        "UniPCMultistepScheduler",
        "VQDiffusionScheduler",
        "SchedulerMixin",
    ],
    "utils": [
        "is_invisible_watermark_available",
        "logging",
    ],
}

if TYPE_CHECKING:
    from .configuration_utils import ConfigMixin
    from .models import (
        AllegroTransformer3DModel,
        AsymmetricAutoencoderKL,
        AuraFlowTransformer2DModel,
        AutoencoderDC,
        AutoencoderKL,
        AutoencoderKLAllegro,
        AutoencoderKLCogVideoX,
        AutoencoderKLHunyuanVideo,
        AutoencoderKLLTXVideo,
        AutoencoderKLMochi,
        AutoencoderKLTemporalDecoder,
        AutoencoderOobleck,
        AutoencoderTiny,
        CogVideoXTransformer3DModel,
        CogView3PlusTransformer2DModel,
        CogView4Transformer2DModel,
        ConsistencyDecoderVAE,
        ControlNetModel,
        ControlNetUnionModel,
        ControlNetXSAdapter,
        DiTTransformer2DModel,
        FluxControlNetModel,
        FluxMultiControlNetModel,
        FluxTransformer2DModel,
        HunyuanDiT2DControlNetModel,
        HunyuanDiT2DModel,
        HunyuanDiT2DMultiControlNetModel,
        HunyuanVideoTransformer3DModel,
        I2VGenXLUNet,
        Kandinsky3UNet,
        LatteTransformer3DModel,
        LTXVideoTransformer3DModel,
        LuminaNextDiT2DModel,
        MochiTransformer3DModel,
        ModelMixin,
        MotionAdapter,
        MultiAdapter,
        MultiControlNetModel,
        PixArtTransformer2DModel,
        PriorTransformer,
        SanaTransformer2DModel,
        SD3ControlNetModel,
        SD3MultiControlNetModel,
        SD3Transformer2DModel,
        SparseControlNetModel,
        StableAudioDiTModel,
        StableCascadeUNet,
        T2IAdapter,
        T5FilmDecoder,
        Transformer2DModel,
        UNet1DModel,
        UNet2DConditionModel,
        UNet2DModel,
        UNet3DConditionModel,
        UNetControlNetXSModel,
        UNetMotionModel,
        UNetSpatioTemporalConditionModel,
        UVit2DModel,
        VQModel,
    )
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from .pipelines import (
        AllegroPipeline,
        AmusedImg2ImgPipeline,
        AmusedInpaintPipeline,
        AmusedPipeline,
        AnimateDiffControlNetPipeline,
        AnimateDiffPAGPipeline,
        AnimateDiffPipeline,
        AnimateDiffSDXLPipeline,
        AnimateDiffSparseControlNetPipeline,
        AnimateDiffVideoToVideoControlNetPipeline,
        AnimateDiffVideoToVideoPipeline,
        AudioLDMPipeline,
        AuraFlowPipeline,
        AutoPipelineForImage2Image,
        AutoPipelineForInpainting,
        AutoPipelineForText2Image,
        BlipDiffusionControlNetPipeline,
        BlipDiffusionPipeline,
        CLIPImageProjection,
        CogVideoXFunControlPipeline,
        CogVideoXImageToVideoPipeline,
        CogVideoXPipeline,
        CogVideoXVideoToVideoPipeline,
        CogView3PlusPipeline,
        CogView4Pipeline,
        ConsistencyModelPipeline,
        DDIMPipeline,
        DDPMPipeline,
        DiffusionPipeline,
        DiTPipeline,
        FluxControlImg2ImgPipeline,
        FluxControlInpaintPipeline,
        FluxControlNetImg2ImgPipeline,
        FluxControlNetInpaintPipeline,
        FluxControlNetPipeline,
        FluxControlPipeline,
        FluxFillPipeline,
        FluxImg2ImgPipeline,
        FluxInpaintPipeline,
        FluxPipeline,
        FluxPriorReduxPipeline,
        HunyuanDiTControlNetPipeline,
        HunyuanDiTPAGPipeline,
        HunyuanDiTPipeline,
        HunyuanVideoPipeline,
        I2VGenXLPipeline,
        IFImg2ImgPipeline,
        IFImg2ImgSuperResolutionPipeline,
        IFInpaintingPipeline,
        IFInpaintingSuperResolutionPipeline,
        IFPipeline,
        IFSuperResolutionPipeline,
        Kandinsky3Img2ImgPipeline,
        Kandinsky3Pipeline,
        KandinskyCombinedPipeline,
        KandinskyImg2ImgCombinedPipeline,
        KandinskyImg2ImgPipeline,
        KandinskyInpaintCombinedPipeline,
        KandinskyInpaintPipeline,
        KandinskyPipeline,
        KandinskyPriorPipeline,
        KandinskyV22CombinedPipeline,
        KandinskyV22ControlnetImg2ImgPipeline,
        KandinskyV22ControlnetPipeline,
        KandinskyV22Img2ImgCombinedPipeline,
        KandinskyV22Img2ImgPipeline,
        KandinskyV22InpaintCombinedPipeline,
        KandinskyV22InpaintPipeline,
        KandinskyV22Pipeline,
        KandinskyV22PriorEmb2EmbPipeline,
        KandinskyV22PriorPipeline,
        KolorsImg2ImgPipeline,
        KolorsPAGPipeline,
        KolorsPipeline,
        LatentConsistencyModelImg2ImgPipeline,
        LatentConsistencyModelPipeline,
        LattePipeline,
        LDMSuperResolutionPipeline,
        LDMTextToImagePipeline,
        LEditsPPPipelineStableDiffusion,
        LEditsPPPipelineStableDiffusionXL,
        LTXImageToVideoPipeline,
        LTXPipeline,
        LuminaText2ImgPipeline,
        MarigoldDepthPipeline,
        MarigoldNormalsPipeline,
        MochiPipeline,
        MusicLDMPipeline,
        PixArtAlphaPipeline,
        PixArtSigmaPAGPipeline,
        PixArtSigmaPipeline,
        SanaPAGPipeline,
        SanaPipeline,
        ShapEImg2ImgPipeline,
        ShapEPipeline,
        StableAudioPipeline,
        StableCascadeCombinedPipeline,
        StableCascadeDecoderPipeline,
        StableCascadePriorPipeline,
        StableDiffusion3ControlNetInpaintingPipeline,
        StableDiffusion3ControlNetPipeline,
        StableDiffusion3Img2ImgPipeline,
        StableDiffusion3InpaintPipeline,
        StableDiffusion3PAGImg2ImgPipeline,
        StableDiffusion3PAGPipeline,
        StableDiffusion3Pipeline,
        StableDiffusionAdapterPipeline,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionControlNetPAGInpaintPipeline,
        StableDiffusionControlNetPAGPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionControlNetXSPipeline,
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionDiffEditPipeline,
        StableDiffusionGLIGENPipeline,
        StableDiffusionGLIGENTextImagePipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionLatentUpscalePipeline,
        StableDiffusionMixin,
        StableDiffusionPAGImg2ImgPipeline,
        StableDiffusionPAGInpaintPipeline,
        StableDiffusionPAGPipeline,
        StableDiffusionPipeline,
        StableDiffusionUpscalePipeline,
        StableDiffusionXLAdapterPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLControlNetPAGImg2ImgPipeline,
        StableDiffusionXLControlNetPAGPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLControlNetUnionImg2ImgPipeline,
        StableDiffusionXLControlNetUnionInpaintPipeline,
        StableDiffusionXLControlNetUnionPipeline,
        StableDiffusionXLControlNetXSPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLInstructPix2PixPipeline,
        StableDiffusionXLPAGImg2ImgPipeline,
        StableDiffusionXLPAGInpaintPipeline,
        StableDiffusionXLPAGPipeline,
        StableDiffusionXLPipeline,
        StableVideoDiffusionPipeline,
        UnCLIPImageVariationPipeline,
        UnCLIPPipeline,
        UniDiffuserModel,
        UniDiffuserPipeline,
        UniDiffuserTextDecoder,
        WuerstchenCombinedPipeline,
        WuerstchenDecoderPipeline,
        WuerstchenPriorPipeline,
    )
    from .schedulers import (
        AmusedScheduler,
        CMStochasticIterativeScheduler,
        CogVideoXDDIMScheduler,
        CogVideoXDPMScheduler,
        ConsistencyDecoderScheduler,
        CosineDPMSolverMultistepScheduler,
        DDIMInverseScheduler,
        DDIMParallelScheduler,
        DDIMScheduler,
        DDPMParallelScheduler,
        DDPMScheduler,
        DDPMWuerstchenScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepInverseScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSDEScheduler,
        DPMSolverSinglestepScheduler,
        EDMDPMSolverMultistepScheduler,
        EDMEulerScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        FlowMatchEulerDiscreteScheduler,
        FlowMatchHeunDiscreteScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        LCMScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        RePaintScheduler,
        SASolverScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
        TCDScheduler,
        UnCLIPScheduler,
        UniPCMultistepScheduler,
        VQDiffusionScheduler,
    )
    from .utils import logging

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
