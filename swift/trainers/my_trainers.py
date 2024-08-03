import time
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, Callable
import warnings
import inspect
from collections import defaultdict

import torch
from torch import Tensor, nn
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers import DataCollator, PreTrainedTokenizerBase, AutoModelForCausalLM
from datasets import Dataset
from transformers.models.auto.modeling_auto import \
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from trl.models import create_reference_model

try:
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_zero3_enabled

# ########################################################
# from transformers.training_args import OptimizerNames, TrainingArguments, ParallelMode
from transformers.training_args import ParallelMode
from swift.trainers.my_arguments import MyOptimizerNames as OptimizerNames
from swift.trainers.my_arguments import MyHfTrainingArguments as TrainingArguments
# from .arguments import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import Adafactor, AdamW
from accelerate import PartialState
from transformers.utils import is_bitsandbytes_available, version, strtobool, logging, is_galore_torch_available, is_sagemaker_mp_enabled, is_apex_available
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)
# from transformers.trainer import logger
from transformers.trainer_utils import check_target_module_exists, EvalLoopOutput
from transformers.trainer_pt_utils import LayerWiseDummyOptimizer
from transformers.trainer_callback import TrainerCallback
from trl.import_utils import is_peft_available, is_wandb_available
import importlib

from trl import DPOTrainer as HFDPOTrainer

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

if is_apex_available():
    from apex import amp

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

logger = logging.get_logger(__name__)

class MyHfTrainer(HfTrainer):
    def __init__(self, *args, **kwargs):
        
        # argargs = args[1]
        # argargs = kwargs['args']
        # if argargs.optim == 'AGMA' or argargs.optim == 'AGMA_Lion':
        #     self.agma_gradient_accumulation_steps = argargs.gradient_accumulation_steps
        #     kwargs['args'].gradient_accumulation_steps = 1
        
        super().__init__(*args, **kwargs)


    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     # only for test, if there is an error, remove this function
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)

    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)

    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)
            
    #     # if self.args.n_gpu > 1:
    #     #     loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #     if self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss)

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #     return loss.detach() / self.args.gradient_accumulation_steps

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from .optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_TORCH_NPU_FUSED:
            try:
                from torch_npu.optim import NpuFusedAdamW

                optimizer_cls = NpuFusedAdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import FusedAdamW from torch_npu.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif args.optim == OptimizerNames.AGMA:
            from swift.trainers.optimizers.galore.AGMA import AGMA 

            optimizer_cls = AGMA
            optimizer_kwargs.update({
                "lr": args.learning_rate,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                # "accumulate_steps": self.agma_gradient_accumulation_steps,
                "weight_decay": args.weight_decay,
                "accumulate_steps": args.agma_gradient_accumulation_steps,
            })
        
        elif args.optim == OptimizerNames.AGMA_LION:
            from swift.trainers.optimizers.galore.AGMA_Lion import AGMA_Lion

            optimizer_cls = AGMA_Lion
            optimizer_kwargs.update({
                "lr": args.learning_rate,
                "betas": (args.agma_lion_beta1, args.agma_lion_beta2),
                "weight_decay": args.weight_decay,
                "accumulate_steps": args.agma_gradient_accumulation_steps,
            })

        elif args.optim == OptimizerNames.ADAN:
            from swift.trainers.optimizers.galore.adan import Adan

            optimizer_cls = Adan
            optimizer_kwargs.update({
                "lr": args.learning_rate,
                "betas": (args.adan_beta1, args.adan_beta2, args.adan_beta3),
                "eps": args.adan_epsilon,
                "weight_decay": args.weight_decay,
            })
        
        elif args.optim == OptimizerNames.LAMB:
            from swift.trainers.optimizers.galore.lamb import Lamb

            optimizer_cls = Lamb
            optimizer_kwargs.update({
                "lr": args.learning_rate,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                "weight_decay": args.weight_decay,
            })
        
        elif args.optim == OptimizerNames.SOPHIA:
            from swift.trainers.optimizers.galore.sophia import SophiaG as Sophia

            optimizer_cls = Sophia
            optimizer_kwargs.update({
                "lr": args.learning_rate,
                "betas": (args.adam_beta1, args.adam_beta2),
                "weight_decay": args.weight_decay,
            })
            
        elif args.optim in [
            OptimizerNames.ADAMW_BNB,
            OptimizerNames.ADAMW_8BIT,
            OptimizerNames.PAGED_ADAMW,
            OptimizerNames.PAGED_ADAMW_8BIT,
            OptimizerNames.LION,
            OptimizerNames.LION_8BIT,
            OptimizerNames.PAGED_LION,
            OptimizerNames.PAGED_LION_8BIT,
            OptimizerNames.RMSPROP_BNB,
            OptimizerNames.RMSPROP_8BIT,
            OptimizerNames.RMSPROP_32BIT,
        ]:
            try:
                from bitsandbytes.optim import AdamW, Lion, RMSprop

                is_paged = False
                optim_bits = 32
                optimizer_cls = None
                additional_optim_kwargs = adam_kwargs
                if "paged" in args.optim:
                    is_paged = True
                if "8bit" in args.optim:
                    optim_bits = 8
                if "adam" in args.optim:
                    optimizer_cls = AdamW
                elif "lion" in args.optim:
                    optimizer_cls = Lion
                    additional_optim_kwargs = {"betas": (args.adam_beta1, args.adam_beta2)}
                elif "rmsprop" in args.optim:
                    optimizer_cls = RMSprop
                    # Above we pass all `adam_kwargs` to the optimizer, here
                    # we only pass `optim_args` which can be passed by the user.
                    additional_optim_kwargs = optim_args

                bnb_kwargs = {"optim_bits": optim_bits}
                if "rmsprop" not in args.optim:
                    bnb_kwargs["is_paged"] = is_paged

                optimizer_kwargs.update(additional_optim_kwargs)
                optimizer_kwargs.update(bnb_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
            if is_bitsandbytes_available() and version.parse(
                importlib.metadata.version("bitsandbytes")
            ) < version.parse("0.41.1"):
                logger.warning(
                    "You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. "
                    "It is recommended to update your version as a major bug has been fixed in 8-bit optimizers."
                )
        elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
            try:
                from torchdistx.optimizers import AnyPrecisionAdamW

                optimizer_cls = AnyPrecisionAdamW
                optimizer_kwargs.update(adam_kwargs)

                # TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
                optimizer_kwargs.update(
                    {
                        "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                        "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                        "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                        "compensation_buffer_dtype": getattr(
                            torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
                        ),
                    }
                )
            except ImportError:
                raise ValueError("Please install https://github.com/pytorch/torchdistx")
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        elif args.optim == OptimizerNames.RMSPROP:
            optimizer_cls = torch.optim.RMSprop
        elif args.optim in [
            OptimizerNames.GALORE_ADAMW,
            OptimizerNames.GALORE_ADAMW_8BIT,
            OptimizerNames.GALORE_ADAFACTOR,
            OptimizerNames.GALORE_ADAMW_LAYERWISE,
            OptimizerNames.GALORE_ADAMW_8BIT_LAYERWISE,
            OptimizerNames.GALORE_ADAFACTOR_LAYERWISE,
        ]:
            if not is_galore_torch_available():
                raise ImportError(
                    "You need to install `galore_torch` in order to use GaLore optimizers"
                    " install it with `pip install git+https://github.com/jiaweizzhao/GaLore`"
                )
            from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit

            is_layerwise = args.optim.lower().endswith("layerwise")
            if is_layerwise and args.parallel_mode == ParallelMode.DISTRIBUTED:
                raise NotImplementedError("Layer-wise GaLore does not support DDP at this time")

            optimizer_mapping = {
                OptimizerNames.GALORE_ADAMW: GaLoreAdamW,
                OptimizerNames.GALORE_ADAMW_8BIT: GaLoreAdamW8bit,
                OptimizerNames.GALORE_ADAFACTOR: GaLoreAdafactor,
                OptimizerNames.GALORE_ADAMW_LAYERWISE: GaLoreAdamW,
                OptimizerNames.GALORE_ADAMW_8BIT_LAYERWISE: GaLoreAdamW8bit,
                OptimizerNames.GALORE_ADAFACTOR_LAYERWISE: GaLoreAdafactor,
            }

            optimizer_cls = optimizer_mapping[args.optim]

            if args.optim_target_modules is None:
                raise ValueError(
                    "You need to define a `optim_target_modules` in order to properly use GaLore optimizers"
                )

            if not isinstance(args.optim_target_modules, (list, str)):
                raise ValueError(
                    f"`optim_target_modules` has to be a list of strings, a string corresponding to a regex, or a specific module or 'all-linear', you passed {args.optim_target_modules}"
                )

            if model is None:
                raise ValueError("You need to pass a model in order to correctly initialize a GaLore optimizer.")

            logger.warning(
                "Activated GaLoRE fine-tuning, depending on your model size and hardware, the training might take a while before starting. Please be patient !"
            )

            all_linear = (
                isinstance(args.optim_target_modules, str)
                and args.optim_target_modules.replace("_", "-") == "all-linear"
            )

            galore_params = []
            galore_params_names = []
            for module_name, module in model.named_modules():
                target_module_exists, is_regex = check_target_module_exists(
                    args.optim_target_modules, module_name, return_is_regex=True
                )

                if not isinstance(module, nn.Linear):
                    # Warn in case we match but it's not a linear layer
                    if target_module_exists and not is_regex:
                        logger.warning(
                            f"{module_name} has been matched but ignored as GaLore only supports linear layers. Please double check your `optim_target_modules`!"
                        )

                    continue

                if not target_module_exists and not all_linear:
                    continue

                galore_params.append(module.weight)
                galore_params_names.append(module_name + ".weight")

            if len(galore_params) == 0:
                raise ValueError(
                    f"None of the target modules were found! ({args.optim_target_modules}). Please make sure to pass a valid `target_modules`."
                )

            non_galore_params = [p for n, p in model.named_parameters() if n not in galore_params_names]

            galore_optim_kwargs = {
                "rank": int(optim_args.pop("rank", 128)),
                "update_proj_gap": int(optim_args.pop("update_proj_gap", 200)),
                "scale": float(optim_args.pop("scale", 0.25)),
                "proj_type": optim_args.pop("proj_type", "std"),
            }

            # The default args are from the official repository: https://github.com/jiaweizzhao/GaLore
            param_groups = [
                {"params": non_galore_params},
                {"params": galore_params, **galore_optim_kwargs},
            ]

            if is_layerwise:
                # For layer-wise optimizers, the optimization step is done through post accumulation
                # gradient hooks. The trick is to first attach these hooks to the model parameters then
                # create a dummy optimizer that will perform no-ops in the Trainer.
                # See the original implementation or the nice implementation from @hiyouga
                # here: https://github.com/hiyouga/LLaMA-Factory/commit/8664262cde3919e10eaecbd66e8c5d356856362e#diff-ebe08ab14496dfb9e06075f0fdd36799ef6d1535cc4dd4715b74c4e3e06fe3ba
                if args.gradient_accumulation_steps != 1:
                    raise ValueError("Layerwise GaLoRE optimizer do not support gradient accumulation !")

                optimizer_dict = {}
                for param in non_galore_params:
                    param_groups = [{"params": [param]}]
                    optimizer_dict[param] = optimizer_cls(param_groups, **optimizer_kwargs)
                for param in galore_params:
                    param_groups = [{"params": [param], **galore_optim_kwargs}]
                    optimizer_dict[param] = optimizer_cls(param_groups, **optimizer_kwargs)

                def optimizer_hook(param):
                    if param.grad is not None:
                        optimizer_dict[param].step()
                        optimizer_dict[param].zero_grad()

                for param in model.parameters():
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(optimizer_hook)

                optimizer_cls = LayerWiseDummyOptimizer
                optimizer_kwargs.update({"optimizer_dict": optimizer_dict})

            optimizer_kwargs.update({"params": param_groups})

            if args.optim == OptimizerNames.GALORE_ADAFACTOR:
                optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

class MyHfSeq2SeqTrainer(MyHfTrainer, HfSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        # MyHfTrainer.__init__(self, *args, **kwargs)
        HfSeq2SeqTrainer.__init__(self, *args, **kwargs)
        MyHfTrainer.__init__(self, *args, **kwargs)

# ########################################################################


class MyHfDPOTrainer(MyHfTrainer, HFDPOTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid",
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name
        self.reference_free = reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOTrainer's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.dataset_num_proc = dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)
        
        MyHfTrainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)