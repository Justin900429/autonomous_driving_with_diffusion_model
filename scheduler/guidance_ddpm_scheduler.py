from typing import Optional, Tuple, Union

import torch
from diffusers.schedulers import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor

from control import GuidanceLoss
from misc.constant import GuidanceType


class GuidanceDDPMScheduler(DDPMScheduler):
    def __init__(self, cfg, **kwargs):
        super(GuidanceDDPMScheduler, self).__init__(**kwargs)
        self.use_classifier_guidance = (
            cfg.GUIDANCE.USE_COND == GuidanceType.CLASSIFIER_GUIDANCE.name
            and cfg.GUIDANCE.LOSS_LIST is not None
        )
        if self.use_classifier_guidance:
            self.guidance_loss = GuidanceLoss(cfg)

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = (
                sample.float()
            )  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = (
            torch.clamp(sample, -s, s) / s
        )  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
        target: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)
        variance = self._get_variance(timestep)

        if self.use_classifier_guidance and target is not None:
            with torch.enable_grad():
                model_std = torch.exp(0.5 * variance)
                model_output = self.guidance_loss(model_output, action, target, model_std)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        )

        # 6. Add noise
        variance = 0
        variance_scale = self._get_variance(t, predicted_variance=predicted_variance)
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            if self.variance_type == "fixed_small_log":
                variance = variance_scale * variance_noise
            elif self.variance_type == "learned_range":
                variance = variance_scale
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (variance_scale**0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )
