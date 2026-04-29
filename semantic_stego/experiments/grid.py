from __future__ import annotations

from semantic_stego.config.schemas import AttackConfig, ExperimentConfig


def build_attack_grid(config: ExperimentConfig) -> list[AttackConfig]:
    grid: list[AttackConfig] = []
    for attack in config.attacks:
        if attack == "none":
            grid.append(AttackConfig("none", strength=None, params={}))
        elif attack == "gaussian_noise":
            for sigma in config.noise_sigmas:
                grid.append(AttackConfig("gaussian_noise", strength=str(sigma), params={"sigma": sigma, "mean": 0.0}))
        elif attack == "gaussian_blur":
            for kernel in config.blur_kernels:
                grid.append(AttackConfig("gaussian_blur", strength=str(kernel), params={"kernel_size": kernel}))
        elif attack in {"jpeg", "jpeg_compression"}:
            for quality in config.jpeg_qualities:
                grid.append(AttackConfig("jpeg_compression", strength=str(quality), params={"quality": quality}))
        else:
            raise ValueError(f"Unsupported attack: {attack}")
    return grid
