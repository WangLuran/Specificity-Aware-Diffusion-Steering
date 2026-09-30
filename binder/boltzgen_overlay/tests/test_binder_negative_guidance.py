import types

import torch

from boltzgen.data import const
from boltzgen.model.modules.diffusion import (
    AtomDiffusion,
    _compose_binder_denoised,
    _dng_update_log_posterior,
    _ess_fraction_from_log_weights,
    _guidance_scale_at_progress,
    _pa_gate_theta_and_scale,
    _pa_prop2_candidate_terms,
    _pa_prop2_robustify_candidate_terms,
    _rotation_angle_degrees,
    _row_kabsch,
    _select_pa_rho_candidate,
    _systematic_resample_indices,
    _temper_log_weights_to_ess,
    _winsorize_centered_log_weights,
)


def test_pa_rho_candidate_objectives_select_requested_post_clip_surface() -> None:
    ess = [0.80, 0.90, 0.85]
    variance = [0.20, 0.30, 0.10]
    assert _select_pa_rho_candidate(
        objective="full_ess",
        total_ess=ess,
        total_logw_variance=variance,
        zero_index=0,
    ) == 1
    assert _select_pa_rho_candidate(
        objective="log_weight_variance",
        total_ess=ess,
        total_logw_variance=variance,
        zero_index=0,
    ) == 2


def test_prop2_two_field_terms_match_scalar_expansion() -> None:
    rho2 = torch.tensor([0.0, 0.75])
    d = torch.tensor([[[2.0]]])
    zeta = torch.tensor([3.0])
    g = zeta[:, None, None] * d
    delta = torch.tensor([[[0.5]]])
    jd_delta = torch.tensor([[[0.25]]])
    terms = _pa_prop2_candidate_terms(
        rho1=-2.0,
        rho2=rho2,
        gate_time=torch.tensor([0.4]),
        theta=torch.tensor([0.2]),
        gate_power=5.0,
        eta=0.1,
        lhat_temperature=0.5,
        variance=0.25,
        nu_a=torch.tensor([[[0.3]]]),
        d_y=d,
        g_y=g,
        zeta=zeta,
        delta0=delta,
        jd_delta0=jd_delta,
        jvp_shrink_alpha=1.0,
    )

    chi = 5.0 * 0.1**2 * 0.5**2 * 0.2 * 0.8
    expected = []
    for candidate in rho2.tolist():
        u = -2.0 * 2.0 + candidate * 6.0
        q = u - 6.0
        forward_alpha = -0.5 * 0.25 * u**2
        reverse_alpha = q * 0.3 + 0.5 * 0.25 * q**2
        curvature = (
            -( -2.0 + candidate * 3.0) * (0.5 * 0.25)
            + chi * (candidate - 0.5) * (2.0 * 0.5) ** 2
        )
        expected.append(0.4 + forward_alpha + reverse_alpha + curvature)
    assert torch.allclose(
        terms["total"][:, 0],
        torch.tensor(expected),
        atol=1e-6,
    )


def test_prop2_robustification_clips_terms_by_source() -> None:
    base = torch.tensor([[0.0, 1.0, 20.0, 2.0]])
    terms = {
        "gate": base,
        "reverse": torch.zeros_like(base),
        "forward_zero": torch.zeros_like(base),
        "jvp": -base,
    }
    total, diagnostics = _pa_prop2_robustify_candidate_terms(
        terms,
        gate_topk=1,
        gate_max_abs=0.0,
        jvp_topk=1,
        jvp_max_abs=0.0,
        kernel_topk=0,
        kernel_max_abs=0.0,
    )

    assert torch.isfinite(total).all()
    assert bool(diagnostics["gate_mask"][0, 2])
    assert bool(diagnostics["jvp_mask"][0, 2])
    assert abs(float(diagnostics["gate_used"][0, 2])) < 20.0
    assert abs(float(diagnostics["jvp_used"][0, 2])) < 20.0


def test_pa_gate_matches_selected_autoresearch_initial_scale() -> None:
    theta, scale = _pa_gate_theta_and_scale(
        torch.tensor(0.0),
        c=9.0,
        eta=0.005,
        gate_power=1925.92592593,
        lhat_temperature=0.75,
    )

    assert abs(float(theta) - 0.9) < 1e-7
    assert abs(float(scale) - 6.5) < 1e-6


def test_particle_ess_and_systematic_resampling() -> None:
    uniform = torch.zeros(8)
    assert abs(_ess_fraction_from_log_weights(uniform) - 1.0) < 1e-7
    concentrated = torch.tensor([0.0] + [-100.0] * 7)
    assert abs(_ess_fraction_from_log_weights(concentrated) - 0.125) < 1e-7
    indices = _systematic_resample_indices(
        concentrated,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(indices, torch.zeros(8, dtype=torch.long))


def test_log_weight_tempering_reaches_requested_ess() -> None:
    concentrated = torch.tensor([0.0, -1.0, -3.0, -8.0])
    tempered, alpha = _temper_log_weights_to_ess(
        concentrated,
        0.8,
    )

    assert 0 < alpha < 1
    assert abs(_ess_fraction_from_log_weights(tempered) - 0.8) < 1e-5


def test_gate_log_weight_winsorization_changes_only_extremes() -> None:
    values = torch.tensor([-20.0, -2.0, 0.0, 1.0, 3.0])
    clipped, mask, threshold = _winsorize_centered_log_weights(
        values,
        topk=1,
    )

    assert threshold == 3.0
    assert torch.equal(mask, torch.tensor([True, False, False, False, False]))
    assert torch.equal(clipped, torch.tensor([-3.0, -2.0, 0.0, 1.0, 3.0]))


def test_dng_posterior_tracks_which_branch_explains_the_step() -> None:
    logp = torch.tensor(-4.0)
    sample = torch.tensor([[1.0, 0.0, 0.0]])
    mean_a = torch.tensor([[0.0, 0.0, 0.0]])
    mean_b = sample.clone()

    toward_b, terms_b = _dng_update_log_posterior(
        log_posterior=logp,
        sampled_next=sample,
        mean_a=mean_a,
        mean_b=mean_b,
        variance=1.0,
        temperature=1.0,
        offset=0.0,
        p_min=1e-6,
        p_max=0.8,
    )
    toward_a, terms_a = _dng_update_log_posterior(
        log_posterior=logp,
        sampled_next=mean_a,
        mean_a=mean_a,
        mean_b=mean_b,
        variance=1.0,
        temperature=1.0,
        offset=0.0,
        p_min=1e-6,
        p_max=0.8,
    )

    assert float(toward_b) > float(logp)
    assert float(toward_a) < float(logp)
    assert terms_b["kernel_log_ratio"] > 0
    assert terms_a["kernel_log_ratio"] < 0


def test_row_kabsch_rotates_coordinates_and_vectors() -> None:
    mobile = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    expected_rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected_translation = torch.tensor([4.0, -2.0, 1.0])
    target = mobile @ expected_rotation + expected_translation

    rotation, translation, rmsd = _row_kabsch(mobile, target)

    assert torch.allclose(rotation, expected_rotation, atol=1e-5)
    assert torch.allclose(translation, expected_translation, atol=1e-5)
    assert float(rmsd) < 1e-5
    vector = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(
        vector @ rotation,
        vector @ expected_rotation,
        atol=1e-5,
    )
    assert abs(_rotation_angle_degrees(rotation) - 90.0) < 1e-4
    assert _rotation_angle_degrees(torch.eye(3)) == 0.0


def test_composition_changes_binder_only_and_aligns_negative_velocity() -> None:
    positive = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
        ]
    )
    rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([2.0, -3.0, 1.0])
    negative_binder_aligned = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    negative_noisy_binder_aligned = torch.tensor(
        [
            [3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0],
        ]
    )
    positive_indices = torch.tensor([0, 1])
    negative_indices = torch.tensor([1, 0])
    negative = torch.zeros_like(positive)
    negative[negative_indices] = (
        negative_binder_aligned - translation
    ) @ rotation.mT
    negative_noisy = torch.zeros_like(positive)
    negative_noisy[negative_indices] = (
        negative_noisy_binder_aligned - translation
    ) @ rotation.mT
    positive_noisy = torch.tensor(
        [
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    t_hat = 2.0

    guided, effective_scale, _, _ = _compose_binder_denoised(
        positive_denoised=positive,
        negative_denoised=negative,
        positive_noisy=positive_noisy,
        negative_noisy=negative_noisy,
        t_hat=t_hat,
        positive_binder_indices=positive_indices,
        negative_binder_indices=negative_indices,
        negative_to_positive_rotation=rotation,
        guidance_scale=0.5,
        max_delta_ratio=0.0,
    )

    positive_velocity = (
        positive_noisy[positive_indices] - positive[positive_indices]
    ) / t_hat
    negative_velocity_aligned = (
        negative_noisy_binder_aligned - negative_binder_aligned
    ) / t_hat
    expected = positive.clone()
    expected[positive_indices] = (
        positive[positive_indices]
        - 0.5
        * t_hat
        * (positive_velocity - negative_velocity_aligned)
    )
    assert effective_scale == 0.5
    assert torch.allclose(guided, expected)
    assert torch.equal(guided[2:], positive[2:])


def test_cosine_schedule_is_zero_then_ramps_to_full_scale() -> None:
    assert _guidance_scale_at_progress(0.8, 0.1, "cosine", 0.2, 0.6) == 0
    assert abs(
        _guidance_scale_at_progress(0.8, 0.4, "cosine", 0.2, 0.6) - 0.4
    ) < 1e-8
    assert _guidance_scale_at_progress(0.8, 0.8, "cosine", 0.2, 0.6) == 0.8


def test_sampler_accepts_mutated_peptide_atom_occupancy() -> None:
    """A Pro/Phe-like side-chain mask difference must not alter context output."""
    diffusion = AtomDiffusion.__new__(AtomDiffusion)
    torch.nn.Module.__init__(diffusion)
    diffusion.score_model = torch.nn.Linear(1, 1, bias=False)
    diffusion.sigma_data = 1.0
    diffusion.sigma_min = 0.01
    diffusion.sigma_max = 1.0
    diffusion.rho = 7
    diffusion.gamma_min = 2.0
    diffusion.gamma_0 = 0.0
    diffusion.step_scale = 1.0
    diffusion.step_scale_function = "constant"
    diffusion.step_scale_random = None
    diffusion.noise_scale = 0.0
    diffusion.noise_scale_function = "constant"
    diffusion.coordinate_augmentation_inference = False
    diffusion.alignment_reverse_diff = False
    diffusion.sampling_schedule = "af3"
    diffusion.num_sampling_steps = 2
    diffusion.eval()

    token_count = 6
    atom_count = 10
    atom_to_token = torch.zeros(
        2,
        atom_count,
        token_count,
        dtype=torch.bool,
    )
    # The unwanted mutation has two additional packed atoms, shifting the
    # binder's native slots from 5:7 to 7:9.
    packed_token_indices = (
        [0, 1, 2, 3, 4, 5, 5],
        [0, 1, 2, 3, 4, 4, 4, 5, 5],
    )
    for batch_index, token_indices in enumerate(packed_token_indices):
        for atom_index, token_index in enumerate(token_indices):
            atom_to_token[batch_index, atom_index, token_index] = 1
    atom_pad_mask = torch.zeros(2, atom_count, dtype=torch.bool)
    atom_pad_mask[0, :7] = True
    atom_pad_mask[1, :9] = True

    def encoded_name(name: str) -> torch.Tensor:
        encoded = torch.zeros(4, 64, dtype=torch.bool)
        for char_index, char in enumerate(name):
            encoded[char_index, ord(char) - 32] = True
        return encoded

    ref_atom_name_chars = torch.zeros(
        2,
        atom_count,
        4,
        64,
        dtype=torch.bool,
    )
    packed_atom_names = (
        ["CA", "CA", "CA", "N", "N", "N", "CA"],
        ["CA", "CA", "CA", "N", "N", "CB", "CG", "N", "CA"],
    )
    for batch_index, atom_names in enumerate(packed_atom_names):
        for atom_index, atom_name in enumerate(atom_names):
            ref_atom_name_chars[batch_index, atom_index] = encoded_name(
                atom_name
            )

    chain_design_mask = torch.tensor(
        [[False, False, False, False, False, True]] * 2
    )
    coords = torch.zeros(2, 1, atom_count, 3)
    coords[:, 0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[:, 0, 1] = torch.tensor([1.0, 0.0, 0.0])
    coords[:, 0, 2] = torch.tensor([0.0, 1.0, 0.0])
    feats = {
        "atom_to_token": atom_to_token,
        "ref_atom_name_chars": ref_atom_name_chars,
        "atom_pad_mask": atom_pad_mask,
        "atom_resolved_mask": atom_pad_mask.clone(),
        "fake_atom_mask": torch.zeros_like(atom_pad_mask),
        "token_pad_mask": torch.ones(2, token_count, dtype=torch.bool),
        "mol_type": torch.full(
            (2, token_count),
            const.chain_type_ids["PROTEIN"],
        ),
        "chain_design_mask": chain_design_mask,
        "asym_id": torch.tensor([[0, 0, 0, 1, 2, 3]] * 2),
        "residue_index": torch.tensor([[0, 1, 2, 0, 0, 0]] * 2),
        "coords": coords,
    }

    positive_denoised_history: list[torch.Tensor] = []

    def fake_preconditioned_forward(
        self,
        noised_atom_coords,
        sigma,
        network_condition_kwargs,
        training,
    ):
        del network_condition_kwargs, training
        updates = torch.zeros_like(noised_atom_coords)
        updates[0, 5:7] = torch.tensor([1.0, 0.0, 0.0])
        updates[1, 7:9] = torch.tensor([0.5, 0.0, 0.0])
        sigma_tensor = torch.tensor(
            sigma,
            dtype=noised_atom_coords.dtype,
            device=noised_atom_coords.device,
        )
        denoised = (
            self.c_skip(sigma_tensor) * noised_atom_coords
            + self.c_out(sigma_tensor) * updates
        )
        positive_denoised_history.append(denoised[0].clone())
        return denoised, {"r_update": updates}

    diffusion.preconditioned_network_forward = types.MethodType(
        fake_preconditioned_forward,
        diffusion,
    )
    result = diffusion._sample_binder_negative_guidance(
        atom_mask=atom_pad_mask.float(),
        num_sampling_steps=2,
        multiplicity=1,
        step_scale=1.0,
        noise_scale=0.0,
        inference_logging=False,
        feats=feats,
    )

    binder_mask = torch.tensor(
        [False, False, False, False, False, True, True, False, False, False]
    )
    assert result["sample_atom_coords"].shape == (1, atom_count, 3)
    for guided, positive in zip(
        result["x0_coords_traj"],
        positive_denoised_history,
    ):
        assert torch.equal(guided[0, ~binder_mask], positive[~binder_mask])
        assert not torch.equal(guided[0, binder_mask], positive[binder_mask])
