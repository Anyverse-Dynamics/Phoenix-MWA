# Embodiment registry configuration

The included JSON files are templates. Slot order, capacity and assigned/reserved
status define the model interface. `example_profile_*` values are placeholders,
not training dataset identities. The RoboCasa template additionally assigns slot
18 to `robocasa_gr1_delta80`.

For inference, use the registry supplied with your checkpoint and configure
`EMBODIMENT_REGISTRY` or `local_robot_softprompt.registry_path` as appropriate.
The evaluation preparation tools accept `--registry /path/to/registry.json`.
Do not change a pretrained model's slot order. The templates do not establish
compatibility with an arbitrary external checkpoint.

For package-based training, replace every example profile with your actual
profile ID, map every assigned slot, and set `package_contract` to a mapping
containing your package's `package_sha256` and the SHA256 of its `norm.json`.
These values must match the selected package; leaving the contract null is
rejected by package validation. Configure statistics groups for your package
when needed. Keep your populated registry and data manifest outside this source
repository.

The optional seven-camera recipe in `multicamera_multiview.py` uses the explicit
source label `example_multicamera`; configure it and the camera keys for your
own package. The RH20T multiview adapter uses source label `rh20t`.
