# monailabel-sam-runtime

Bundled inference code and YAML configurations from [MedSAM2](https://github.com/bowang-lab/MedSAM2/tree/332f30d420f1d1b08e2a79b3ae6a602458808383), using the `sam2` and `efficient_track_anything` namespaces. Included with `monailabel`.

Unmodified upstream files; [upstream.json](upstream.json) records the revision and file hashes. See [LICENSE](LICENSE) (Apache-2.0) and [NOTICE](NOTICE).

Use a fresh environment when upgrading: separate `sam2`, `MedSAM2` and EfficientTAM installs can own the same files. Weights, training applications and native extensions are excluded; adapters disable hole/sprinkle postprocessing.

To update, copy Python/YAML runtime files from a reviewed commit, refresh the manifest and licenses, and run package and SAM inference tests. Do not reformat upstream files.
