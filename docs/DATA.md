# Getting the training data (AMASS)

Qudmi trains on [AMASS](https://amass.is.tue.mpg.de/), a unified motion-capture dataset in the
SMPL body format. It's free for non-commercial/research use, but requires creating a personal
account and accepting their license — that's a manual step only you can do (Claude can't create
accounts or accept license terms on your behalf).

## Steps

1. Go to https://amass.is.tue.mpg.de/ and register for a free account.
2. Accept the license terms for the subsets you want. To start, just download one small subset —
   **ACCAD** or **HumanEva** are good first picks (small download, enough to validate the
   pipeline end-to-end before committing to the full ~20GB+ dataset).
3. Download the SMPL+H (or SMPL-X, if you prefer) body-model-parameterized `.npz` files for that
   subset.
4. Place the extracted files under `data/amass/<subset_name>/...` at the repo root (this `data/`
   directory is gitignored — motion-capture data is not meant to be committed to the repo).
5. You'll also need the SMPL+H body model files themselves (joint regressor, template mesh) from
   https://mano.is.tue.mpg.de/ (same account works) — place under `data/body_models/smplh/`.

Once one small subset is downloaded, the preprocessing script (`src/qudmi/data/`, coming in a
later step) can run against it — no need to download everything up front.

## Why not automate this download

AMASS explicitly requires an authenticated, license-accepting human to download its data — there
is no anonymous/API download path, by design (it protects the original mocap contributors'
licensing terms). Any tool that tried to script around that would be circumventing the license,
so this step stays manual.
