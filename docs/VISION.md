# Vision

## Mission

Make believable full-body motion from sparse tracking a free, open, drop-in commodity — not
something that requires extra hardware, an in-house ML team, or a licensing fee. Any developer,
solo indie or small studio, should be able to give their users a full body from just a headset
and two controllers, with one component and zero ML code — and the entire model, training
pipeline, and weights stay open so anyone can inspect, retrain, or extend it.

## Why this isn't "just another VR plugin"

Sparse-to-full-body motion estimation isn't novel research — AvatarPoser, AGRoL, and Meta's
"Generative Legs" work already solve versions of this problem. If Qudmi were only a packaged
re-implementation of that idea for Unity, its value would be real but narrow: free, open, and
actually easy to integrate, for an audience (indie/small VR studios) that today either skips
full-body avatars entirely, buys $100-300/unit extra trackers, or would need to hire an ML
engineer to implement a research paper themselves. That's a legitimate niche to serve — but it's
not infrastructure other things get built on.

Three things would move it from "useful tool" to "hard to replace":

1. **The underlying capability is bigger than VR.** "Reconstruct full-body human motion from
   whatever sparse points you happen to have" also applies to robotics teleoperation (driving a
   robot from an operator's tracked hands), budget motion capture for indie film/game studios,
   remote physical-therapy/rehab monitoring from a phone + watch, and sports form-checking. The
   data pipeline (`qudmi.data.amass.process_sequence`) takes `tracker_joints` as a parameter
   rather than hardcoding "head + 2 hands" specifically, so new sparse-input configurations for
   these domains are a preprocessing + retrain, not a rewrite.
2. **Publish it as a reference point, not just a repo.** A short technical report/benchmark
   (MPJPE, jitter, foot-skate on a standardized AMASS test split) that other projects measure
   themselves against is what makes something the thing a field points to, rather than one
   option among many nobody found. Not yet done — tracked in docs/ROADMAP.md.
3. **Publish the interface, not just the weights.** The sparse-in/full-body-out format
   (docs/SPEC.md) is currently an implementation detail. Documenting it as an open, stable
   protocol lets other researchers train competing or better models against the same interface
   while Qudmi stays relevant as the reference implementation — the ONNX/USB-C playbook, not the
   "one clever model" playbook.

VR is the first proven case, not the whole scope. Ship that well first; keep the architecture
from foreclosing the rest.

## Does using the SDK improve the model? (Not by default.)

The SDK runs a static, already-trained model entirely on-device (ONNX Runtime, no network
calls) — deliberately, for privacy. That means integrating it does **not**, by itself, feed
future training. There's also a structural reason passive usage can't work as a training signal:
a real app only ever has the sparse tracker input; the true full-body pose is exactly what the
model is trying to predict, so there's no ground truth sitting there to learn from.

A genuine data flywheel is possible, but has to be built deliberately, opt-in only, and with
real care given that motion/gait data is biometric-identifying information:

- **Tier 1 (weaker signal, easy)**: users with only headset+hands opt in to share raw sparse
  input sequences. No ground truth, but real-world input diversity (vs. AMASS's studio mocap)
  is valuable for future self-supervised domain adaptation.
- **Tier 2 (real supervised signal)**: users who *also* own extra body trackers (already common
  in the existing VR full-body-tracking community) opt in to contribute genuine
  `(sparse input, true full-body pose)` pairs — directly useful training data, and unlike Tier 1,
  a real substitute for more motion-capture data collection. Analogous to how sensor-rich
  vehicles generate labels that help sensor-sparse ones in autonomous driving fleets.

Not built yet. If pursued, it's a separate project from the SDK itself (needs a backend,
explicit consent flow, anonymization, and data governance) — noted here so the direction isn't
lost, not because it's scoped for now.
