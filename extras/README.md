# Optional Ultralytics adapter (not used by `monitor.py`)

`hailo10h_ultralytics_patch.py` is the historical monkey patch from this
project. It redirects Ultralytics' Hailo inference backend to HailoRT 5.4
`InferModel` and optimizes the preprocessing/buffer reuse path. It was designed
around Ultralytics **8.4.142**; newer versions may change internals.

**The monitor uses HailoRT directly and does not import this patch or
Ultralytics.** The adapter is provided separately for developers who want to
run other applications through the Ultralytics `YOLO(...).predict(...)` API.
If you choose to use it, install compatible Ultralytics and its dependencies,
then import the patch before creating a `YOLO` instance. It has not been
hardware-regression-tested as part of this repository's release.

Note: `set_batch_size(1)` in both implementations denotes a single real image;
it does **not** imply dummy-image batching.

Ultralytics has its own licensing terms (AGPL-3.0 / enterprise options). These
apply to Ultralytics code and model artifacts independently of this project's
MIT license. See https://www.ultralytics.com/license.