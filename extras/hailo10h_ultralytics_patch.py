from pathlib import Path

import numpy as np

from ultralytics.engine.predictor import BasePredictor
from ultralytics.nn.backends.hailo import HailoBackend
from ultralytics.utils import LOGGER


# ============================================================
# Save original Ultralytics preprocess
#
# Non-Hailo models and unusual tensor inputs can fall back
# to the official preprocessing implementation.
# ============================================================

_ORIGINAL_PREPROCESS = BasePredictor.preprocess


# ============================================================
# Fast Hailo preprocessing
#
# Original Ultralytics path:
#
# BGR uint8 NumPy
#     ↓
# Torch tensor
#     ↓
# BCHW
#     ↓
# RGB
#     ↓
# float32
#     ↓
# /255
#
# Then the old Hailo patch did:
#
# float32
#     ↓
# NHWC
#     ↓
# NumPy
#     ↓
# ×255
#     ↓
# uint8
#
# New path:
#
# BGR uint8
#     ↓
# official Ultralytics letterbox
#     ↓
# RGB NCHW uint8 NumPy
#     ↓
# Hailo backend
#
# No Torch float input stage.
# ============================================================

def h10_fast_preprocess(self, im):

    # --------------------------------------------------------
    # Only change Hailo models.
    # Other Ultralytics backends keep official behavior.
    # --------------------------------------------------------

    if (
        getattr(
            getattr(self, "model", None),
            "format",
            None,
        )
        != "hailo"
    ):
        return _ORIGINAL_PREPROCESS(
            self,
            im,
        )

    # --------------------------------------------------------
    # If somebody explicitly passes a Torch tensor,
    # keep official Ultralytics behavior.
    # Our camera pipeline passes NumPy frames.
    # --------------------------------------------------------

    try:
        import torch

        if isinstance(im, torch.Tensor):
            return _ORIGINAL_PREPROCESS(
                self,
                im,
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # Normal camera path should be:
    #
    # list[np.ndarray]
    # --------------------------------------------------------

    if (
        not isinstance(
            im,
            (list, tuple),
        )
        or not im
    ):
        return _ORIGINAL_PREPROCESS(
            self,
            im,
        )

    # --------------------------------------------------------
    # Keep official Ultralytics geometric preprocessing.
    #
    # This performs the normal resize / letterbox.
    # --------------------------------------------------------

    transformed = self.pre_transform(
        im
    )

    # --------------------------------------------------------
    # Single-image path
    #
    # BGR HWC uint8
    #     ↓
    # RGB CHW uint8
    #
    # One contiguous copy is still required here.
    # --------------------------------------------------------

    if len(transformed) == 1:

        image = transformed[0]

        if not isinstance(
            image,
            np.ndarray,
        ):
            return _ORIGINAL_PREPROCESS(
                self,
                im,
            )

        image = np.ascontiguousarray(
            image[
                ...,
                ::-1
            ].transpose(
                2,
                0,
                1,
            ),
            dtype=np.uint8,
        )

        batch = image[
            None,
            ...
        ]

    # --------------------------------------------------------
    # Batch compatibility
    # --------------------------------------------------------

    else:

        batch = np.stack(
            [
                np.ascontiguousarray(
                    image[
                        ...,
                        ::-1
                    ].transpose(
                        2,
                        0,
                        1,
                    ),
                    dtype=np.uint8,
                )
                for image
                in transformed
            ],
            axis=0,
        )

    # --------------------------------------------------------
    # Print once so we know optimization is active.
    # --------------------------------------------------------

    if not getattr(
        self,
        "_h10_fast_preprocess_logged",
        False,
    ):

        LOGGER.info(
            "Hailo fast preprocess enabled: "
            "BGR uint8 -> RGB NCHW uint8 "
            "(Torch float input stage bypassed)."
        )

        self._h10_fast_preprocess_logged = (
            True
        )

    return batch


# ============================================================
# Hailo-10H model loading
# ============================================================

def h10_load_model(
    self,
    weight,
):
    """
    HailoRT 5.4 InferModel implementation for Hailo-10H.

    Features:
    - InferModel API
    - SHARED VDevice
    - ROUND_ROBIN scheduler
    - reusable output buffers
    - reusable bindings
    """

    from hailo_platform import (
        FormatType,
        HailoSchedulingAlgorithm,
        VDevice,
    )

    w = Path(weight)

    hef_file = next(
        w.rglob(
            "*.hef"
        ),
        None,
    )

    if (
        hef_file is None
        or not hef_file.is_file()
    ):

        raise FileNotFoundError(
            f"No .hef file found in: {w}"
        )

    LOGGER.info(
        f"Loading {hef_file} "
        "for Hailo-10H InferModel inference..."
    )

    # ========================================================
    # Ultralytics metadata
    # ========================================================

    self.apply_metadata(
        self.read_metadata(
            hef_file
        )
    )

    if (
        self.task
        and self.task
        not in {
            "detect",
            "segment",
            "pose",
            "obb",
            "classify",
            "semantic",
            "depth",
        }
    ):

        raise ValueError(
            f"Unsupported Hailo task: "
            f"{self.task}"
        )

    # ========================================================
    # Shared Hailo-10H VDevice
    # ========================================================

    params = (
        VDevice.create_params()
    )

    params.group_id = (
        "SHARED"
    )

    params.scheduling_algorithm = (
        HailoSchedulingAlgorithm
        .ROUND_ROBIN
    )

    self._h10_vdevice = (
        VDevice(
            params
        )
    )

    # ========================================================
    # InferModel
    # ========================================================

    self.model = (
        self._h10_vdevice
        .create_infer_model(
            str(
                hef_file
            )
        )
    )

    # Current monitoring pipeline is batch=1.
    self.model.set_batch_size(
        1
    )

    # ========================================================
    # Input
    # ========================================================

    self.input_info = (
        self.model.input()
    )

    self.input_info.set_format_type(
        FormatType.UINT8
    )

    # ========================================================
    # Outputs
    #
    # Keep FLOAT32 because official Ultralytics YOLO26
    # decoder expects dequantized floating point outputs.
    # ========================================================

    self.output_infos = list(
        self.model.outputs
    )

    for output in (
        self.output_infos
    ):

        output.set_format_type(
            FormatType.FLOAT32
        )

    # ========================================================
    # Configure model
    # ========================================================

    self._h10_config_context = (
        self.model.configure()
    )

    self._h10_configured_model = (
        self._h10_config_context
        .__enter__()
    )

    # ========================================================
    # Allocate output buffers ONCE
    #
    # Old code allocated np.empty() every frame.
    # ========================================================

    self._h10_output_buffers = {}

    for output in (
        self.output_infos
    ):

        self._h10_output_buffers[
            output.name
        ] = np.empty(
            tuple(
                output.shape
            ),
            dtype=np.float32,
        )

    # ========================================================
    # Create bindings ONCE
    #
    # Old code called create_bindings() every inference.
    # ========================================================

    self._h10_bindings = (
        self._h10_configured_model
        .create_bindings(
            output_buffers=
            self._h10_output_buffers
        )
    )

    # ========================================================
    # Preserve official Ultralytics decoder state
    # ========================================================

    self._anchors = None

    if self.task in {
        "segment",
        "pose",
        "obb",
    }:

        from ultralytics.nn.modules import (
            DFL
        )

        self._dfl = DFL()

    self.end2end = (
        self.task
        not in {
            "segment",
            "pose",
            "obb",
        }
    )

    LOGGER.info(
        "Hailo-10H InferModel configured successfully "
        "(SHARED + ROUND_ROBIN + reusable buffers)."
    )


# ============================================================
# Convert inference input to Hailo NHWC uint8
# ============================================================

def _h10_to_nhwc_uint8(
    im,
):

    # ========================================================
    # FAST PATH
    #
    # Input produced by h10_fast_preprocess:
    #
    # RGB
    # NCHW
    # uint8
    # NumPy
    # ========================================================

    if isinstance(
        im,
        np.ndarray,
    ):

        images = im

        if images.ndim == 3:

            images = images[
                None,
                ...
            ]

        if images.ndim != 4:

            raise ValueError(
                "Unexpected Hailo NumPy "
                f"input shape: {images.shape}"
            )

        # ----------------------------------------------------
        # NCHW -> NHWC
        # ----------------------------------------------------

        if images.shape[1] == 3:

            images = (
                np.ascontiguousarray(
                    images.transpose(
                        0,
                        2,
                        3,
                        1,
                    ),
                    dtype=np.uint8,
                )
            )

        # ----------------------------------------------------
        # Already NHWC
        # ----------------------------------------------------

        elif images.shape[-1] == 3:

            images = (
                np.ascontiguousarray(
                    images,
                    dtype=np.uint8,
                )
            )

        else:

            raise ValueError(
                "Cannot determine channel "
                "dimension for Hailo input: "
                f"{images.shape}"
            )

        return images

    # ========================================================
    # COMPATIBILITY PATH
    #
    # If some other code still feeds normal Ultralytics
    # Torch BCHW float [0,1], keep old behavior.
    # ========================================================

    images = (
        im.permute(
            0,
            2,
            3,
            1,
        )
        .cpu()
        .numpy()
    )

    # Already uint8 tensor
    if (
        images.dtype
        == np.uint8
    ):

        return (
            np.ascontiguousarray(
                images
            )
        )

    # float [0,1]
    return (
        np.ascontiguousarray(
            np.clip(
                images * 255.0,
                0,
                255,
            ).astype(
                np.uint8
            )
        )
    )


# ============================================================
# Hailo-10H forward
# ============================================================

def h10_forward(
    self,
    im,
):
    """
    Optimized Hailo-10H forward.

    Normal project path:

    NumPy RGB NCHW uint8
        ↓
    NumPy RGB NHWC uint8
        ↓
    Hailo-10H

    No Torch float input round-trip.
    """

    images = (
        _h10_to_nhwc_uint8(
            im
        )
    )

    batch_size = (
        images.shape[0]
    )

    # ========================================================
    # FAST batch=1 path
    #
    # Our surveillance project always uses batch=1.
    #
    # Reuses:
    # - Hailo bindings
    # - output buffers
    #
    # Eliminates:
    # - np.empty each frame
    # - create_bindings each frame
    # - buffer.copy
    # - np.stack
    # ========================================================

    if batch_size == 1:

        image = images[0]

        self._h10_bindings.input().set_buffer(
            image
        )

        # ----------------------------------------------------
        # Actual Hailo-10H inference
        # ----------------------------------------------------

        self._h10_configured_model.run(
            [
                self._h10_bindings
            ],
            10000,
        )

        # ----------------------------------------------------
        # Add batch dimension as a VIEW.
        #
        # No copy.
        # ----------------------------------------------------

        outputs = [
            self._h10_output_buffers[
                output.name
            ][
                None,
                ...
            ]
            for output
            in self.output_infos
        ]

    # ========================================================
    # Batch > 1 compatibility path
    # ========================================================

    else:

        collected_outputs = [
            []
            for _
            in self.output_infos
        ]

        for image in images:

            self._h10_bindings.input().set_buffer(
                image
            )

            self._h10_configured_model.run(
                [
                    self._h10_bindings
                ],
                10000,
            )

            # Must copy here because next image overwrites
            # the same reusable Hailo buffers.
            for (
                index,
                output,
            ) in enumerate(
                self.output_infos
            ):

                collected_outputs[
                    index
                ].append(
                    self._h10_output_buffers[
                        output.name
                    ].copy()
                )

        outputs = [
            np.stack(
                output_batch,
                axis=0,
            )
            for output_batch
            in collected_outputs
        ]

    # ========================================================
    # Official Ultralytics task decoders
    # ========================================================

    if (
        self.task
        == "segment"
    ):

        return (
            self._decode_segment(
                outputs
            )
        )

    if (
        self.task
        == "pose"
    ):

        return (
            self._decode_pose(
                outputs
            )
        )

    if (
        self.task
        == "obb"
    ):

        return (
            self._decode_obb(
                outputs
            )
        )

    if (
        self.task
        == "classify"
    ):

        import torch

        return (
            torch.from_numpy(
                outputs[0]
            )
            .reshape(
                outputs[0].shape[0],
                -1,
            )
        )

    if (
        self.task
        == "semantic"
    ):

        import torch

        out = (
            torch.from_numpy(
                outputs[0]
            )
        )

        if self.metadata.get(
            "semantic_baked"
        ):

            return out.reshape(
                out.shape[0],
                out.shape[1],
                out.shape[2],
            )

        return out.permute(
            0,
            3,
            1,
            2,
        )

    if (
        self.task
        == "depth"
    ):

        return (
            self._decode_depth(
                outputs[0]
            )
        )

    # ========================================================
    # YOLO26 Detect
    #
    # Continue using official Ultralytics YOLO26 decoder.
    # ========================================================

    if not self.metadata.get(
        "nms",
        False,
    ):

        return (
            self._decode_raw(
                outputs
            )
        )

    return (
        self._decode_nms(
            outputs[0]
        )
    )


# ============================================================
# Cleanup
# ============================================================

def h10_del(
    self,
):
    """
    Release Hailo-10H resources cleanly.
    """

    # --------------------------------------------------------
    # Drop binding / NumPy buffer references
    # --------------------------------------------------------

    try:

        self._h10_bindings = None
        self._h10_output_buffers = None

    except Exception:
        pass

    # --------------------------------------------------------
    # Close configured InferModel
    # --------------------------------------------------------

    context = getattr(
        self,
        "_h10_config_context",
        None,
    )

    if context is not None:

        try:

            context.__exit__(
                None,
                None,
                None,
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # Release VDevice
    # --------------------------------------------------------

    device = getattr(
        self,
        "_h10_vdevice",
        None,
    )

    if device is not None:

        try:

            device.release()

        except Exception:
            pass


# ============================================================
# Monkey patches
# ============================================================

# Ultralytics input pipeline:
#
# BGR NumPy uint8
# ->
# letterbox
# ->
# RGB NumPy uint8
#
# instead of Torch float.
BasePredictor.preprocess = (
    h10_fast_preprocess
)

# HailoRT 5.4 / Hailo-10H backend.
HailoBackend.load_model = (
    h10_load_model
)

HailoBackend.forward = (
    h10_forward
)

HailoBackend.__del__ = (
    h10_del
)


print(
    "[Patch] Ultralytics Hailo backend patched for Hailo-10H: "
    "fast NumPy uint8 preprocess + reusable buffers/bindings + "
    "SHARED ROUND_ROBIN."
)