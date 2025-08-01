"""
This module provides a ComfyUI custom node to align two images based on the
dominant orientation of their edge gradients.  The implementation is a
translation of a Julia Pluto notebook into pure Python and uses OpenCV
and Pillow to perform image manipulation.  It takes two images and two masks
as inputs, calculates the relative rotation between them by comparing
histograms of gradient directions, rotates the first image and its mask to
match the orientation of the second, rescales it to fit the target mask’s
bounding box and composites it into the second image.  The resulting
image, the original target mask and the rotation angle are returned.

Inputs
======
```
image_a: IMAGE     – The source image which will be rotated and placed onto the target.
mask_a:  MASK      – Binary mask for the source image. Non‑zero pixels indicate the
                     part of the source image that should be considered for rotation
                     and placement.
image_b: IMAGE     – The target image into which the rotated source will be placed.
mask_b:  MASK      – Binary mask for the target image. Non‑zero pixels indicate the
                     area where the source image should be inserted.
delta_h_exp: INT   – Exponent used to derive the histogram bin size.  The actual
                     bin size ``delta_h`` is computed as ``10**(-delta_h_exp)``.
                     Setting ``delta_h_exp`` to 0 yields a bin size of 1.0°,
                     while a value of 4 yields 0.0001°.  This slider steps
                     through discrete values [0,1,2,3,4].
grad_threshold: FLOAT – Threshold on the relative gradient magnitude used to ignore
                     weak edges when computing the histogram.  Values are
                     selected linearly between 0.04 and 0.10 in 0.01 steps.
```

Outputs
=======
```
IMAGE – The composited image with the rotated source inserted into the target.
MASK  – The original target mask (unchanged).  Returned for convenience so
         subsequent nodes can continue to use it without having to branch.
FLOAT – The rotation angle (in degrees) that was applied to the source
         image.  This is useful for debugging or for chaining multiple
         alignment operations.
```

The node operates on batched inputs.  Each element of the batch is processed
independently: the gradient histogram is computed, the optimal circular
shift between histograms is determined, and the corresponding rotation is
applied.  Cropping and resizing are driven purely by the masks supplied
as input and no attempt is made to guess the object boundary from the
image data itself.

Note: This code expects that `torch` and `cv2` (OpenCV) are available in
the runtime environment.  ComfyUI ships with both, but if OpenCV is
missing the node will raise an exception at runtime.
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Any, List

import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None  # will error at runtime if unavailable

try:
    import torch
except Exception as e:
    # Torch is required by ComfyUI, but we import lazily to allow module
    # introspection without immediately loading torch in environments where
    # it isn't present.  An informative error will be raised if the
    # processing function is called without torch.
    torch = None  # type: ignore


def _gradient_histogram(
    image: np.ndarray,
    mask: np.ndarray,
    delta: float,
    threshold: float,
) -> np.ndarray:
    """
    Compute a normalized histogram of gradient directions for the given image.

    Parameters
    ----------
    image : np.ndarray
        An RGB image with shape (H, W, 3).  Expected to be of dtype uint8 or
        float in [0, 255] range.  Values outside this range are clipped.
    mask : np.ndarray
        A binary mask with shape (H, W).  Only pixels where mask is non‑zero
        contribute to the histogram.  Mask values are ignored when computing
        gradients – the full image is used for gradient computation and the
        mask restricts which pixels contribute to the histogram.
    delta : float
        The bin size of the histogram in degrees.  For example, delta=0.1
        yields 360/0.1 = 3600 bins.
    threshold : float
        Relative threshold on the gradient magnitude used to filter out
        insignificant gradients.  The magnitude at each pixel is divided by
        the global maximum magnitude before comparison.

    Returns
    -------
    hist : np.ndarray
        A one‑dimensional array of length `int(360 / delta)` containing the
        normalized histogram (summing to 1).  If no valid gradient pixels are
        found, a uniform histogram is returned.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for gradient computation but was not available."
        )

    # Ensure image is float32 in [0, 1]
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # Convert to grayscale by summing channels (similar to the Julia function zahl)
    # Equivalent to red + green + blue, which yields a reasonable edge image.
    # Alternatively one could use cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).
    gray = img[..., 0] + img[..., 1] + img[..., 2]

    # Compute horizontal and vertical gradients using Sobel operator.
    # OpenCV's Sobel returns gradients scaled by 8; the absolute scale is unimportant
    # since we normalize by the maximum magnitude below.
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)

    # Compute gradient magnitude and direction (0–360 degrees)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    # Use arctan2 directly instead of cv2.phase to avoid dependency on cv2.cuda
    angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

    # Normalize magnitude and apply threshold relative to max
    max_mag = float(np.max(mag)) if np.max(mag) > 0 else 1.0
    mag_norm = mag / max_mag

    # Build mask of pixels contributing to histogram: mask and magnitude threshold
    contrib = (mask > 0) & (mag_norm > threshold)
    angles_valid = angles[contrib]

    bins = int(360.0 / delta)
    if bins <= 0:
        raise ValueError(f"delta must divide 360 into a positive number of bins, got {delta}")

    # If no valid pixels, return a uniform distribution to avoid division by zero later
    if angles_valid.size == 0:
        return np.ones(bins, dtype=np.float32) / bins

    hist, _ = np.histogram(
        angles_valid,
        bins=bins,
        range=(0.0, 360.0),
        density=False,
    )
    hist = hist.astype(np.float32)
    hist_sum = float(hist.sum())
    if hist_sum > 0:
        hist /= hist_sum
    else:
        hist = np.ones_like(hist) / bins
    return hist


def _find_optimal_shift(hist_a: np.ndarray, hist_b: np.ndarray) -> int:
    """
    Find the circular shift of hist_b which best aligns it to hist_a.

    The optimal shift minimizes the L1 distance between hist_a and the rolled
    version of hist_b.  Returns the index of the shift (an integer in
    [0, len(hist_a) - 1]).
    """
    # Compute L1 distances for all possible circular shifts
    length = hist_a.shape[0]
    # Precompute the FFT of histograms to speed up correlation when bins are large
    # However, for clarity and because bins are moderate (e.g. 3600 for delta=0.1),
    # a simple Python loop is sufficient and mirrors the Julia implementation.
    min_err = None
    min_shift = 0
    for shift in range(length):
        # np.roll performs a circular shift by 'shift' positions
        shifted = np.roll(hist_b, shift)
        err = float(np.sum(np.abs(hist_a - shifted)))
        if min_err is None or err < min_err:
            min_err = err
            min_shift = shift
    return min_shift


def _rotate_and_compose(
    image_a: np.ndarray,
    mask_a: np.ndarray,
    image_b: np.ndarray,
    mask_b: np.ndarray,
    angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image_a (and its mask) by `angle` degrees, crop to the rotated
    mask’s bounding box, resize to match the bounding box of mask_b, and
    composite into image_b.  The returned image is multiplied by mask_b to
    zero out any pixels outside the target mask.  The returned mask is
    simply mask_b for convenience.

    Parameters
    ----------
    image_a : np.ndarray
        Source RGB image with shape (H, W, 3), dtype float32 in [0, 1] or uint8.
    mask_a : np.ndarray
        Binary mask for image_a with shape (H, W).  Values >0 indicate the
        region of interest.
    image_b : np.ndarray
        Target RGB image into which the rotated image_a will be inserted.
    mask_b : np.ndarray
        Binary mask for image_b indicating where the insertion should occur.
    angle : float
        Rotation angle in degrees.  Positive values rotate counter‑clockwise.

    Returns
    -------
    composed : np.ndarray
        The resulting composited image, same shape and dtype as image_b.
    mask_b : np.ndarray
        The original target mask, returned unchanged.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for rotation and resizing but was not available."
        )
    # Convert images to float32 in [0,1] for blending
    def to_float(img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32)
        if out.max() > 1.0:
            out /= 255.0
        return out
    img_a = to_float(image_a)
    img_b = to_float(image_b)
    msk_a = (mask_a > 0).astype(np.uint8)
    msk_b = (mask_b > 0).astype(np.uint8)

    # Rotate image and mask around their center and expand to fit
    h_a, w_a = img_a.shape[:2]
    center_a = (w_a / 2.0, h_a / 2.0)
    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center_a, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    # Compute new bounding dimensions
    new_w = int((h_a * sin) + (w_a * cos))
    new_h = int((h_a * cos) + (w_a * sin))
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2.0) - center_a[0]
    M[1, 2] += (new_h / 2.0) - center_a[1]

    # Perform the rotation for image and mask
    rotated_a = cv2.warpAffine(
        img_a,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )
    rotated_mask_a = cv2.warpAffine(
        msk_a,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Find bounding box of rotated mask (non‑zero region)
    ys, xs = np.where(rotated_mask_a > 0)
    if ys.size == 0 or xs.size == 0:
        # Nothing to insert, return original B and mask
        return img_b, msk_b
    top_a = int(ys.min())
    bottom_a = int(ys.max())
    left_a = int(xs.min())
    right_a = int(xs.max())
    # Crop rotated image to the bounding box
    cropped_a = rotated_a[top_a : bottom_a + 1, left_a : right_a + 1]

    # Find bounding box of target mask
    ys_b, xs_b = np.where(msk_b > 0)
    if ys_b.size == 0 or xs_b.size == 0:
        # No place to insert, return original B and mask
        return img_b, msk_b
    top_b = int(ys_b.min())
    bottom_b = int(ys_b.max())
    left_b = int(xs_b.min())
    right_b = int(xs_b.max())
    target_h = bottom_b - top_b + 1
    target_w = right_b - left_b + 1

    # Resize cropped source image to fit the target bounding box
    # cv2.resize expects size in (width, height)
    resized_a = cv2.resize(
        cropped_a,
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC,
    )

    # Compose onto a copy of image_b
    composed = img_b.copy()
    composed[top_b : bottom_b + 1, left_b : right_b + 1] = resized_a
    # Multiply by target mask to zero out any pixels outside of mask_b
    composed = composed * msk_b[..., None].astype(np.float32)
    return composed, msk_b


class AlignImagesNode:
    """Align two images by matching their dominant orientation using edge histograms."""

    # The category in the ComfyUI node selection menu
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define the input types for the node.  The keys of the returned
        dictionary correspond to argument names of the processing function.

        The `delta_h_exp` slider controls the histogram bin size indirectly.
        The actual bin size used is `10 ** (-delta_h_exp)`, allowing the
        histogram resolution to vary across the range 1.0 → 0.0001.  The
        exponent is constrained to integer values between 0 and 4 so the
        resulting bin sizes are [1, 0.1, 0.01, 0.001, 0.0001].

        The `grad_threshold` slider selects a relative gradient magnitude
        threshold between 0.04 and 0.1 in steps of 0.01.
        """
        return {
            "required": {
                "image_a": ("IMAGE",),
                "mask_a": ("MASK",),
                "image_b": ("IMAGE",),
                "mask_b": ("MASK",),
                # Exponent for the histogram bin size.  The actual delta used
                # is 10**(-delta_h_exp), giving values from 1 down to 0.0001.
                "delta_h_exp": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 4,
                        "step": 1,
                        "display": "slider",
                    },
                ),
                # Gradient magnitude threshold (relative) between 0.04 and 0.1
                "grad_threshold": (
                    "FLOAT",
                    {
                        "default": 0.04,
                        "min": 0.04,
                        "max": 0.1,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            }
        }

    # We return an image, a mask and a float (rotation angle)
    RETURN_TYPES: Tuple[str, ...] = ("IMAGE", "MASK", "FLOAT")

    # The name of the method that will be called when executing the node
    FUNCTION: str = "align_images"

    @staticmethod
    def align_images(
        image_a: "torch.Tensor",
        mask_a: "torch.Tensor",
        image_b: "torch.Tensor",
        mask_b: "torch.Tensor",
        delta_h_exp: int = 1,
        grad_threshold: float = 0.04,
    ) -> Tuple["torch.Tensor", "torch.Tensor", Any]:
        """
        Main processing function executed by ComfyUI.  It handles batches of
        images and masks, computing the rotation for each pair independently.

        Parameters
        ----------
        image_a : torch.Tensor
            A batch of source images with shape ``[B, H, W, C]`` where ``C=3``.
        mask_a : torch.Tensor
            A batch of source masks with shape ``[B, H, W]``.  Values should lie
            in ``[0, 1]``.
        image_b : torch.Tensor
            A batch of target images with shape ``[B, H, W, C]``.
        mask_b : torch.Tensor
            A batch of target masks with shape ``[B, H, W]``.
        delta_h_exp : int
            Exponent used to derive the histogram bin size.  The actual bin
            size ``delta_h`` is computed as ``10 ** (-delta_h_exp)``.  When
            ``delta_h_exp = 0``, ``delta_h = 1.0``; when ``delta_h_exp = 4``,
            ``delta_h = 0.0001``.
        grad_threshold : float
            Relative gradient magnitude threshold for histogram construction.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, List[float]]
            A tuple containing the aligned images, the unchanged target masks,
            and a list (wrapped in a tensor) of the applied rotation angles.
        """
        # Validate torch is available
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for this custom node but was not imported."
            )
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required for this custom node but was not imported."
            )

        # The actual histogram bin size is 10 ** (-delta_h_exp)
        # Constrain exponent to avoid invalid values
        if delta_h_exp < 0:
            delta_h_exp = 0
        delta_h = 10.0 ** float(-delta_h_exp)

        # Ensure batch dimensions match
        B = image_a.shape[0]
        if (mask_a.shape[0] != B) or (image_b.shape[0] != B) or (mask_b.shape[0] != B):
            raise ValueError("All inputs must have the same batch size")

        aligned_images: List["torch.Tensor"] = []
        out_masks: List["torch.Tensor"] = []
        angles: List[float] = []

        # Iterate over the batch dimension
        for idx in range(B):
            # Convert tensors to numpy on CPU for processing
            img_a_np = image_a[idx].detach().cpu().numpy()
            mask_a_np = mask_a[idx].detach().cpu().numpy()
            img_b_np = image_b[idx].detach().cpu().numpy()
            mask_b_np = mask_b[idx].detach().cpu().numpy()

            # Compute histograms for both images using their masks.  The
            # histogram bin size is derived from the exponent: delta_h = 10**(-exp)
            hist_a = _gradient_histogram(img_a_np, mask_a_np, delta_h, grad_threshold)
            hist_b = _gradient_histogram(img_b_np, mask_b_np, delta_h, grad_threshold)
            # Find the shift which best aligns B to A
            shift_index = _find_optimal_shift(hist_a, hist_b)
            angle = shift_index * delta_h
            # Rotate and compose the images using the computed angle
            composed_np, new_mask_np = _rotate_and_compose(
                img_a_np, mask_a_np, img_b_np, mask_b_np, angle
            )
            # Convert back to torch tensors
            composed_t = torch.from_numpy(composed_np).to(image_a.dtype)
            # When converting masks we preserve the original dtype (float or bool)
            new_mask_t = torch.from_numpy(new_mask_np).to(mask_b.dtype)
            aligned_images.append(composed_t)
            out_masks.append(new_mask_t)
            angles.append(float(angle))

        # Stack the results back into batch form
        out_images = torch.stack(aligned_images, dim=0)
        out_masks_tensor = torch.stack(out_masks, dim=0)
        # Return angles as a torch tensor for consistency
        angles_tensor = torch.tensor(angles, dtype=torch.float32)
        return out_images, out_masks_tensor, angles_tensor


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "AlignImagesNode": AlignImagesNode,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "AlignImagesNode": "Align Images (Pluto)",
}

##########################################################
# Auto‑refining version of the alignment node
#
# This class implements a multi‑resolution search for the relative
# rotation angle.  It starts with a coarse histogram bin size (1°)
# and iteratively refines the bin size by a factor of ten until
# it reaches a resolution of 0.0001°.  At each refinement step,
# instead of evaluating every possible shift, it searches only
# within a window around the best shift found at the previous
# resolution.  This drastically reduces computation time while
# still converging on a very precise angle.

def _find_angle_auto_refine(
    img_a: np.ndarray,
    mask_a: np.ndarray,
    img_b: np.ndarray,
    mask_b: np.ndarray,
    grad_threshold: float,
    exponents: List[int] | Tuple[int, ...] = (0, 1, 2, 3, 4),
    window_bins: int = 3,
    search_min_deg: float | None = None,
    search_max_deg: float | None = None,
) -> float:
    """
    Determine the rotation angle needed to align the orientation histograms
    of ``img_a`` and ``img_b`` using a progressive, multi‑resolution search.

    Parameters
    ----------
    img_a, mask_a, img_b, mask_b : np.ndarray
        Arrays representing the source/target images and masks.
    grad_threshold : float
        Relative gradient magnitude threshold for histogram construction.
    exponents : sequence of int, optional
        A sequence of histogram exponents to use.  Each exponent ``e``
        corresponds to a bin size ``delta = 10**(-e)``.  The search
        proceeds through each exponent in order.  The default
        ``(0,1,2,3,4)`` yields resolutions 1°, 0.1°, 0.01°, 0.001° and
        0.0001°.
    window_bins : int, optional
        The number of coarse bins on either side of the current best
        estimate to search when refining.  For example, with
        ``window_bins=3`` the algorithm evaluates shifts across a window
        spanning ``±3`` bins at the coarse level, which expands to
        ``±30`` bins at the next finer level.

    Returns
    -------
    float
        The rotation angle (in degrees) that best aligns ``img_a`` to
        ``img_b`` according to the orientation histograms.
    """
    # Normalize search range
    if search_min_deg is None or search_max_deg is None:
        # Default to full 360° search
        search_min_deg = 0.0
        search_max_deg = 360.0
    # Convert min and max to floats
    search_min_deg = float(search_min_deg)
    search_max_deg = float(search_max_deg)

    # Initialize best shift and previous exponent
    best_shift = 0
    prev_exp = exponents[0]
    for idx, exp in enumerate(exponents):
        delta = 10.0 ** float(-exp)
        # Compute histograms at this resolution
        hist_a = _gradient_histogram(img_a, mask_a, delta, grad_threshold)
        hist_b = _gradient_histogram(img_b, mask_b, delta, grad_threshold)
        bins = len(hist_a)
        if idx == 0:
            # At the coarsest level, search across the user‑specified global range
            # Convert degree range into histogram index range
            # Normalize degrees into [0, 360)
            # Because Python's mod on negatives returns positive remainder
            min_deg = search_min_deg % 360.0
            max_deg = search_max_deg % 360.0
            if min_deg == max_deg:
                # full circle
                global_indices = list(range(bins))
            elif min_deg < max_deg:
                start_idx = int(math.floor(min_deg / delta))
                end_idx = int(math.ceil(max_deg / delta))
                # Clamp to [0, bins)
                start_idx = max(0, start_idx)
                end_idx = min(bins - 1, end_idx)
                global_indices = list(range(start_idx, end_idx + 1))
            else:
                # Wrap‑around interval
                start_idx1 = int(math.floor(min_deg / delta))
                end_idx1 = bins - 1
                start_idx2 = 0
                end_idx2 = int(math.ceil(max_deg / delta))
                # Clamp
                start_idx1 = max(0, start_idx1)
                end_idx1 = min(bins - 1, end_idx1)
                start_idx2 = max(0, start_idx2)
                end_idx2 = min(bins - 1, end_idx2)
                global_indices = list(range(start_idx1, end_idx1 + 1)) + list(range(start_idx2, end_idx2 + 1))
            # At the coarsest level, initial candidate shifts are exactly the global indices
            search_indices = global_indices
        else:
            # Determine how many fine bins correspond to one coarse bin
            # Example: going from exp=0 (delta=1) to exp=1 (delta=0.1)
            factor = int(round(10 ** (exp - prev_exp)))
            # Convert the previous best shift to the current resolution
            center = best_shift * factor
            # Build a list of candidate shifts within the search window
            # The search window width is scaled by factor
            offsets = range(-window_bins * factor, window_bins * factor + 1)
            local_candidates = [(center + off) % bins for off in offsets]
            # Compute the global search indices for this resolution
            min_deg = search_min_deg % 360.0
            max_deg = search_max_deg % 360.0
            if min_deg == max_deg:
                global_indices = list(range(bins))
            elif min_deg < max_deg:
                start_idx = int(math.floor(min_deg / delta))
                end_idx = int(math.ceil(max_deg / delta))
                start_idx = max(0, start_idx)
                end_idx = min(bins - 1, end_idx)
                global_indices = list(range(start_idx, end_idx + 1))
            else:
                start_idx1 = int(math.floor(min_deg / delta))
                end_idx1 = bins - 1
                start_idx2 = 0
                end_idx2 = int(math.ceil(max_deg / delta))
                start_idx1 = max(0, start_idx1)
                end_idx1 = min(bins - 1, end_idx1)
                start_idx2 = max(0, start_idx2)
                end_idx2 = min(bins - 1, end_idx2)
                global_indices = list(range(start_idx1, end_idx1 + 1)) + list(range(start_idx2, end_idx2 + 1))
            global_set = set(global_indices)
            # Intersection of local candidates with global search interval
            search_indices = [c for c in local_candidates if c in global_set]
        # Find the best shift within the candidate indices
        min_err = None
        min_idx = 0
        # Precompute a copy of hist_a to avoid reindexing inside the loop
        # We'll compare against shifted versions of hist_b
        for sidx in search_indices:
            shifted_b = np.roll(hist_b, sidx)
            err = float(np.sum(np.abs(hist_a - shifted_b)))
            if (min_err is None) or (err < min_err):
                min_err = err
                min_idx = sidx
        # Update best shift and previous exponent for the next iteration
        best_shift = min_idx
        prev_exp = exp
    # Compute the final delta corresponding to the finest exponent
    delta_final = 10.0 ** float(-exponents[-1])
    return best_shift * delta_final


class AlignImagesAutoRefineNode:
    """Align images using a multi‑resolution search to refine the rotation angle."""

    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define the input types for the auto‑refining alignment node.

        Only a gradient threshold is exposed to the user.  The search
        resolutions are fixed internally to progress through 1°, 0.1°,
        0.01°, 0.001° and 0.0001°.
        """
        return {
            "required": {
                "image_a": ("IMAGE",),
                "mask_a": ("MASK",),
                "image_b": ("IMAGE",),
                "mask_b": ("MASK",),
                # Gradient magnitude threshold (relative) between 0.04 and 0.1
                "grad_threshold": (
                    "FLOAT",
                    {
                        "default": 0.04,
                        "min": 0.04,
                        "max": 0.1,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                # Optional search window size (coarse bins).  Larger values
                # increase accuracy at the cost of more comparisons.
                "window_bins": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "display": "slider",
                    },
                ),
                # Minimum angle (in degrees) for the global search interval.
                # Values can be negative and wrap around 360.  For example,
                # -20 means 340°.
                "search_min_deg": (
                    "FLOAT",
                    {
                        "default": -180.0,
                        "min": -180.0,
                        "max": 180.0,
                        "step": 1.0,
                    },
                ),
                # Maximum angle (in degrees) for the global search interval.
                # If the maximum is less than the minimum, the interval wraps
                # around 360°.  For example, min=350 and max=10 defines a
                # search over [350°, 360°) ∪ [0°, 10°].
                "search_max_deg": (
                    "FLOAT",
                    {
                        "default": 180.0,
                        "min": -180.0,
                        "max": 180.0,
                        "step": 1.0,
                    },
                ),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("IMAGE", "MASK", "FLOAT")

    FUNCTION: str = "align_images_auto"

    @staticmethod
    def align_images_auto(
        image_a: "torch.Tensor",
        mask_a: "torch.Tensor",
        image_b: "torch.Tensor",
        mask_b: "torch.Tensor",
        grad_threshold: float = 0.04,
        window_bins: int = 3,
        search_min_deg: float = -180.0,
        search_max_deg: float = 180.0,
    ) -> Tuple["torch.Tensor", "torch.Tensor", Any]:
        """
        Align a batch of images using the multi‑resolution angle search.

        Parameters
        ----------
        image_a : torch.Tensor
            Batch of source images of shape ``[B, H, W, C]``.
        mask_a : torch.Tensor
            Batch of source masks of shape ``[B, H, W]``.
        image_b : torch.Tensor
            Batch of target images of shape ``[B, H, W, C]``.
        mask_b : torch.Tensor
            Batch of target masks of shape ``[B, H, W]``.
        grad_threshold : float
            Relative gradient magnitude threshold for histogram construction.
        window_bins : int
            Number of coarse bins on either side of the current best shift
            to search when refining.  A value of 3 is usually sufficient.
        search_min_deg : float
            Minimum angle (in degrees) of the global search interval.  Values
            outside the range [0, 360) wrap around.  If the minimum is
            greater than the maximum, the search wraps across 0°.
        search_max_deg : float
            Maximum angle (in degrees) of the global search interval.  If
            equal to the minimum, the search covers the full 360°.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The aligned images, the unchanged target masks and the
            rotation angles (one per batch element).
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for this custom node but was not imported.")
        if cv2 is None:
            raise RuntimeError("OpenCV is required for this custom node but was not imported.")

        B = image_a.shape[0]
        aligned_images: List["torch.Tensor"] = []
        out_masks: List["torch.Tensor"] = []
        angles: List[float] = []
        for idx in range(B):
            img_a_np = image_a[idx].detach().cpu().numpy()
            msk_a_np = mask_a[idx].detach().cpu().numpy()
            img_b_np = image_b[idx].detach().cpu().numpy()
            msk_b_np = mask_b[idx].detach().cpu().numpy()
            # Determine the rotation angle using the progressive search
            angle = _find_angle_auto_refine(
                img_a_np,
                msk_a_np,
                img_b_np,
                msk_b_np,
                grad_threshold=grad_threshold,
                exponents=(0, 1, 2, 3, 4),
                window_bins=window_bins,
                search_min_deg=search_min_deg,
                search_max_deg=search_max_deg,
            )
            # Compose and convert back to torch
            composed_np, out_msk_np = _rotate_and_compose(
                img_a_np,
                msk_a_np,
                img_b_np,
                msk_b_np,
                angle,
            )
            aligned_images.append(torch.from_numpy(composed_np).to(image_a.dtype))
            out_masks.append(torch.from_numpy(out_msk_np).to(mask_b.dtype))
            angles.append(float(angle))
        out_images = torch.stack(aligned_images, dim=0)
        out_masks_tensor = torch.stack(out_masks, dim=0)
        angles_tensor = torch.tensor(angles, dtype=torch.float32)
        return out_images, out_masks_tensor, angles_tensor


# Register the auto‑refine node alongside the original alignment node
NODE_CLASS_MAPPINGS.update({
    "AlignImagesAutoRefineNode": AlignImagesAutoRefineNode,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "AlignImagesAutoRefineNode": "Align Images (Auto Refine)",
})