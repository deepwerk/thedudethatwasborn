"""Custom ComfyUI node to align and overlay one image on top of another.

This node implements the logic contained within the provided Pluto.jl
notebook.  Given two input images (`images_A` and `images_B`) it
computes a rotation angle that best aligns the structure of `images_A`
with `images_B` based upon a simple gradient-orientation histogram
analysis.  It then rotates `images_A` by the estimated angle, crops
out the non‑zero region from both images, resizes the rotated crop of
`images_A` to the bounding box of `images_B` and overlays it onto
`images_B`.  Finally the output image is masked by the original
non‑zero pixels of `images_B` to retain transparency where the
background is black.  Both the composed RGB image and the binary
transparency mask are returned.

Inputs
------
images_A: IMAGE
    A batch of images to be rotated and overlaid onto the
    corresponding images in `images_B`.  The tensor must have shape
    `[B,H,W,C]` with `C=3` and dtype `float32`.  Pixel values are
    expected to be in the range `[0,1]`.
images_B: IMAGE
    A batch of images that provide the target orientation and
    bounding region.  Must have the same batch size and spatial
    dimensions as `images_A`.

Outputs
-------
IMAGE
    A batch of images with the rotated contents of `images_A`
    composited onto `images_B` wherever `images_B` was originally
    non‑zero.  Outside the masked region the pixels remain zero.
MASK
    A batch of binary masks of shape `[B,H,W]` that are `1` where the
    original `images_B` has any non‑zero value and `0` elsewhere.  It
    can be passed downstream to SaveImage or other nodes requiring
    alpha information.

Notes
-----
• This node operates on a per‑item basis: each image in the batch is
  paired with the image at the same index in the second batch.  If
  the batch sizes differ an exception will be raised.
• The rotation angle is determined by computing a histogram of edge
  orientations for both images using a Sobel operator and finding the
  circular shift that minimises the L¹ distance between the
  histograms.  The histogram bin size (`_DELTA_DEG`) is 0.1° by
  default, replicating the resolution used in the original Julia
  implementation.
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Any

import numpy as np
import torch
from scipy.ndimage import rotate as nd_rotate
import cv2  # OpenCV is used for Sobel gradient computation and resizing


class AlignAndOverlay:
    """ComfyUI custom node for aligning and compositing two images."""

    # Category under which the node will appear in the ComfyUI add‑node menu
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple[str, Dict[str, Any]]]]:
        """Define the input types for the node, including rotation matching mode selector."""
        return {
            "required": {
                "images_A": ("IMAGE", {}),
                "images_B": ("IMAGE", {}),
                "rotation_mode": ("STRING", {"default": "pca", "choices": ["pca", "hog"]}),
            },
        }

    # The node returns an image and a mask
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    # Name of the function that will be invoked to perform the work
    FUNCTION = "align_overlay"

    # Histogram bin width in degrees.  0.1° matches the Julia code
    _DELTA_DEG: float = 0.1
    # Gradient magnitude threshold.  Pixels with gradient magnitude below
    # this value are ignored when constructing the orientation histogram.
    _MAG_THRESHOLD: float = 0.04

    def _gradient_orientation_hist(self, image: np.ndarray) -> np.ndarray:
        """Compute a histogram of gradient orientations for a single image.

        Parameters
        ----------
        image: np.ndarray
            Input image of shape `(H,W,3)` with float values in [0,1].

        Returns
        -------
        np.ndarray
            A 1‑D array containing the normalised histogram of gradient
            orientations.  The histogram covers 0–360° in steps of
            `_DELTA_DEG`.  Entries sum to 1.  If no gradients exceed
            `_MAG_THRESHOLD` the histogram will be all zeros.
        """
        # Convert to float64 for improved numerical precision during the
        # Sobel operation.  Each channel is processed separately and
        # summed to approximate the behaviour of the Julia code where
        # `zahl(x)` sums the RGB components.
        img = image.astype(np.float64)
        # Compute Sobel derivatives for each channel
        grad_x = np.zeros(img.shape[:2], dtype=np.float64)
        grad_y = np.zeros(img.shape[:2], dtype=np.float64)
        for ch in range(3):
            # cv2.Sobel returns the derivative of the image with respect to x
            # and y.  Using ksize=3 for a 3×3 kernel.  cv2.CV_64F is used
            # to get floating point precision.
            gx = cv2.Sobel(img[:, :, ch], cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img[:, :, ch], cv2.CV_64F, 0, 1, ksize=3)
            grad_x += gx
            grad_y += gy

        # Compute orientation and magnitude.  Orientation is mapped from
        # [−π, π] to [0°,360°).  np.arctan2 returns values in [−π, π].
        # Convert the raw arctan2 angle from [−π,π] to degrees in [0,360).
        # `np.arctan2` returns an angle in radians centred on the negative
        # x‑axis.  The original implementation mistakenly shifted
        # orientations by π (180°), which effectively rotated all
        # orientations by half a circle.  The correct conversion is to
        # multiply by 180/π to get degrees and then take the result mod
        # 360 to wrap negative angles into the 0–360° range.  Without
        # the modulo, angles near −180° would be incorrectly mapped
        # near 180°.
        angles = (np.arctan2(grad_y, grad_x) * (180.0 / math.pi)) % 360.0
        magnitudes = np.hypot(grad_x, grad_y)

        # Threshold the magnitudes to remove low‑contrast regions.  This
        # threshold mirrors the Julia code's 0.04 cut‑off.  Because the
        # Sobel operator scales gradients into an arbitrary range dependent
        # on image resolution, this threshold may need tuning for other
        # datasets.  Users can modify `_MAG_THRESHOLD` if required.
        mask = magnitudes > self._MAG_THRESHOLD
        angles_sel = angles[mask]

        # Build histogram.  Include the right edge to cover 360°.
        num_bins = int(math.ceil(360.0 / self._DELTA_DEG))
        # Use numpy's histogram; bins are [0,360) in steps of delta
        # We compute bin edges explicitly to avoid floating point drift
        bin_edges = np.linspace(0.0, 360.0, num_bins + 1)
        hist, _ = np.histogram(angles_sel, bins=bin_edges)
        # Normalise to sum to one
        total = hist.sum()
        if total > 0:
            hist = hist.astype(np.float64) / total
        else:
            hist = hist.astype(np.float64)
        return hist

    @staticmethod
    def _mask_orientation_angle(mask: np.ndarray) -> float:
        """Compute the dominant orientation of a binary mask.

        The orientation is derived from the principal component of the
        coordinates of non‑zero pixels.  It returns an angle in
        degrees within [0,180).  Because a shape and its 180° rotated
        counterpart have identical second moments, the orientation is
        inherently ambiguous by 180°; this function normalises the
        angle into a half‑circle.

        Parameters
        ----------
        mask: np.ndarray
            A 2‑D boolean or integer array indicating the pixels of
            interest.  True/1 values denote foreground.

        Returns
        -------
        float
            The orientation angle in degrees in the range [0,180).
        """
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 0.0
        # Stack x and y coordinates; centre them to compute covariance
        coords = np.vstack((xs, ys))  # shape 2 × N
        coords_centered = coords - coords.mean(axis=1, keepdims=True)
        # Covariance matrix of coordinates
        cov = coords_centered @ coords_centered.T / coords_centered.shape[1]
        # Eigen decomposition; the largest eigenvalue corresponds to the
        # major axis of the distribution
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = int(np.argmax(eigvals))
        v = eigvecs[:, idx]
        # Angle of the eigenvector; arctan2 returns angle in [−π,π]
        angle = math.degrees(math.atan2(v[1], v[0]))
        # Map into [0,180)
        return angle % 180.0

    def _estimate_rotation_pca(self, image_A: np.ndarray, image_B: np.ndarray) -> float:
        """Estimate the rotation needed to align two images via PCA on masks.

        This method computes the orientation of the non‑zero masks of
        `image_A` and `image_B` using principal component analysis.
        The difference between the two orientations yields a candidate
        rotation.  Because the principal orientation is ambiguous up
        to 180°, four candidate rotations are considered: the
        positive and negative differences modulo 360°, and those
        differences plus 180°.  Each candidate rotation is applied
        (without reshaping) to `image_A` and the resulting mask is
        compared to the mask of `image_B` using a simple XOR count.
        The candidate yielding the smallest mask mismatch is selected.

        Parameters
        ----------
        image_A: np.ndarray
            Source image of shape `(H,W,C)`.
        image_B: np.ndarray
            Target image of shape `(H,W,C)`.

        Returns
        -------
        float
            Estimated rotation in degrees to apply to `image_A` to best
            align it with `image_B`.
        """
        eps = 1e-8
        mask_A = np.any(image_A > eps, axis=2)
        mask_B = np.any(image_B > eps, axis=2)
        # Orientation angles in [0,180)
        ang_A = self._mask_orientation_angle(mask_A)
        ang_B = self._mask_orientation_angle(mask_B)
        # Nominal difference in [0,180)
        diff = (ang_B - ang_A) % 180.0
        # Generate four candidate angles in [0,360): diff, diff+180, -diff, -diff+180
        c1 = diff
        c2 = (diff + 180.0) % 360.0
        c3 = (-diff) % 360.0
        c4 = ((-diff) + 180.0) % 360.0
        candidates = [c1, c2, c3, c4]
        best_theta = 0.0
        best_score = None
        for cand in candidates:
            # Rotate without reshape so that the mask sizes match B
            rot = nd_rotate(image_A, angle=cand, reshape=False, order=1, mode="constant", cval=0.0)
            mask_rot = np.any(rot > eps, axis=2)
            # Resize mask_rot to match mask_B if needed
            if mask_rot.shape != mask_B.shape:
                mask_rot = cv2.resize(mask_rot.astype(np.uint8), (mask_B.shape[1], mask_B.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            # Score: count of differing pixels between masks
            score = np.count_nonzero(mask_rot != mask_B)
            if best_score is None or score < best_score:
                best_score = score
                best_theta = cand
        return float(best_theta)

    def _find_best_rotation(self, hist_A: np.ndarray, hist_B: np.ndarray) -> float:
        """Find the rotation (in degrees) that aligns two orientation histograms.

        This method searches over all circular shifts of `hist_B` and
        identifies the shift that minimises the L¹ distance to
        `hist_A`.  The shift index is multiplied by `_DELTA_DEG` to
        produce the rotation angle in degrees.

        Parameters
        ----------
        hist_A: np.ndarray
            Normalised histogram for the first image.
        hist_B: np.ndarray
            Normalised histogram for the second image.

        Returns
        -------
        float
            The estimated rotation in degrees to apply to image A so as to
            align it with image B.
        """
        # Ensure histograms are one‑dimensional numpy arrays of the same length
        if hist_A.shape != hist_B.shape:
            raise ValueError("Histogram lengths do not match")
        n = hist_A.size
        # Preallocate an array to hold error values
        errors = np.empty(n, dtype=np.float64)
        # Compute the L¹ distance for each circular shift.  Although this
        # could be implemented using FFT convolution for efficiency,
        # here we use a direct approach for clarity and because n is
        # relatively small (≈3600 bins for 0.1° resolution).
        for i in range(n):
            # np.roll performs a circular shift of hist_B by i positions
            errors[i] = np.sum(np.abs(hist_A - np.roll(hist_B, i)))
        # Identify the shift yielding the minimal error
        best_index = int(np.argmin(errors))
        # Convert the index to degrees
        return best_index * self._DELTA_DEG

    @staticmethod
    def _nonzero_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Compute the bounding box of non‑zero pixels in a binary mask.

        Parameters
        ----------
        mask: np.ndarray
            A 2‑D boolean or integer array where non‑zero values indicate
            pixels of interest.

        Returns
        -------
        Tuple[int, int, int, int]
            `(y1, y2, x1, x2)` indices delimiting the bounding box.  The
            returned ranges are inclusive: the rows range from `y1` to
            `y2` and columns from `x1` to `x2`.  If the mask contains no
            true pixels the ranges will be `(0, -1, 0, -1)`.
        """
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            # Return empty bounding box
            return (0, -1, 0, -1)
        y1 = int(ys.min())
        y2 = int(ys.max())
        x1 = int(xs.min())
        x2 = int(xs.max())
        return (y1, y2, x1, x2)

    @staticmethod
    def _crop(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop an image using an inclusive bounding box.

        Parameters
        ----------
        image: np.ndarray
            Image of shape `(H,W,C)` to be cropped.
        bbox: Tuple[int, int, int, int]
            A tuple `(y1, y2, x1, x2)` specifying the inclusive
            bounds along rows and columns.

        Returns
        -------
        np.ndarray
            Cropped image of shape `(y2-y1+1, x2-x1+1, C)`.  If the
            bounding box is empty (`y2 < y1` or `x2 < x1`) an empty
            array with zero height and width is returned.
        """
        y1, y2, x1, x2 = bbox
        if y2 < y1 or x2 < x1:
            return image[0:0, 0:0, :]
        return image[y1 : y2 + 1, x1 : x2 + 1, :]

    def _rotate_image(self, image: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate an image by a given angle.

        Uses `scipy.ndimage.rotate` with `reshape=True` so that the entire
        rotated image fits within the output.  Pixels outside the input
        image are filled with zeros.

        Parameters
        ----------
        image: np.ndarray
            Image of shape `(H,W,3)` with float values.
        angle_deg: float
            Angle in degrees.  Positive values rotate counter‑clockwise.

        Returns
        -------
        np.ndarray
            Rotated image of shape `(H',W',3)` with float values.
        """
        # `nd_rotate` expects channel last, but rotates over the first two
        # axes.  Using order=1 for bilinear interpolation and constant
        # zero padding.
        rotated = nd_rotate(image, angle=angle_deg, reshape=True, order=1, mode="constant", cval=0.0)
        return rotated

    def align_overlay(self, images_A: torch.Tensor, images_B: torch.Tensor, rotation_mode: str = "pca") -> Tuple[torch.Tensor, torch.Tensor]:
        """Align and overlay a batch of images.

        Parameters
        ----------
        images_A: torch.Tensor
            Tensor of shape `[B,H,W,C]` containing the source images to be
            rotated and composited onto `images_B`.  `C` must be 3.
        images_B: torch.Tensor
            Tensor of shape `[B,H,W,C]` containing the target images used
            for alignment and the destination for the compositing.  `C`
            must be 3.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple `(images_out, masks_out)` where `images_out` has shape
            `[B,H,W,C]` containing the composited images and
            `masks_out` has shape `[B,H,W]` containing the binary mask
            derived from the non‑zero regions of `images_B`.
        """
        # Validate inputs
        if images_A.dim() != 4 or images_B.dim() != 4:
            raise ValueError("Input images must be 4‑D tensors with shape [B,H,W,C]")
        if images_A.shape[0] != images_B.shape[0]:
            raise ValueError(
                f"Batch size mismatch: images_A batch {images_A.shape[0]} vs images_B batch {images_B.shape[0]}"
            )
        if images_A.shape[-1] != 3 or images_B.shape[-1] != 3:
            raise ValueError("Images must have 3 channels (RGB)")
        batch_size = images_A.shape[0]

        # Prepare lists to collect outputs for each item in the batch
        out_images: list[torch.Tensor] = []
        out_masks: list[torch.Tensor] = []

        for idx in range(batch_size):
            # Convert tensors to numpy arrays on CPU.  We call .clone().detach()
            # to ensure there is no link to the computation graph.
            A_np = images_A[idx].cpu().detach().numpy().astype(np.float64)
            B_np = images_B[idx].cpu().detach().numpy().astype(np.float64)

            # Select rotation matching mode
            if rotation_mode == "hog":
                hist_A = self._gradient_orientation_hist(A_np)
                hist_B = self._gradient_orientation_hist(B_np)
                theta = self._find_best_rotation(hist_A, hist_B)  # No negation for HOG mode
                # Uncomment for debugging:
                # print(f"HOG rotation angle: {theta}")
            else:
                theta = -self._estimate_rotation_pca(A_np, B_np)

            # Rotate image A by the estimated angle
            A_rot = self._rotate_image(A_np, theta)

            # Compute masks of non‑black pixels for A_rot and B.  A pixel is
            # considered black if all its channels are zero.  Note: due to
            # floating point operations we test against a small epsilon
            # rather than exact zero to avoid misclassifying tiny values.
            eps = 1e-8
            mask_A_rot = np.any(A_rot > eps, axis=2)
            mask_B = np.any(B_np > eps, axis=2)

            # Determine bounding boxes of the non‑zero regions
            bbox_A = self._nonzero_bbox(mask_A_rot)
            bbox_B = self._nonzero_bbox(mask_B)

            # Crop the non‑zero regions
            A_crop = self._crop(A_rot, bbox_A)
            B_crop = self._crop(B_np, bbox_B)

            # Resize the cropped rotated A to match the size of B's crop
            if B_crop.size == 0:
                # If B has no non‑zero region, the mask will be all zeros
                # and the output image should be entirely black
                composite_np = B_np.copy()
            else:
                # Determine target dimensions (height and width) from B's crop
                target_h = B_crop.shape[0]
                target_w = B_crop.shape[1]

                # Handle the degenerate case where A has no non‑zero region
                if A_crop.size == 0:
                    resized_A = np.zeros((target_h, target_w, 3), dtype=A_crop.dtype)
                else:
                    # Resize using bilinear interpolation.  cv2.resize expects
                    # (width,height) order for the dsize argument.
                    resized_A = cv2.resize(A_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                # Composite: start with a copy of B and replace the crop
                composite_np = B_np.copy()
                y1_B, y2_B, x1_B, x2_B = bbox_B
                # Ensure the shapes agree
                if (y2_B - y1_B + 1) != target_h or (x2_B - x1_B + 1) != target_w:
                    raise AssertionError(
                        "Internal error: mismatch between crop size and target dimensions"
                    )
                composite_np[y1_B : y2_B + 1, x1_B : x2_B + 1, :] = resized_A

            # Apply the original B mask to preserve transparency.  mask_B has
            # shape (H,W); broadcast it to (H,W,3) for multiplication.
            composite_np *= mask_B[:, :, None].astype(composite_np.dtype)

            # Convert back to torch.Tensor with the same dtype as the inputs
            composite_t = torch.from_numpy(composite_np.astype(np.float32))
            mask_t = torch.from_numpy(mask_B.astype(np.float32))
            # Add batch dimension
            out_images.append(composite_t.unsqueeze(0))
            out_masks.append(mask_t.unsqueeze(0))

        # Concatenate along the batch dimension
        images_out = torch.cat(out_images, dim=0)
        masks_out = torch.cat(out_masks, dim=0)
        return (images_out, masks_out)


# Mapping from class name to class object required by ComfyUI.
NODE_CLASS_MAPPINGS = {
    "AlignAndOverlay": AlignAndOverlay,
}

# Optionally you can define display names for your nodes.  If omitted,
# ComfyUI will use the class names.
NODE_DISPLAY_NAME_MAPPINGS = {
    "AlignAndOverlay": "Align and Overlay (Julia port)",
}