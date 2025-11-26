import logging
import os
import tempfile
import time
from typing import Iterable

import numpy as np
import pydicom

logger = logging.getLogger(__name__)

# Global singleton instances for nvimgcodec encoder/decoder
# These are initialized lazily on first use to avoid import errors
# when nvimgcodec is not available
_NVIMGCODEC_ENCODER = None
_NVIMGCODEC_DECODER = None


def _get_nvimgcodec_encoder():
    """Get or create the global nvimgcodec encoder instance."""
    global _NVIMGCODEC_ENCODER
    if _NVIMGCODEC_ENCODER is None:
        from nvidia import nvimgcodec

        _NVIMGCODEC_ENCODER = nvimgcodec.Encoder()
    return _NVIMGCODEC_ENCODER


def _get_nvimgcodec_decoder():
    """Get or create the global nvimgcodec decoder instance."""
    global _NVIMGCODEC_DECODER
    if _NVIMGCODEC_DECODER is None:
        from nvidia import nvimgcodec

        _NVIMGCODEC_DECODER = nvimgcodec.Decoder(options=":fancy_upsampling=1")
    return _NVIMGCODEC_DECODER


def _setup_htj2k_decode_params(color_spec=None):
    """
    Create nvimgcodec decoding parameters for DICOM images.

    Args:
        color_spec: Color specification to use. If None, defaults to UNCHANGED.

    Returns:
        nvimgcodec.DecodeParams: Decode parameters configured for DICOM
    """
    from nvidia import nvimgcodec

    if color_spec is None:
        color_spec = nvimgcodec.ColorSpec.UNCHANGED
    decode_params = nvimgcodec.DecodeParams(
        allow_any_depth=True,
        color_spec=color_spec,
    )
    return decode_params


def _setup_htj2k_encode_params(
    num_resolutions: int = 6, code_block_size: tuple = (64, 64), progression_order: str = "RPCL"
):
    """
    Create nvimgcodec encoding parameters for HTJ2K lossless compression.

    Args:
        num_resolutions: Number of wavelet decomposition levels
        code_block_size: Code block size as (height, width) tuple
        progression_order: Progression order for encoding. Must be one of:
            - "LRCP": Layer-Resolution-Component-Position (quality scalability)
            - "RLCP": Resolution-Layer-Component-Position (resolution scalability)
            - "RPCL": Resolution-Position-Component-Layer (progressive by resolution)
            - "PCRL": Position-Component-Resolution-Layer (progressive by spatial area)
            - "CPRL": Component-Position-Resolution-Layer (component scalability)

    Returns:
        tuple: (encode_params, target_transfer_syntax)

    Raises:
        ValueError: If progression_order is not one of the valid values
    """
    from nvidia import nvimgcodec

    # Valid progression orders and their mappings
    VALID_PROG_ORDERS = {
        "LRCP": (nvimgcodec.Jpeg2kProgOrder.LRCP, "1.2.840.10008.1.2.4.201"),  # HTJ2K (Lossless Only)
        "RLCP": (nvimgcodec.Jpeg2kProgOrder.RLCP, "1.2.840.10008.1.2.4.201"),  # HTJ2K (Lossless Only)
        "RPCL": (nvimgcodec.Jpeg2kProgOrder.RPCL, "1.2.840.10008.1.2.4.202"),  # HTJ2K with RPCL Options
        "PCRL": (nvimgcodec.Jpeg2kProgOrder.PCRL, "1.2.840.10008.1.2.4.201"),  # HTJ2K (Lossless Only)
        "CPRL": (nvimgcodec.Jpeg2kProgOrder.CPRL, "1.2.840.10008.1.2.4.201"),  # HTJ2K (Lossless Only)
    }

    # Validate progression order
    if progression_order not in VALID_PROG_ORDERS:
        valid_orders = ", ".join(f"'{o}'" for o in VALID_PROG_ORDERS.keys())
        raise ValueError(f"Invalid progression_order '{progression_order}'. " f"Must be one of: {valid_orders}")

    # Get progression order enum and transfer syntax
    prog_order_enum, target_transfer_syntax = VALID_PROG_ORDERS[progression_order]

    quality_type = nvimgcodec.QualityType.LOSSLESS

    # Configure JPEG2K encoding parameters
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
    jpeg2k_encode_params.num_resolutions = num_resolutions
    jpeg2k_encode_params.code_block_size = code_block_size
    jpeg2k_encode_params.bitstream_type = nvimgcodec.Jpeg2kBitstreamType.J2K
    jpeg2k_encode_params.prog_order = prog_order_enum
    jpeg2k_encode_params.ht = True  # Enable High Throughput mode

    encode_params = nvimgcodec.EncodeParams(
        quality_type=quality_type,
        jpeg2k_encode_params=jpeg2k_encode_params,
    )

    return encode_params, target_transfer_syntax


def _extract_frames_from_compressed(ds, number_of_frames=None):
    """
    Extract frames from encapsulated (compressed) DICOM pixel data.

    Args:
        ds: pydicom Dataset with encapsulated PixelData
        number_of_frames: Expected number of frames (from NumberOfFrames tag)

    Returns:
        list: List of compressed frame data (bytes)
    """
    # Default to 1 frame if not specified (for single-frame images without NumberOfFrames tag)
    if number_of_frames is None:
        number_of_frames = 1

    frames = list(pydicom.encaps.generate_frames(ds.PixelData, number_of_frames=number_of_frames))
    return frames


def _extract_frames_from_uncompressed(pixel_array, num_frames_tag):
    """
    Extract individual frames from uncompressed pixel array.

    Handles different array shapes:
    - 2D (H, W): single frame grayscale
    - 3D (N, H, W): multi-frame grayscale OR (H, W, C): single frame color
    - 4D (N, H, W, C): multi-frame color

    Args:
        pixel_array: Numpy array of pixel data
        num_frames_tag: NumberOfFrames value from DICOM tag

    Returns:
        list: List of frame arrays
    """
    if not isinstance(pixel_array, np.ndarray):
        pixel_array = np.array(pixel_array)

    # 2D: single frame grayscale
    if pixel_array.ndim == 2:
        return [pixel_array]

    # 3D: multi-frame grayscale OR single-frame color
    if pixel_array.ndim == 3:
        if num_frames_tag > 1 or pixel_array.shape[0] == num_frames_tag:
            # Multi-frame grayscale: (N, H, W)
            return [pixel_array[i] for i in range(pixel_array.shape[0])]
        # Single-frame color: (H, W, C)
        return [pixel_array]

    # 4D: multi-frame color
    if pixel_array.ndim == 4:
        return [pixel_array[i] for i in range(pixel_array.shape[0])]

    raise ValueError(f"Unexpected pixel array dimensions: {pixel_array.ndim}")


def _validate_frames(frames, context_msg="Frame"):
    """
    Check for None values in decoded/encoded frames.

    Args:
        frames: List of frames to validate
        context_msg: Context message for error reporting

    Raises:
        ValueError: If any frame is None
    """
    for idx, frame in enumerate(frames):
        if frame is None:
            raise ValueError(f"{context_msg} {idx} failed (returned None)")


def _find_dicom_files(input_dir):
    """
    Recursively find all valid DICOM files in a directory.

    Args:
        input_dir: Directory to search

    Returns:
        list: Sorted list of DICOM file paths
    """
    valid_dicom_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "rb") as fp:
                        fp.seek(128)
                        if fp.read(4) == b"DICM":
                            valid_dicom_files.append(file_path)
                except Exception:
                    continue

    valid_dicom_files.sort()  # For reproducible processing order
    return valid_dicom_files


def _get_transfer_syntax_constants():
    """
    Get transfer syntax UID constants for categorizing DICOM files.

    Returns:
        dict: Dictionary with keys 'JPEG2000', 'HTJ2K', 'JPEG', 'NVIMGCODEC' (combined set)
    """
    JPEG2000_SYNTAXES = frozenset(
        [
            "1.2.840.10008.1.2.4.90",  # JPEG 2000 Image Compression (Lossless Only)
            "1.2.840.10008.1.2.4.91",  # JPEG 2000 Image Compression
        ]
    )

    HTJ2K_SYNTAXES = frozenset(
        [
            "1.2.840.10008.1.2.4.201",  # High-Throughput JPEG 2000 Image Compression (Lossless Only)
            "1.2.840.10008.1.2.4.202",  # High-Throughput JPEG 2000 with RPCL Options Image Compression (Lossless Only)
            "1.2.840.10008.1.2.4.203",  # High-Throughput JPEG 2000 Image Compression
        ]
    )

    JPEG_SYNTAXES = frozenset(
        [
            "1.2.840.10008.1.2.4.50",  # JPEG Baseline (Process 1)
            "1.2.840.10008.1.2.4.51",  # JPEG Extended (Process 2 & 4)
            "1.2.840.10008.1.2.4.57",  # JPEG Lossless, Non-Hierarchical (Process 14)
            "1.2.840.10008.1.2.4.70",  # JPEG Lossless, Non-Hierarchical, First-Order Prediction
        ]
    )

    return {
        "JPEG2000": JPEG2000_SYNTAXES,
        "HTJ2K": HTJ2K_SYNTAXES,
        "JPEG": JPEG_SYNTAXES,
        "NVIMGCODEC": JPEG2000_SYNTAXES | HTJ2K_SYNTAXES | JPEG_SYNTAXES,
    }


class DicomFileLoader:
    """
    Simple iterable that auto-discovers DICOM files from a directory and yields batches.

    This class provides a simple interface for batch processing DICOM files without
    requiring external dependencies like PyTorch. It can be used with any function
    that accepts an iterable of (input_batch, output_batch) tuples.

    Args:
        input_dir: Path to directory containing DICOM files to process
        output_dir: Path to output directory. Output paths will preserve the directory
                   structure relative to input_dir.
        batch_size: Number of files to include in each batch (default: 256)

    Yields:
        tuple: (batch_input, batch_output) where both are lists of file paths
               batch_input contains source file paths
               batch_output contains corresponding output file paths with preserved directory structure

    Example:
        >>> loader = DicomFileLoader("/path/to/dicoms", "/path/to/output", batch_size=50)
        >>> for batch_in, batch_out in loader:
        ...     print(f"Processing {len(batch_in)} files")
        ...     print(f"Input: {batch_in[0]}")
        ...     print(f"Output: {batch_out[0]}")
    """

    def __init__(self, input_dir: str, output_dir: str, batch_size: int = 256):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self._files = None

    def _discover_files(self):
        """Discover DICOM files in the input directory."""
        if self._files is None:
            # Validate input
            if not os.path.exists(self.input_dir):
                raise ValueError(f"Input directory does not exist: {self.input_dir}")

            if not os.path.isdir(self.input_dir):
                raise ValueError(f"Input path is not a directory: {self.input_dir}")

            # Find all valid DICOM files
            self._files = _find_dicom_files(self.input_dir)
            if not self._files:
                raise ValueError(f"No valid DICOM files found in {self.input_dir}")

            logger.info(f"Found {len(self._files)} DICOM files to process")

    def __iter__(self):
        """Iterate over batches of DICOM files."""
        self._discover_files()

        total_files = len(self._files)
        for batch_start in range(0, total_files, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_files)
            batch_input = self._files[batch_start:batch_end]

            # Compute output paths preserving directory structure
            batch_output = []
            for input_path in batch_input:
                relative_path = os.path.relpath(input_path, self.input_dir)
                output_path = os.path.join(self.output_dir, relative_path)
                batch_output.append(output_path)

            yield batch_input, batch_output


def transcode_dicom_to_htj2k(
    file_loader: Iterable[tuple[list[str], list[str]]],
    num_resolutions: int = 6,
    code_block_size: tuple = (64, 64),
    progression_order: str = "RPCL",
    max_batch_size: int = 256,
    add_basic_offset_table: bool = True,
    skip_transfer_syntaxes: list = (
        _get_transfer_syntax_constants()["HTJ2K"]
        | frozenset(
            [
                # Lossy JPEG 2000
                "1.2.840.10008.1.2.4.91",  # JPEG 2000 Image Compression (lossy allowed)
                # Lossy JPEG
                "1.2.840.10008.1.2.4.50",  # JPEG Baseline (Process 1) - always lossy
                "1.2.840.10008.1.2.4.51",  # JPEG Extended (Process 2 & 4, can be lossy)
            ]
        )
    ),
):
    """
    Transcode DICOM files to HTJ2K (High Throughput JPEG 2000) lossless compression.

    HTJ2K is a faster variant of JPEG 2000 that provides better compression performance
    for medical imaging applications. This function uses NVIDIA's nvimgcodec for hardware-
    accelerated decoding and encoding with batch processing for optimal performance.
    All transcoding is performed using lossless compression to preserve image quality.

    The function processes files with streaming decode-encode batches:
    1. Categorizes files by transfer syntax (HTJ2K/JPEG2000/JPEG/uncompressed)
    2. Extracts all frames from source files
    3. Processes frames in batches for efficient encoding:
       - Decodes batch using nvimgcodec (compressed) or pydicom (uncompressed)
       - Immediately encodes batch to HTJ2K
       - Discards decoded frames to save memory (streaming)
    4. Saves transcoded files with updated transfer syntax and optional Basic Offset Table

    This streaming approach minimizes memory usage by never holding all decoded frames
    in memory simultaneously.

    Supported source transfer syntaxes:
    - HTJ2K (High-Throughput JPEG 2000) - decoded and re-encoded (add bot if needed)
    - JPEG 2000 (lossless and lossy)
    - JPEG (baseline, extended, lossless)
    - Uncompressed (Explicit/Implicit VR Little/Big Endian)

    Typical compression ratios of 60-70% with lossless quality.
    Processing speed depends on batch size and GPU capabilities.

    Args:
        file_loader:
            Iterable of (input_files, output_files) tuples, where:
                - input_files: List[str] of input DICOM file paths to transcode as a batch.
                - output_files: List[str] of output file paths to write the transcoded DICOMs.
            The recommended usage is to provide a DicomFileLoader instance, which automatically yields
            appropriately sized batches of file paths for efficient streaming. Custom iterables can also
            be used to precisely control batching or file selection.

            Each yielded tuple should contain two lists of identical length, specifying the correspondence
            between input and output files for each batch. The function will read each input file,
            transcode to HTJ2K if necessary, and write the result to the corresponding output file.

            Example:
                for batch_input, batch_output in file_loader:
                    # len(batch_input) == len(batch_output)
                    # batch_input: ['a.dcm', 'b.dcm'], batch_output: ['a_out.dcm', 'b_out.dcm']
            The loader should guarantee that input and output lists are aligned and consistent across batches.
        num_resolutions: Number of wavelet decomposition levels (default: 6)
                        Higher values = better compression but slower encoding
        code_block_size: Code block size as (height, width) tuple (default: (64, 64))
                        Must be powers of 2. Common values: (32,32), (64,64), (128,128)
        progression_order: Progression order for HTJ2K encoding (default: "RPCL")
                          Must be one of: "LRCP", "RLCP", "RPCL", "PCRL", "CPRL"
                          - "LRCP": Layer-Resolution-Component-Position (quality scalability)
                          - "RLCP": Resolution-Layer-Component-Position (resolution scalability)
                          - "RPCL": Resolution-Position-Component-Layer (progressive by resolution)
                          - "PCRL": Position-Component-Resolution-Layer (progressive by spatial area)
                          - "CPRL": Component-Position-Resolution-Layer (component scalability)
        max_batch_size: Maximum number of frames to decode/encode in parallel (default: 256)
                       This controls internal frame-level batching for GPU operations, not file-level batching.
                       Lower values reduce memory usage, higher values may improve GPU utilization.
                       Note: File-level batching is controlled by the DicomFileLoader's batch_size parameter.
        add_basic_offset_table: If True, creates Basic Offset Table for multi-frame DICOMs (default: True)
                               BOT enables O(1) frame access without parsing entire pixel data stream
                               Per DICOM Part 5 Section A.4. Only affects multi-frame files.
        skip_transfer_syntaxes: Optional list of Transfer Syntax UIDs to skip transcoding (default: HTJ2K, lossy JPEG 2000, and lossy JPEG)
                               Files with these transfer syntaxes will be copied directly to output
                               without transcoding. Useful for preserving already-compressed formats.
                               Example: ["1.2.840.10008.1.2.4.201", "1.2.840.10008.1.2.4.202"]

    Raises:
        ImportError: If nvidia-nvimgcodec is not available
        ValueError: If DICOM files are missing required attributes (TransferSyntaxUID, PixelData)
        ValueError: If progression_order is not one of: "LRCP", "RLCP", "RPCL", "PCRL", "CPRL"

    Example:
        >>> # Basic usage with DicomFileLoader
        >>> loader = DicomFileLoader("/path/to/input", "/path/to/output")
        >>> transcode_dicom_to_htj2k(loader)
        >>> print(f"Transcoded files saved to: {loader.output_dir}")

        >>> # Custom settings with DicomFileLoader
        >>> loader = DicomFileLoader("/path/to/input", "/path/to/output", batch_size=50)
        >>> transcode_dicom_to_htj2k(
        ...     file_loader=loader,
        ...     num_resolutions=5,
        ...     code_block_size=(32, 32)
        ... )

        >>> # Skip transcoding for files already in HTJ2K format
        >>> loader = DicomFileLoader("/path/to/input", "/path/to/output")
        >>> transcode_dicom_to_htj2k(
        ...     file_loader=loader,
        ...     skip_transfer_syntaxes=["1.2.840.10008.1.2.4.201", "1.2.840.10008.1.2.4.202"]
        ... )

    Note:
        Requires nvidia-nvimgcodec to be installed:
            pip install nvidia-nvimgcodec-cu{XX}[all]
        Replace {XX} with your CUDA version (e.g., cu13 for CUDA 13.x)

        The function preserves all DICOM metadata including Patient, Study, and Series
        information. Only the transfer syntax and pixel data encoding are modified.
    """
    import shutil

    # Check for nvidia-nvimgcodec
    try:
        from nvidia import nvimgcodec
    except ImportError:
        raise ImportError(
            "nvidia-nvimgcodec is required for HTJ2K transcoding. "
            "Install it with: pip install nvidia-nvimgcodec-cu{XX}[all] "
            "(replace {XX} with your CUDA version, e.g., cu13)"
        )

    # Create encoder and decoder instances (reused for all files)
    encoder = _get_nvimgcodec_encoder()
    decoder = _get_nvimgcodec_decoder()  # Always needed for decoding input DICOM images

    # Setup HTJ2K encoding parameters
    encode_params, target_transfer_syntax = _setup_htj2k_encode_params(
        num_resolutions=num_resolutions, code_block_size=code_block_size, progression_order=progression_order
    )
    # Note: decode_params is created per-PhotometricInterpretation group in the batch processing
    logger.info("Using lossless HTJ2K compression")

    # Get transfer syntax constants
    ts_constants = _get_transfer_syntax_constants()
    NVIMGCODEC_SYNTAXES = ts_constants["NVIMGCODEC"]

    # Initialize skip list
    if skip_transfer_syntaxes is None:
        skip_transfer_syntaxes = []
    else:
        # Convert to set of strings for faster lookup
        skip_transfer_syntaxes = {str(ts) for ts in skip_transfer_syntaxes}
        logger.info(f"Files with these transfer syntaxes will be copied without transcoding: {skip_transfer_syntaxes}")

    start_time = time.time()
    transcoded_count = 0
    skipped_count = 0
    total_files = 0

    # Iterate over batches from file_loader
    for batch_in, batch_out in file_loader:
        batch_datasets = [pydicom.dcmread(file) for file in batch_in]
        total_files += len(batch_datasets)
        nvimgcodec_batch = []
        pydicom_batch = []
        skip_batch = []  # Indices of files to skip (copy directly)

        for idx, ds in enumerate(batch_datasets):
            current_ts = getattr(ds, "file_meta", {}).get("TransferSyntaxUID", None)
            if current_ts is None:
                raise ValueError(f"DICOM file {os.path.basename(batch_in[idx])} does not have a Transfer Syntax UID")

            ts_str = str(current_ts)

            # Check if this transfer syntax should be skipped
            if ts_str in skip_transfer_syntaxes:
                skip_batch.append(idx)
                logger.info(f"  Skipping {os.path.basename(batch_in[idx])} (Transfer Syntax: {ts_str})")
                continue

            if ts_str in NVIMGCODEC_SYNTAXES:
                if not hasattr(ds, "PixelData") or ds.PixelData is None:
                    raise ValueError(f"DICOM file {os.path.basename(batch_in[idx])} does not have a PixelData member")
                nvimgcodec_batch.append(idx)
            else:
                pydicom_batch.append(idx)

        # Handle skip_batch: copy files directly to output
        if skip_batch:
            for idx in skip_batch:
                source_file = batch_in[idx]
                output_file = batch_out[idx]

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                shutil.copy2(source_file, output_file)
                skipped_count += 1
                logger.info(f"  Copied {os.path.basename(source_file)} to output (skipped transcoding)")

        num_frames = []
        encoded_data = []

        # Process nvimgcodec_batch: extract frames, decode, encode in streaming batches
        if nvimgcodec_batch:
            from collections import defaultdict

            # First, extract all compressed frames and group by PhotometricInterpretation
            grouped_frames = defaultdict(list)  # Key: PhotometricInterpretation, Value: list of (file_idx, frame_data)
            frame_counts = {}  # Track number of frames per file
            successful_nvimgcodec_batch = []  # Track successfully processed files

            logger.info(f"  Extracting frames from {len(nvimgcodec_batch)} nvimgcodec files:")
            for idx in nvimgcodec_batch:
                try:
                    ds = batch_datasets[idx]
                    number_of_frames = int(ds.NumberOfFrames) if hasattr(ds, "NumberOfFrames") else None

                    if "PixelData" not in ds:
                        logger.warning(f"Skipping file {batch_in[idx]} (index {idx}): no PixelData found")
                        skipped_count += 1
                        continue

                    frames = _extract_frames_from_compressed(ds, number_of_frames)
                    logger.info(
                        f"    File idx={idx} ({os.path.basename(batch_in[idx])}): extracted {len(frames)} frames (expected: {number_of_frames})"
                    )

                    # Get PhotometricInterpretation for this file
                    photometric = getattr(ds, "PhotometricInterpretation", "UNKNOWN")

                    # Store frames grouped by PhotometricInterpretation
                    for frame in frames:
                        grouped_frames[photometric].append((idx, frame))

                    frame_counts[idx] = len(frames)
                    num_frames.append(len(frames))
                    successful_nvimgcodec_batch.append(idx)  # Only add if successful
                except Exception as e:
                    logger.warning(f"Skipping file {batch_in[idx]} (index {idx}): {e}")
                    skipped_count += 1

            # Update nvimgcodec_batch to only include successfully processed files
            nvimgcodec_batch = successful_nvimgcodec_batch

            # Process each PhotometricInterpretation group separately
            logger.info(f"  Found {len(grouped_frames)} unique PhotometricInterpretation groups")

            # Track encoded frames per file to maintain order
            encoded_frames_by_file = {idx: [] for idx in nvimgcodec_batch}

            for photometric, frame_list in grouped_frames.items():
                # Determine color_spec based on PhotometricInterpretation
                if photometric.startswith("YBR"):
                    color_spec = nvimgcodec.ColorSpec.RGB
                    logger.info(
                        f"  Processing {len(frame_list)} frames with PhotometricInterpretation={photometric} using color_spec=RGB"
                    )
                else:
                    color_spec = nvimgcodec.ColorSpec.UNCHANGED
                    logger.info(
                        f"  Processing {len(frame_list)} frames with PhotometricInterpretation={photometric} using color_spec=UNCHANGED"
                    )

                # Create decode params for this group
                group_decode_params = _setup_htj2k_decode_params(color_spec=color_spec)

                # Extract just the frame data (without file index)
                compressed_frames = [frame_data for _, frame_data in frame_list]

                # Decode and encode in batches (streaming to reduce memory)
                total_frames = len(compressed_frames)

                for frame_batch_start in range(0, total_frames, max_batch_size):
                    frame_batch_end = min(frame_batch_start + max_batch_size, total_frames)
                    compressed_batch = compressed_frames[frame_batch_start:frame_batch_end]
                    file_indices_batch = [file_idx for file_idx, _ in frame_list[frame_batch_start:frame_batch_end]]

                    if total_frames > max_batch_size:
                        logger.info(
                            f"    Processing frames [{frame_batch_start}..{frame_batch_end}) of {total_frames} for {photometric}"
                        )

                    # Decode batch with appropriate color_spec
                    decoded_batch = decoder.decode(compressed_batch, params=group_decode_params)
                    _validate_frames(decoded_batch, f"Decoded frame [{frame_batch_start}+")

                    # Encode batch immediately (streaming - no need to keep decoded data)
                    encoded_batch = encoder.encode(decoded_batch, codec="jpeg2k", params=encode_params)
                    _validate_frames(encoded_batch, f"Encoded frame [{frame_batch_start}+")

                    # Store encoded frames by file index to maintain order
                    for file_idx, encoded_frame in zip(file_indices_batch, encoded_batch):
                        encoded_frames_by_file[file_idx].append(encoded_frame)

                    # decoded_batch is automatically freed here

            # Reconstruct encoded_data in original file order
            for idx in nvimgcodec_batch:
                encoded_data.extend(encoded_frames_by_file[idx])

        # Process pydicom_batch: extract frames and encode in streaming batches
        if pydicom_batch:
            # Extract all frames from uncompressed files
            all_decoded_frames = []
            successful_pydicom_batch = []  # Track successfully processed files

            for idx in pydicom_batch:
                try:
                    ds = batch_datasets[idx]
                    num_frames_tag = int(ds.NumberOfFrames) if hasattr(ds, "NumberOfFrames") else 1
                    if "PixelData" in ds:
                        frames = _extract_frames_from_uncompressed(ds.pixel_array, num_frames_tag)
                        all_decoded_frames.extend(frames)
                        num_frames.append(len(frames))
                        successful_pydicom_batch.append(idx)  # Only add if successful
                    else:
                        # No PixelData - log warning and skip file completely
                        logger.warning(f"Skipping file {batch_in[idx]} (index {idx}): no PixelData found")
                        skipped_count += 1
                except Exception as e:
                    logger.warning(f"Skipping file {batch_in[idx]} (index {idx}): {e}")
                    skipped_count += 1

            # Encode in batches (streaming)
            total_frames = len(all_decoded_frames)
            if total_frames > 0:
                logger.info(f"  Encoding {total_frames} uncompressed frames in batches of {max_batch_size}")

                for frame_batch_start in range(0, total_frames, max_batch_size):
                    frame_batch_end = min(frame_batch_start + max_batch_size, total_frames)
                    decoded_batch = all_decoded_frames[frame_batch_start:frame_batch_end]

                    if total_frames > max_batch_size:
                        logger.info(f"    Encoding frames [{frame_batch_start}..{frame_batch_end}) of {total_frames}")

                    # Encode batch
                    encoded_batch = encoder.encode(decoded_batch, codec="jpeg2k", params=encode_params)
                    _validate_frames(encoded_batch, f"Encoded frame [{frame_batch_start}+")

                    # Store encoded frames
                    encoded_data.extend(encoded_batch)

            # Update pydicom_batch to only include successfully processed files
            pydicom_batch = successful_pydicom_batch

        # Reassemble and save transcoded files
        frame_offset = 0
        files_to_process = nvimgcodec_batch + pydicom_batch

        for list_idx, dataset_idx in enumerate(files_to_process):
            nframes = num_frames[list_idx]
            encoded_frames = [bytes(enc) for enc in encoded_data[frame_offset : frame_offset + nframes]]
            frame_offset += nframes

            # Update dataset with HTJ2K encoded data
            # Create Basic Offset Table for multi-frame files if requested
            batch_datasets[dataset_idx].PixelData = pydicom.encaps.encapsulate(
                encoded_frames, has_bot=add_basic_offset_table
            )
            batch_datasets[dataset_idx].file_meta.TransferSyntaxUID = pydicom.uid.UID(target_transfer_syntax)

            # Update PhotometricInterpretation to RGB for YBR images since we decoded with RGB color_spec
            # The pixel data is now in RGB color space, so the metadata must reflect this
            # to prevent double conversion by DICOM readers
            if hasattr(batch_datasets[dataset_idx], "PhotometricInterpretation"):
                original_pi = batch_datasets[dataset_idx].PhotometricInterpretation
                if original_pi.startswith("YBR"):
                    batch_datasets[dataset_idx].PhotometricInterpretation = "RGB"
                    logger.info(f"  Updated PhotometricInterpretation: {original_pi} -> RGB")

            try:
                # Save transcoded file using output path from file_loader
                input_file = batch_in[dataset_idx]
                output_file = batch_out[dataset_idx]

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                batch_datasets[dataset_idx].save_as(output_file)
                transcoded_count += 1
                logger.info(f"#{transcoded_count}: Transcoded {input_file}, saving as: {output_file}")
            except Exception as e:
                logger.error(f"Error saving transcoded file {batch_in[dataset_idx]}: {output_file}")
                logger.error(f"Error: {e}")

    elapsed_time = time.time() - start_time

    logger.info(f"Transcoding complete:")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Successfully transcoded: {transcoded_count}")
    logger.info(f"  Skipped (copied without transcoding): {skipped_count}")
    logger.info(f"  Time elapsed: {elapsed_time:.2f} seconds")


def convert_single_frame_dicom_series_to_multiframe(
    input_dir: str,
    output_dir: str = None,
    convert_to_htj2k: bool = False,
    num_resolutions: int = 6,
    code_block_size: tuple = (64, 64),
    progression_order: str = "RPCL",
    add_basic_offset_table: bool = True,
) -> str:
    """
    Convert single-frame DICOM series to multi-frame DICOM files, optionally with HTJ2K compression.

    This function groups DICOM files by SeriesInstanceUID and combines all frames from each series
    into a single multi-frame DICOM file. This is useful for:
    - Reducing file count (one file per series instead of many)
    - Improving storage efficiency
    - Enabling more efficient frame-level access patterns

    The function:
    1. Scans input directory recursively for DICOM files
    2. Groups files by StudyInstanceUID and SeriesInstanceUID
    3. For each series, decodes all frames and combines them
    4. Optionally encodes combined frames to HTJ2K (if convert_to_htj2k=True)
    5. Creates a Basic Offset Table for efficient frame access (per DICOM Part 5 Section A.4)
    6. Saves as a single multi-frame DICOM file per series

    Args:
        input_dir: Path to directory containing DICOM files (will scan recursively)
        output_dir: Path to output directory for transcoded files. If None, creates temp directory
        convert_to_htj2k: If True, convert frames to HTJ2K compression; if False, use uncompressed format (default: False)
        num_resolutions: Number of wavelet decomposition levels (default: 6, only used if convert_to_htj2k=True)
        code_block_size: Code block size as (height, width) tuple (default: (64, 64), only used if convert_to_htj2k=True)
        progression_order: Progression order for HTJ2K encoding (default: "RPCL", only used if convert_to_htj2k=True)
                          Must be one of: "LRCP", "RLCP", "RPCL", "PCRL", "CPRL"
                          - "LRCP": Layer-Resolution-Component-Position (quality scalability)
                          - "RLCP": Resolution-Layer-Component-Position (resolution scalability)
                          - "RPCL": Resolution-Position-Component-Layer (progressive by resolution)
                          - "PCRL": Position-Component-Resolution-Layer (progressive by spatial area)
                          - "CPRL": Component-Position-Resolution-Layer (component scalability)
        add_basic_offset_table: If True, creates Basic Offset Table for multi-frame DICOMs (default: True)
                               BOT enables O(1) frame access without parsing entire pixel data stream
                               Per DICOM Part 5 Section A.4. Only affects multi-frame files.

    Returns:
        str: Path to output directory containing multi-frame DICOM files

    Raises:
        ImportError: If nvidia-nvimgcodec is not available and convert_to_htj2k=True
        ValueError: If input directory doesn't exist or contains no valid DICOM files
        ValueError: If progression_order is not one of: "LRCP", "RLCP", "RPCL", "PCRL", "CPRL"

    Example:
        >>> # Combine series without HTJ2K conversion (uncompressed)
        >>> output_dir = convert_single_frame_dicom_series_to_multiframe("/path/to/dicoms")
        >>> print(f"Multi-frame files saved to: {output_dir}")

        >>> # Combine series with HTJ2K conversion
        >>> output_dir = convert_single_frame_dicom_series_to_multiframe(
        ...     "/path/to/dicoms",
        ...     convert_to_htj2k=True
        ... )

    Note:
        Each output file is named using the SeriesInstanceUID:
        <StudyUID>/<SeriesUID>.dcm

        The NumberOfFrames tag is set to the total frame count.
        All other DICOM metadata is preserved from the first instance in each series.

        Basic Offset Table:
        A Basic Offset Table is automatically created containing byte offsets to each frame.
        This allows DICOM readers to quickly locate and extract individual frames without
        parsing the entire encapsulated pixel data stream. The offsets are 32-bit unsigned
        integers measured from the first byte of the first Item Tag following the BOT.
    """
    import glob
    import shutil
    import tempfile
    from collections import defaultdict
    from pathlib import Path

    # Check for nvidia-nvimgcodec only if HTJ2K conversion is requested
    if convert_to_htj2k:
        try:
            from nvidia import nvimgcodec
        except ImportError:
            raise ImportError(
                "nvidia-nvimgcodec is required for HTJ2K transcoding. "
                "Install it with: pip install nvidia-nvimgcodec-cu{XX}[all] "
                "(replace {XX} with your CUDA version, e.g., cu13)"
            )

    import time

    import numpy as np
    import pydicom

    # Validate input
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Get all DICOM files recursively
    dicom_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".dcm") or file.endswith(".DCM"):
                dicom_files.append(os.path.join(root, file))

    # Also check for files without extension
    for pattern in ["*"]:
        found_files = glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
        for file_path in found_files:
            if os.path.isfile(file_path) and file_path not in dicom_files:
                try:
                    with open(file_path, "rb") as f:
                        f.seek(128)
                        magic = f.read(4)
                        if magic == b"DICM":
                            dicom_files.append(file_path)
                except Exception:
                    continue

    if not dicom_files:
        raise ValueError(f"No valid DICOM files found in {input_dir}")

    logger.info(f"Found {len(dicom_files)} DICOM files to process")

    # Group files by study and series
    series_groups = defaultdict(list)  # Key: (StudyUID, SeriesUID), Value: list of file paths

    logger.info("Grouping DICOM files by series...")
    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            study_uid = str(ds.StudyInstanceUID)
            series_uid = str(ds.SeriesInstanceUID)
            instance_number = int(getattr(ds, "InstanceNumber", 0))
            series_groups[(study_uid, series_uid)].append((instance_number, file_path))
        except Exception as e:
            logger.warning(f"Failed to read metadata from {file_path}: {e}")
            continue

    # Sort files within each series by InstanceNumber
    for key in series_groups:
        series_groups[key].sort(key=lambda x: x[0])  # Sort by instance number

    logger.info(f"Found {len(series_groups)} unique series")

    # Create output directory
    if output_dir is None:
        prefix = "htj2k_multiframe_" if convert_to_htj2k else "multiframe_"
        output_dir = tempfile.mkdtemp(prefix=prefix)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Setup encoder/decoder and parameters based on conversion mode
    if convert_to_htj2k:
        # Create encoder and decoder instances for HTJ2K
        encoder = _get_nvimgcodec_encoder()
        decoder = _get_nvimgcodec_decoder()

        # Setup HTJ2K encoding parameters
        encode_params, target_transfer_syntax = _setup_htj2k_encode_params(
            num_resolutions=num_resolutions, code_block_size=code_block_size, progression_order=progression_order
        )
        # Note: decode_params is created per-series based on PhotometricInterpretation
        logger.info("HTJ2K conversion enabled")
    else:
        # No conversion - preserve original transfer syntax
        encoder = None
        decoder = None
        encode_params = None
        target_transfer_syntax = None  # Will be determined from first dataset
        logger.info("Preserving original transfer syntax (no HTJ2K conversion)")

    # Get transfer syntax constants
    ts_constants = _get_transfer_syntax_constants()
    NVIMGCODEC_SYNTAXES = ts_constants["NVIMGCODEC"]

    start_time = time.time()
    processed_series = 0
    total_frames = 0

    # Process each series
    for (study_uid, series_uid), file_list in series_groups.items():
        try:
            logger.info(f"Processing series {series_uid} ({len(file_list)} instances)")

            # Load all datasets for this series
            file_paths = [fp for _, fp in file_list]
            datasets = [pydicom.dcmread(fp) for fp in file_paths]

            # CRITICAL: Sort datasets by ImagePositionPatient Z-coordinate
            # This ensures Frame[0] is the first slice, Frame[N] is the last slice
            if all(hasattr(ds, "ImagePositionPatient") for ds in datasets):
                # Sort by Z coordinate (3rd element of ImagePositionPatient)
                datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
                logger.info(f"  ✓ Sorted {len(datasets)} frames by ImagePositionPatient Z-coordinate")
                logger.info(f"    First frame Z: {datasets[0].ImagePositionPatient[2]}")
                logger.info(f"    Last frame Z:  {datasets[-1].ImagePositionPatient[2]}")

                # NOTE: We keep anatomically correct order (Z-ascending)
                # Cornerstone3D should use per-frame ImagePositionPatient from PerFrameFunctionalGroupsSequence
                # We provide complete per-frame metadata (PlanePositionSequence + PlaneOrientationSequence)
                logger.info(f"  ✓ Frames in anatomical order (lowest Z first)")
                logger.info(
                    f"    Cornerstone3D should use per-frame ImagePositionPatient for correct volume reconstruction"
                )
            else:
                logger.warning(f"  ⚠️  Some frames missing ImagePositionPatient, using file order")

            # Use first dataset as template
            template_ds = datasets[0]

            # Determine transfer syntax from first dataset
            if target_transfer_syntax is None:
                target_transfer_syntax = str(getattr(template_ds.file_meta, "TransferSyntaxUID", "1.2.840.10008.1.2.1"))
                logger.info(f"  Using original transfer syntax: {target_transfer_syntax}")

            # Check if we're dealing with encapsulated (compressed) data
            is_encapsulated = (
                hasattr(template_ds, "PixelData")
                and template_ds.file_meta.TransferSyntaxUID != pydicom.uid.ExplicitVRLittleEndian
            )

            # Determine color_spec for this series based on PhotometricInterpretation
            if convert_to_htj2k:
                photometric = getattr(template_ds, "PhotometricInterpretation", "UNKNOWN")
                if photometric.startswith("YBR"):
                    series_color_spec = nvimgcodec.ColorSpec.RGB
                    logger.info(f"  Series PhotometricInterpretation={photometric}, using color_spec=RGB")
                else:
                    series_color_spec = nvimgcodec.ColorSpec.UNCHANGED
                    logger.info(f"  Series PhotometricInterpretation={photometric}, using color_spec=UNCHANGED")
                series_decode_params = _setup_htj2k_decode_params(color_spec=series_color_spec)

            # Collect all frames from all instances
            all_frames = []  # Will contain either numpy arrays (for HTJ2K) or bytes (for preserving)

            if convert_to_htj2k:
                # HTJ2K mode: decode all frames
                for ds in datasets:
                    current_ts = str(getattr(ds.file_meta, "TransferSyntaxUID", None))

                    if current_ts in NVIMGCODEC_SYNTAXES:
                        # Compressed format - use nvimgcodec decoder
                        frames = [fragment for fragment in pydicom.encaps.generate_frames(ds.PixelData)]
                        decoded = decoder.decode(frames, params=series_decode_params)
                        all_frames.extend(decoded)
                    else:
                        # Uncompressed format - use pydicom
                        pixel_array = ds.pixel_array
                        if not isinstance(pixel_array, np.ndarray):
                            pixel_array = np.array(pixel_array)

                        # Handle single frame vs multi-frame
                        if pixel_array.ndim == 2:
                            all_frames.append(pixel_array)
                        elif pixel_array.ndim == 3:
                            for frame_idx in range(pixel_array.shape[0]):
                                all_frames.append(pixel_array[frame_idx, :, :])
            else:
                # Preserve original encoding: extract frames without decoding
                first_ts = str(getattr(datasets[0].file_meta, "TransferSyntaxUID", None))

                if first_ts in NVIMGCODEC_SYNTAXES or pydicom.encaps.encapsulate_extended:
                    # Encapsulated data - extract compressed frames
                    for ds in datasets:
                        if hasattr(ds, "PixelData"):
                            try:
                                # Extract compressed frames
                                frames = [fragment for fragment in pydicom.encaps.generate_frames(ds.PixelData)]
                                all_frames.extend(frames)
                            except:
                                # Fall back to pixel_array for uncompressed
                                pixel_array = ds.pixel_array
                                if not isinstance(pixel_array, np.ndarray):
                                    pixel_array = np.array(pixel_array)
                                if pixel_array.ndim == 2:
                                    all_frames.append(pixel_array)
                                elif pixel_array.ndim == 3:
                                    for frame_idx in range(pixel_array.shape[0]):
                                        all_frames.append(pixel_array[frame_idx, :, :])
                else:
                    # Uncompressed data - use pixel arrays
                    for ds in datasets:
                        pixel_array = ds.pixel_array
                        if not isinstance(pixel_array, np.ndarray):
                            pixel_array = np.array(pixel_array)
                        if pixel_array.ndim == 2:
                            all_frames.append(pixel_array)
                        elif pixel_array.ndim == 3:
                            for frame_idx in range(pixel_array.shape[0]):
                                all_frames.append(pixel_array[frame_idx, :, :])

            total_frame_count = len(all_frames)
            logger.info(f"  Total frames in series: {total_frame_count}")

            # Encode frames based on conversion mode
            if convert_to_htj2k:
                logger.info(f"  Encoding {total_frame_count} frames to HTJ2K...")
                # Ensure frames have channel dimension for encoder
                frames_for_encoding = []
                for frame in all_frames:
                    if frame.ndim == 2:
                        frame = frame[:, :, np.newaxis]
                    frames_for_encoding.append(frame)
                encoded_frames = encoder.encode(frames_for_encoding, codec="jpeg2k", params=encode_params)
                # Convert to bytes
                encoded_frames_bytes = [bytes(enc) for enc in encoded_frames]
            else:
                logger.info(f"  Preserving original encoding for {total_frame_count} frames...")
                # Check if frames are already bytes (encapsulated) or numpy arrays (uncompressed)
                if len(all_frames) > 0 and isinstance(all_frames[0], bytes):
                    # Already encapsulated - use as-is
                    encoded_frames_bytes = all_frames
                else:
                    # Uncompressed numpy arrays
                    encoded_frames_bytes = None

            # Create SIMPLE multi-frame DICOM file (like the user's example)
            # Use first dataset as template, keeping its metadata
            logger.info(f"  Creating simple multi-frame DICOM from {total_frame_count} frames...")
            output_ds = datasets[0].copy()  # Start from first dataset

            # CRITICAL: Set SOP Instance UID to match the SeriesInstanceUID (which will be the filename)
            # This ensures the file's internal SOP Instance UID matches its filename
            output_ds.SOPInstanceUID = series_uid

            # Update pixel data based on conversion mode
            if encoded_frames_bytes is not None:
                # Encapsulated data (HTJ2K or preserved compressed format)
                # Use Basic Offset Table for multi-frame efficiency
                output_ds.PixelData = pydicom.encaps.encapsulate(encoded_frames_bytes, has_bot=add_basic_offset_table)
            else:
                # Uncompressed mode: combine all frames into a 3D array
                # Stack frames: (frames, rows, cols)
                combined_pixel_array = np.stack(all_frames, axis=0)
                output_ds.PixelData = combined_pixel_array.tobytes()

            output_ds.file_meta.TransferSyntaxUID = pydicom.uid.UID(target_transfer_syntax)

            # Update PhotometricInterpretation if we converted from YBR to RGB
            if convert_to_htj2k and hasattr(output_ds, "PhotometricInterpretation"):
                original_pi = output_ds.PhotometricInterpretation
                if original_pi.startswith("YBR"):
                    output_ds.PhotometricInterpretation = "RGB"
                    logger.info(f"  Updated PhotometricInterpretation: {original_pi} -> RGB")

            # Set NumberOfFrames (critical!)
            output_ds.NumberOfFrames = total_frame_count

            # DICOM Multi-frame Module (C.7.6.6) - Mandatory attributes

            # FrameIncrementPointer - REQUIRED to tell viewers how frames are ordered
            # Points to ImagePositionPatient (0020,0032) which varies per frame
            output_ds.FrameIncrementPointer = 0x00200032
            logger.info(f"  ✓ Set FrameIncrementPointer to ImagePositionPatient")

            # Ensure all Image Pixel Module attributes are present (C.7.6.3)
            # These should be inherited from first frame, but verify:
            required_pixel_attrs = [
                ("SamplesPerPixel", 1),
                ("PhotometricInterpretation", "MONOCHROME2"),
                ("Rows", 512),
                ("Columns", 512),
            ]

            for attr, default in required_pixel_attrs:
                if not hasattr(output_ds, attr):
                    setattr(output_ds, attr, default)
                    logger.warning(f"  ⚠️  Added missing {attr} = {default}")

            # CRITICAL FIX #1: Remove top-level ImagePositionPatient
            # OHIF reads this for ALL frames instead of per-frame values → spacing[2] = 0
            if hasattr(output_ds, "ImagePositionPatient"):
                delattr(output_ds, "ImagePositionPatient")
                logger.info(f"  ✓ Removed top-level ImagePositionPatient (use per-frame only)")
            
            # CRITICAL FIX #2: Keep ImageOrientationPatient at top level
            # OHIF needs this to recognize file as MPR-capable
            # Safe to keep since orientation is the same for all frames
            if hasattr(datasets[0], "ImageOrientationPatient"):
                output_ds.ImageOrientationPatient = datasets[0].ImageOrientationPatient
                logger.info(f"  ✓ Kept top-level ImageOrientationPatient: {output_ds.ImageOrientationPatient}")
            
            # CRITICAL FIX #3: Set correct SOPClassUID for Enhanced multi-frame CT
            # Use Enhanced CT Image Storage (not legacy CT Image Storage)
            output_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2.1"  # Enhanced CT Image Storage
            logger.info(f"  ✓ Set SOPClassUID to Enhanced CT Image Storage")

            # Keep pixel spacing and slice thickness
            if hasattr(datasets[0], "PixelSpacing"):
                output_ds.PixelSpacing = datasets[0].PixelSpacing
                logger.info(f"  ✓ PixelSpacing: {output_ds.PixelSpacing}")

            if hasattr(datasets[0], "SliceThickness"):
                output_ds.SliceThickness = datasets[0].SliceThickness
                logger.info(f"  ✓ SliceThickness: {output_ds.SliceThickness}")

            # Fix InstanceNumber (should be >= 1)
            output_ds.InstanceNumber = 1

            # Ensure SeriesNumber is present
            if not hasattr(output_ds, "SeriesNumber"):
                output_ds.SeriesNumber = 1

            # Remove per-frame tags that conflict with multi-frame
            if hasattr(output_ds, "SliceLocation"):
                delattr(output_ds, "SliceLocation")
                logger.info(f"  ✓ Removed SliceLocation (per-frame tag)")

            # Add SpacingBetweenSlices
            if len(datasets) > 1:
                pos0 = datasets[0].ImagePositionPatient if hasattr(datasets[0], "ImagePositionPatient") else None
                pos1 = datasets[1].ImagePositionPatient if hasattr(datasets[1], "ImagePositionPatient") else None

                if pos0 and pos1:
                    # Calculate spacing as distance between consecutive slices
                    import math

                    spacing = math.sqrt(sum((float(pos1[i]) - float(pos0[i])) ** 2 for i in range(3)))
                    output_ds.SpacingBetweenSlices = spacing
                    logger.info(f"  ✓ Added SpacingBetweenSlices: {spacing:.6f} mm")

            # Add PerFrameFunctionalGroupsSequence for OHIF/Cornerstone3D compatibility
            # CRITICAL: Structure must be exactly right to avoid "1/Infinity" MPR display bug
            # - Per-frame: PlanePositionSequence ONLY (unique position per frame)
            # - Shared: PlaneOrientationSequence (common orientation for all frames)
            logger.info(f"  Adding per-frame functional groups (OHIF-compatible structure)...")
            from pydicom.dataset import Dataset as DicomDataset
            from pydicom.sequence import Sequence

            per_frame_seq = []
            for frame_idx, ds_frame in enumerate(datasets):
                frame_item = DicomDataset()

                # PlanePositionSequence - ImagePositionPatient for this frame
                # This is REQUIRED - each frame needs its own position
                if hasattr(ds_frame, "ImagePositionPatient"):
                    plane_pos_item = DicomDataset()
                    plane_pos_item.ImagePositionPatient = ds_frame.ImagePositionPatient
                    frame_item.PlanePositionSequence = Sequence([plane_pos_item])

                # CRITICAL: Do NOT add per-frame PlaneOrientationSequence!
                # PlaneOrientationSequence should ONLY be in SharedFunctionalGroupsSequence
                # Having it per-frame triggers different parsing logic in OHIF/Cornerstone3D
                # Result: metadata not read correctly, spacing[2] = 0
                # (The orientation is shared across all frames anyway)

                # FrameContentSequence - helps with frame identification
                frame_content_item = DicomDataset()
                frame_content_item.StackID = "1"
                frame_content_item.InStackPositionNumber = frame_idx + 1
                frame_content_item.DimensionIndexValues = [1, frame_idx + 1]
                frame_item.FrameContentSequence = Sequence([frame_content_item])

                per_frame_seq.append(frame_item)

            output_ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame_seq)
            logger.info(f"  ✓ Added PerFrameFunctionalGroupsSequence with {len(per_frame_seq)} frame items")
            logger.info(f"    Each frame includes: PlanePositionSequence only (orientation in shared)")

            # Add SharedFunctionalGroupsSequence for additional Cornerstone3D compatibility
            # This defines attributes that are common to ALL frames
            shared_item = DicomDataset()

            # PlaneOrientationSequence - same for all frames
            if hasattr(datasets[0], "ImageOrientationPatient"):
                shared_orient_item = DicomDataset()
                shared_orient_item.ImageOrientationPatient = datasets[0].ImageOrientationPatient
                shared_item.PlaneOrientationSequence = Sequence([shared_orient_item])

            # PixelMeasuresSequence - pixel spacing and slice thickness
            if hasattr(datasets[0], "PixelSpacing") or hasattr(datasets[0], "SliceThickness"):
                pixel_measures_item = DicomDataset()
                if hasattr(datasets[0], "PixelSpacing"):
                    pixel_measures_item.PixelSpacing = datasets[0].PixelSpacing
                if hasattr(datasets[0], "SliceThickness"):
                    pixel_measures_item.SliceThickness = datasets[0].SliceThickness
                if hasattr(output_ds, "SpacingBetweenSlices"):
                    pixel_measures_item.SpacingBetweenSlices = output_ds.SpacingBetweenSlices
                shared_item.PixelMeasuresSequence = Sequence([pixel_measures_item])

            output_ds.SharedFunctionalGroupsSequence = Sequence([shared_item])
            logger.info(f"  ✓ Added SharedFunctionalGroupsSequence (common attributes for all frames)")
            logger.info(f"    Includes PlaneOrientationSequence (ONLY location for orientation!)")

            # Verify frame ordering
            if len(per_frame_seq) > 0:
                first_frame_pos = (
                    per_frame_seq[0].PlanePositionSequence[0].ImagePositionPatient
                    if hasattr(per_frame_seq[0], "PlanePositionSequence")
                    else None
                )
                last_frame_pos = (
                    per_frame_seq[-1].PlanePositionSequence[0].ImagePositionPatient
                    if hasattr(per_frame_seq[-1], "PlanePositionSequence")
                    else None
                )

                if first_frame_pos and last_frame_pos:
                    logger.info(f"  ✓ Frame ordering verification:")
                    logger.info(f"    Frame[0] Z = {first_frame_pos[2]} (should match top-level)")
                    logger.info(f"    Frame[{len(per_frame_seq)-1}] Z = {last_frame_pos[2]} (last slice)")

                    # Verify top-level matches Frame[0]
                    if hasattr(output_ds, "ImagePositionPatient"):
                        if abs(float(output_ds.ImagePositionPatient[2]) - float(first_frame_pos[2])) < 0.001:
                            logger.info(f"    ✅ Top-level ImagePositionPatient matches Frame[0]")
                        else:
                            logger.error(
                                f"    ❌ MISMATCH: Top-level Z={output_ds.ImagePositionPatient[2]} != Frame[0] Z={first_frame_pos[2]}"
                            )

            logger.info(f"  ✓ Created multi-frame with {total_frame_count} frames (OHIF-compatible)")
            if encoded_frames_bytes is not None:
                logger.info(f"  ✓ Basic Offset Table included for efficient frame access")

            # Create output directory structure
            study_output_dir = os.path.join(output_dir, study_uid)
            os.makedirs(study_output_dir, exist_ok=True)

            # Save as single multi-frame file
            output_file = os.path.join(study_output_dir, f"{series_uid}.dcm")
            output_ds.save_as(output_file, write_like_original=False)

            logger.info(f"  ✓ Saved multi-frame file: {output_file}")
            processed_series += 1
            total_frames += total_frame_count

        except Exception as e:
            logger.error(f"Failed to process series {series_uid}: {e}")
            import traceback

            traceback.print_exc()
            continue

    elapsed_time = time.time() - start_time

    if convert_to_htj2k:
        logger.info(f"\nMulti-frame HTJ2K conversion complete:")
    else:
        logger.info(f"\nMulti-frame DICOM conversion complete:")
    logger.info(f"  Total series processed: {processed_series}")
    logger.info(f"  Total frames combined: {total_frames}")
    if convert_to_htj2k:
        logger.info(f"  Format: HTJ2K compressed")
    else:
        logger.info(f"  Format: Original transfer syntax preserved")
    logger.info(f"  Time elapsed: {elapsed_time:.2f} seconds")
    logger.info(f"  Output directory: {output_dir}")

    return output_dir
