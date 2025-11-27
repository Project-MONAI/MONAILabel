# Copyright 2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for converting legacy DICOM series to enhanced multi-frame format.

This module provides tools to convert series of single-frame DICOM files into
single multi-frame enhanced DICOM files, with optional HTJ2K compression for
improved storage efficiency.

Key Features:
- Convert legacy CT/MR/PT series to enhanced multi-frame format using highdicom
- Optional HTJ2K (High-Throughput JPEG 2000) lossless compression
- Batch processing of multiple series with automatic grouping by SeriesInstanceUID
- Preserve or generate new SeriesInstanceUID
- Handle unsupported modalities (MG, US, XA) by transcoding or copying
- Comprehensive statistics including frame counts and compression ratios

Enhanced DICOM multi-frame format benefits:
- Single file instead of hundreds of individual files
- Better organization and metadata structure
- More efficient I/O operations
- Standards-compliant with DICOM Part 3

Supported modalities for enhanced conversion:
- CT (Computed Tomography)
- MR (Magnetic Resonance)
- PT (Positron Emission Tomography)

Unsupported modalities (MG, US, XA, etc.) can be:
- Transcoded to HTJ2K (preserving original format)
- Copied without modification

Example:
    >>> from monailabel.datastore.utils.convert_multiframe import (
    ...     convert_to_enhanced_dicom,
    ...     batch_convert_by_series,
    ...     convert_and_convert_to_htj2k,
    ... )
    >>> 
    >>> # Single series conversion (preserves original SeriesInstanceUID by default)
    >>> convert_to_enhanced_dicom(
    ...     input_source="/path/to/legacy/ct/series",
    ...     output_file="/path/to/output/enhanced.dcm"
    ... )
    >>> 
    >>> # Convert with HTJ2K compression
    >>> convert_and_convert_to_htj2k(
    ...     input_source="/path/to/legacy/ct/series",
    ...     output_file="/path/to/output/enhanced_htj2k.dcm",
    ...     num_resolutions=6
    ... )
    >>> 
    >>> # Batch convert multiple series with HTJ2K
    >>> import pydicom
    >>> from pathlib import Path
    >>> 
    >>> # Collect DICOM files
    >>> input_dir = Path("/path/to/mixed/dicoms")
    >>> input_files = [str(f) for f in input_dir.rglob("*.dcm")]
    >>> 
    >>> # Create file_loader
    >>> file_loader = [(input_files, "/path/to/output")]
    >>> 
    >>> # Batch convert
    >>> stats = batch_convert_by_series(
    ...     file_loader=file_loader,
    ...     compress_htj2k=True,
    ...     num_resolutions=6
    ... )
    >>> print(f"Processed {stats['total_frames_input']} frames")
    >>> print(f"Converted {stats['converted_to_multiframe']} series to multi-frame")
"""

import logging
import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np
import pydicom
from pydicom.uid import generate_uid

logger = logging.getLogger(__name__)

# Constants for DICOM modalities
SUPPORTED_MODALITIES = {"CT", "MR", "PT"}

# Transfer syntax UIDs
EXPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2.1"
IMPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2"


def _check_highdicom_available():
    """Check if highdicom is installed."""
    try:
        import highdicom
        return True
    except ImportError:
        return False


@contextmanager
def _suppress_highdicom_warnings():
    """
    Context manager to suppress common highdicom warnings.
    
    Suppresses warnings like:
    - "unknown derived pixel contrast"
    - Other non-critical highdicom warnings
    
    This suppresses both Python warnings and logging-based warnings from highdicom.
    """
    # Suppress Python warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*unknown derived pixel contrast.*')
        warnings.filterwarnings('ignore', category=UserWarning, module='highdicom.*')
        
        # Suppress highdicom logging warnings
        highdicom_logger = logging.getLogger('highdicom')
        highdicom_legacy_logger = logging.getLogger('highdicom.legacy')
        highdicom_sop_logger = logging.getLogger('highdicom.legacy.sop')
        
        # Save original log levels
        original_level = highdicom_logger.level
        original_legacy_level = highdicom_legacy_logger.level
        original_sop_level = highdicom_sop_logger.level
        
        try:
            # Temporarily set to ERROR to suppress WARNING messages
            highdicom_logger.setLevel(logging.ERROR)
            highdicom_legacy_logger.setLevel(logging.ERROR)
            highdicom_sop_logger.setLevel(logging.ERROR)
            yield
        finally:
            # Restore original log levels
            highdicom_logger.setLevel(original_level)
            highdicom_legacy_logger.setLevel(original_legacy_level)
            highdicom_sop_logger.setLevel(original_sop_level)


def _load_dicom_series(input_source: Union[str, Path, List[Union[str, Path]]]) -> List[pydicom.Dataset]:
    """
    Load DICOM files from a directory or list of file paths and sort them by spatial position.

    Args:
        input_source: Either:
            - Directory path containing DICOM files
            - List of DICOM file paths

    Returns:
        List of sorted pydicom.Dataset objects

    Raises:
        ValueError: If no DICOM files found or files have inconsistent metadata
    """
    # Handle different input types
    if isinstance(input_source, (list, tuple)):
        # List of file paths provided
        file_paths = [Path(f) for f in input_source]
        source_desc = f"{len(file_paths)} provided files"
    else:
        # Directory path provided
        input_dir = Path(input_source)
        if not input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")
        
        # Find all DICOM files in directory
        file_paths = [f for f in input_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
        source_desc = f"directory {input_dir}"
    
    # Load DICOM files
    dicom_files = []
    for filepath in file_paths:
        try:
            ds = pydicom.dcmread(filepath)
            dicom_files.append(ds)
        except Exception as e:
            logger.debug(f"Skipping non-DICOM file {filepath.name}: {e}")
            continue
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {source_desc}")
    
    logger.info(f"Loaded {len(dicom_files)} DICOM files from {source_desc}")
    
    # Sort by ImagePositionPatient if available
    if all(hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'ImageOrientationPatient') 
           for ds in dicom_files):
        # Calculate distance along normal vector for each slice
        first_ds = dicom_files[0]
        orientation = np.array(first_ds.ImageOrientationPatient).reshape(2, 3)
        normal = np.cross(orientation[0], orientation[1])
        
        def get_position_along_normal(ds):
            position = np.array(ds.ImagePositionPatient)
            return np.dot(position, normal)
        
        dicom_files.sort(key=get_position_along_normal)
        logger.info("Sorted files by spatial position")
    elif all(hasattr(ds, 'InstanceNumber') for ds in dicom_files):
        # Fall back to InstanceNumber
        dicom_files.sort(key=lambda ds: ds.InstanceNumber)
        logger.info("Sorted files by InstanceNumber")
    else:
        logger.warning("Could not determine optimal sorting order, using file order")
    
    return dicom_files


def _validate_series_consistency(datasets: List[pydicom.Dataset]) -> dict:
    """
    Validate that all datasets in a series are consistent.

    Args:
        datasets: List of pydicom.Dataset objects

    Returns:
        Dictionary with series metadata

    Raises:
        ValueError: If datasets are inconsistent
    """
    if not datasets:
        raise ValueError("Empty dataset list")
    
    first_ds = datasets[0]
    
    # Check modality
    modality = getattr(first_ds, 'Modality', None)
    if not modality:
        raise ValueError("First dataset missing Modality tag")
    
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(
            f"Unsupported modality: {modality}. "
            f"Supported modalities are: {', '.join(SUPPORTED_MODALITIES)}"
        )
    
    # Required attributes that must be consistent
    required_attrs = ['Rows', 'Columns', 'Modality']
    optional_consistent_attrs = [
        'SeriesInstanceUID', 'StudyInstanceUID', 'PatientID',
        'PixelSpacing', 'ImageOrientationPatient'
    ]
    
    # Collect metadata from first dataset
    metadata = {
        'modality': modality,
        'rows': first_ds.Rows,
        'columns': first_ds.Columns,
        'num_frames': len(datasets),
    }
    
    # Check consistency across all datasets
    for attr in required_attrs:
        if not all(hasattr(ds, attr) and getattr(ds, attr) == getattr(first_ds, attr) 
                   for ds in datasets):
            raise ValueError(f"Inconsistent {attr} values across series")
    
    # Collect optional metadata
    for attr in optional_consistent_attrs:
        if hasattr(first_ds, attr):
            metadata[attr.lower()] = getattr(first_ds, attr)
    
    logger.info(
        f"Series validated: {modality} {metadata['rows']}x{metadata['columns']}, "
        f"{metadata['num_frames']} frames"
    )
    
    return metadata


def _fix_dicom_datetime_attributes(datasets: List[pydicom.Dataset]) -> None:
    """
    Fix malformed date/time attributes in DICOM datasets.
    
    Some legacy DICOM files have date/time values stored as strings in non-standard
    formats. This function converts valid date strings to proper Python date objects
    and removes invalid ones. This is necessary because highdicom expects proper
    date/time objects, not strings.
    
    Args:
        datasets: List of pydicom.Dataset objects to modify in-place
    """
    from datetime import datetime, date, time
    
    fixed_attrs = set()
    
    for ds in datasets:
        # List of date/time attributes that might need fixing
        date_attrs = ['StudyDate', 'SeriesDate', 'AcquisitionDate', 'ContentDate']
        time_attrs = ['StudyTime', 'SeriesTime', 'AcquisitionTime', 'ContentTime']
        
        # Fix date attributes - convert strings to date objects
        for attr in date_attrs:
            if hasattr(ds, attr):
                value = getattr(ds, attr)
                # If it's already a proper date/datetime object, skip
                if isinstance(value, (date, datetime)):
                    continue
                # If it's a string, try to convert it to a date object
                if isinstance(value, str) and value:
                    try:
                        # DICOM date format is YYYYMMDD
                        if len(value) >= 8 and value[:8].isdigit():
                            year = int(value[0:4])
                            month = int(value[4:6])
                            day = int(value[6:8])
                            date_obj = date(year, month, day)
                            setattr(ds, attr, date_obj)
                            fixed_attrs.add(f"{attr} (converted to date)")
                        else:
                            # Invalid format, remove it
                            delattr(ds, attr)
                            fixed_attrs.add(f"{attr} (removed)")
                    except (ValueError, IndexError) as e:
                        # Invalid date values, remove it
                        delattr(ds, attr)
                        fixed_attrs.add(f"{attr} (removed - invalid)")
                elif not value:
                    # Empty string, remove it
                    delattr(ds, attr)
                    fixed_attrs.add(f"{attr} (removed - empty)")
        
        # Fix time attributes - convert strings to time objects
        for attr in time_attrs:
            if hasattr(ds, attr):
                value = getattr(ds, attr)
                # If it's already a proper time/datetime object, skip
                if isinstance(value, (time, datetime)):
                    continue
                # If it's a string, try to convert it to a time object
                if isinstance(value, str) and value:
                    try:
                        # DICOM time format is HHMMSS.FFFFFF or HHMMSS
                        # Clean up the string
                        time_str = value.replace(':', '')
                        
                        if '.' in time_str:
                            parts = time_str.split('.')
                            main_part = parts[0]
                            frac_part = parts[1] if len(parts) > 1 else '0'
                        else:
                            main_part = time_str
                            frac_part = '0'
                        
                        # Parse hours, minutes, seconds
                        if len(main_part) >= 2:
                            hour = int(main_part[0:2])
                            minute = int(main_part[2:4]) if len(main_part) >= 4 else 0
                            second = int(main_part[4:6]) if len(main_part) >= 6 else 0
                            microsecond = int(frac_part[:6].ljust(6, '0')) if frac_part else 0
                            
                            time_obj = time(hour, minute, second, microsecond)
                            setattr(ds, attr, time_obj)
                            fixed_attrs.add(f"{attr} (converted to time)")
                        else:
                            # Too short to be valid, remove it
                            delattr(ds, attr)
                            fixed_attrs.add(f"{attr} (removed)")
                    except (ValueError, IndexError) as e:
                        # Invalid time values, remove it
                        delattr(ds, attr)
                        fixed_attrs.add(f"{attr} (removed - invalid)")
                elif not value:
                    # Empty string, remove it
                    delattr(ds, attr)
                    fixed_attrs.add(f"{attr} (removed - empty)")
    
    if fixed_attrs:
        logger.info(
            f"Converted/fixed date/time attributes: {len([a for a in fixed_attrs if 'converted' in a])} converted, "
            f"{len([a for a in fixed_attrs if 'removed' in a])} removed"
        )


def _ensure_required_attributes(datasets: List[pydicom.Dataset]) -> None:
    """
    Ensure that all datasets have the required attributes for enhanced multi-frame conversion.
    
    If required attributes are missing, they are added with sensible default values.
    This is necessary because the DICOM enhanced multi-frame standard requires certain
    attributes that may be missing from legacy DICOM files.
    
    Args:
        datasets: List of pydicom.Dataset objects to modify in-place
    """
    # Required attributes and their default values
    required_attrs = {
        'Manufacturer': 'Unknown',
        'ManufacturerModelName': 'Unknown',
        'DeviceSerialNumber': 'Unknown',
        'SoftwareVersions': 'Unknown',
    }
    
    # Check and add missing attributes to all datasets
    added_attrs = set()
    for ds in datasets:
        for attr, default_value in required_attrs.items():
            if not hasattr(ds, attr):
                setattr(ds, attr, default_value)
                added_attrs.add(attr)
    
    if added_attrs:
        logger.info(
            f"Added missing required attributes with default values: {', '.join(sorted(added_attrs))}"
        )


def _transcode_files_to_htj2k(
    file_paths: List[Path],
    output_dir: Path,
    compression_kwargs: dict,
) -> tuple[bool, float]:
    """
    Transcode DICOM files to HTJ2K format (helper function).
    
    This function handles HTJ2K transcoding for files that cannot be converted to
    enhanced multi-frame format (e.g., unsupported modalities like MG, US, XA).
    The original file format is preserved, only the pixel data is compressed with HTJ2K.
    
    Args:
        file_paths: List of input DICOM file paths to transcode
        output_dir: Output directory for transcoded files
        compression_kwargs: Dictionary with HTJ2K compression parameters:
            - num_resolutions (int): Wavelet decomposition levels (default: 6)
            - code_block_size (tuple): Code block size (default: (64, 64))
            - progression_order (str): JPEG2K progression order (default: 'RPCL')
    
    Returns:
        Tuple of (success: bool, output_size_mb: float):
        - success: True if transcoding succeeded, False otherwise
        - output_size_mb: Total size of transcoded files in MB (0.0 if failed)
    """
    try:
        from monailabel.datastore.utils.convert_htj2k import transcode_dicom_to_htj2k
        
        # Prepare file pairs for transcoding (input -> output)
        input_files = []
        output_files = []
        
        for file_path in file_paths:
            output_path = output_dir / file_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            input_files.append(str(file_path))
            output_files.append(str(output_path))
        
        # Transcode with HTJ2K
        file_loader = [(input_files, output_files)]
        num_resolutions = compression_kwargs.get('num_resolutions', 6)
        code_block_size = compression_kwargs.get('code_block_size', (64, 64))
        progression_order = compression_kwargs.get('progression_order', 'RPCL')
        
        transcode_dicom_to_htj2k(
            file_loader=file_loader,
            num_resolutions=num_resolutions,
            code_block_size=code_block_size,
            progression_order=progression_order,
        )
        
        # Calculate output size
        transcoded_size = sum(Path(f).stat().st_size for f in output_files if Path(f).exists())
        transcoded_size_mb = transcoded_size / (1024 * 1024)
        
        return True, transcoded_size_mb
        
    except Exception as e:
        logger.error(f"HTJ2K transcoding failed: {e}")
        return False, 0.0


def _copy_files(
    file_paths: List[Path],
    output_dir: Path,
) -> int:
    """
    Copy DICOM files to output directory.
    
    Args:
        file_paths: List of input file paths
        output_dir: Output directory
    
    Returns:
        Number of files successfully copied
    """
    copied_count = 0
    for file_path in file_paths:
        try:
            output_path = output_dir / file_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, output_path)
            copied_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {file_path.name}: {e}")
    
    return copied_count


def convert_to_enhanced_dicom(
    input_source: Union[str, Path, List[Union[str, Path]]],
    output_file: Union[str, Path],
    transfer_syntax_uid: Optional[str] = None,
    validate_only: bool = False,
    preserve_series_uid: bool = True,
    show_stats: bool = True,
) -> bool:
    """
    Convert legacy DICOM series to enhanced multi-frame DICOM.

    Args:
        input_source: Either:
            - Directory path containing legacy DICOM files (single-frame per file)
            - List of DICOM file paths to convert as a single series
        output_file: Path for output enhanced DICOM file
        transfer_syntax_uid: Transfer syntax for output. If None, uses Explicit VR Little Endian.
        validate_only: If True, only validate series without creating output file.
        preserve_series_uid: If True (default), preserve the original SeriesInstanceUID from
            the legacy datasets. If False, generate a new SeriesInstanceUID for the enhanced series.
        show_stats: If True (default), display conversion statistics. Set to False to suppress output.

    Returns:
        True if successful, False otherwise

    Raises:
        ImportError: If highdicom is not installed
        ValueError: If series is invalid or inconsistent
        FileNotFoundError: If input directory doesn't exist

    Example:
        >>> # Convert CT series from directory
        >>> convert_to_enhanced_dicom(
        ...     input_source="./ct_series/",
        ...     output_file="./enhanced_ct.dcm"
        ... )
        
        >>> # Convert from list of files (file_loader pattern)
        >>> file_paths = ['/data/ct_001.dcm', '/data/ct_002.dcm', '/data/ct_003.dcm']
        >>> convert_to_enhanced_dicom(
        ...     input_source=file_paths,
        ...     output_file="./enhanced_ct.dcm"
        ... )
        
        >>> # Convert with specific transfer syntax
        >>> convert_to_enhanced_dicom(
        ...     input_source="./mr_series/",
        ...     output_file="./enhanced_mr.dcm",
        ...     transfer_syntax_uid="1.2.840.10008.1.2.4.202"  # HTJ2K
        ... )
    """
    if not _check_highdicom_available():
        raise ImportError(
            "highdicom is not installed. Install it with: pip install highdicom"
        )
    
    import highdicom
    from highdicom.legacy import (
        LegacyConvertedEnhancedCTImage,
        LegacyConvertedEnhancedMRImage,
        LegacyConvertedEnhancedPETImage,
    )
    
    output_file = Path(output_file)
    
    # Set default transfer syntax
    if transfer_syntax_uid is None:
        transfer_syntax_uid = EXPLICIT_VR_LITTLE_ENDIAN
    
    # Describe input source for logging
    if isinstance(input_source, (list, tuple)):
        input_desc = f"{len(input_source)} files"
    else:
        input_desc = str(Path(input_source))
    
    logger.info(f"Converting legacy DICOM series to enhanced multi-frame format")
    logger.info(f"  Input: {input_desc}")
    if not validate_only:
        logger.info(f"  Output: {output_file}")
    logger.info(f"  Transfer Syntax: {transfer_syntax_uid}")
    
    try:
        # Load and sort DICOM files
        datasets = _load_dicom_series(input_source)
        
        # Validate consistency
        metadata = _validate_series_consistency(datasets)
        detected_modality = metadata['modality']

        if validate_only:
            logger.info("Validation successful (validate_only=True, not creating output file)")
            return True
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract SeriesInstanceUID from legacy datasets (preserve original if requested)
        # This maintains traceability between legacy and enhanced series
        original_series_uid = metadata.get('seriesinstanceuid')
        if preserve_series_uid and original_series_uid:
            series_uid = original_series_uid
            logger.info(f"Preserving original SeriesInstanceUID: {series_uid}")
        else:
            series_uid = generate_uid()
            if preserve_series_uid and not original_series_uid:
                logger.warning("SeriesInstanceUID not found in legacy datasets, generating new UID")
            logger.info(f"Generated new SeriesInstanceUID: {series_uid}")
        
        # Extract SeriesNumber and InstanceNumber from legacy datasets (use original if available)
        # Convert to native Python int (highdicom requires Python int, not pydicom IS/DS types)
        first_ds = datasets[0]
        series_number = int(getattr(first_ds, 'SeriesNumber', 1))
        if series_number < 1:
            logger.warning(f"SeriesNumber was {series_number}, using default value: 1")
            series_number = 1
        instance_number = int(getattr(first_ds, 'InstanceNumber', 1))
        if instance_number < 1:
            logger.warning(f"InstanceNumber was {instance_number}, using default value: 1")
            instance_number = 1

        # Note: highdicom's LegacyConverted* classes automatically preserve other important
        # metadata from the legacy datasets including:
        # - StudyInstanceUID
        # - PatientID, PatientName, PatientBirthDate, PatientSex
        # - StudyDate, StudyTime, StudyDescription
        # - Pixel spacing, slice spacing, image orientation/position
        # - And many other standard DICOM attributes
        
        # Fix any malformed date/time attributes that might cause issues
        _fix_dicom_datetime_attributes(datasets)
        
        # Add missing required attributes with default values if needed
        # The enhanced multi-frame DICOM standard requires these attributes
        _ensure_required_attributes(datasets)
        
        # Convert based on modality
        logger.info(f"Converting {detected_modality} series with {len(datasets)} frames...")
        
        # Generate a NEW SOP Instance UID for the enhanced multi-frame DICOM
        # Note: We do NOT use an original SOP Instance UID because:
        # 1. This is a new DICOM instance (different SOP Class)
        # 2. We're combining multiple instances (each with their own SOP Instance UID) into one
        # 3. DICOM standard requires each instance to have a unique identifier
        new_sop_instance_uid = generate_uid()
        
        # Suppress common highdicom warnings during conversion
        with _suppress_highdicom_warnings():
            if detected_modality == "CT":
                enhanced = LegacyConvertedEnhancedCTImage(
                    legacy_datasets=datasets,
                    series_instance_uid=series_uid,
                    series_number=series_number,
                    sop_instance_uid=new_sop_instance_uid,
                    instance_number=instance_number,
                )
            elif detected_modality == "MR":
                enhanced = LegacyConvertedEnhancedMRImage(
                    legacy_datasets=datasets,
                    series_instance_uid=series_uid,
                    series_number=series_number,
                    sop_instance_uid=new_sop_instance_uid,
                    instance_number=instance_number,
                )
            elif detected_modality == "PT":
                enhanced = LegacyConvertedEnhancedPETImage(
                    legacy_datasets=datasets,
                    series_instance_uid=series_uid,
                    series_number=series_number,
                    sop_instance_uid=new_sop_instance_uid,
                    instance_number=instance_number,
                )
            else:
                raise ValueError(f"Unsupported modality: {detected_modality}")
        
        # Set transfer syntax
        enhanced.file_meta.TransferSyntaxUID = transfer_syntax_uid
        
        # Save the enhanced DICOM file
        enhanced.save_as(str(output_file), enforce_file_format=False)

        # Calculate statistics
        output_size_bytes = output_file.stat().st_size
        output_size_mb = output_size_bytes / (1024 * 1024)
        
        # Calculate original combined size
        original_size_bytes = 0
        for ds in datasets:
            if hasattr(ds, 'filename') and ds.filename:
                try:
                    original_size_bytes += Path(ds.filename).stat().st_size
                except Exception:
                    pass
        
        original_size_mb = original_size_bytes / (1024 * 1024)
        
        # Calculate compression statistics
        if original_size_bytes > 0:
            compression_ratio = original_size_bytes / output_size_bytes
            size_reduction_pct = ((original_size_bytes - output_size_bytes) / original_size_bytes) * 100
        else:
            compression_ratio = 0.0
            size_reduction_pct = 0.0
        
        # Display results (only if show_stats is True)
        if show_stats:
            logger.info(f"✓ Successfully created enhanced DICOM file: {output_file}")
            logger.info(f"")
            logger.info(f"  Statistics:")
            logger.info(f"    Original files:  {len(datasets)} files, {original_size_mb:.2f} MB")
            logger.info(f"    Output file:     1 file, {output_size_mb:.2f} MB")
            if original_size_bytes > 0:
                if output_size_bytes < original_size_bytes:
                    logger.info(f"    Size reduction:  {size_reduction_pct:.1f}% smaller")
                    logger.info(f"    Compression:     {compression_ratio:.2f}x")
                else:
                    size_increase_pct = ((output_size_bytes - original_size_bytes) / original_size_bytes) * 100
                    logger.info(f"    Size increase:   {size_increase_pct:.1f}% larger")
                    logger.info(f"    Ratio:           {1/compression_ratio:.2f}x")
            logger.info(f"")
            logger.info(f"  Image info:")
            logger.info(f"    Frames:          {len(datasets)}")
            logger.info(f"    Dimensions:      {metadata['rows']}x{metadata['columns']}")
        
        return True
        
    except AttributeError as e:
        logger.error(
            f"Failed to convert DICOM series - missing required DICOM attribute: {e}\n"
            f"The legacy DICOM files may be missing required attributes such as:\n"
            f"  - Manufacturer\n"
            f"  - ManufacturerModelName\n"
            f"  - SoftwareVersions\n"
            f"These attributes are required by the DICOM enhanced multi-frame standard.",
            exc_info=True
        )
        return False
    except Exception as e:
        logger.error(f"Failed to convert DICOM series: {e}", exc_info=True)
        return False


def validate_dicom_series(input_source: Union[str, Path, List[Union[str, Path]]]) -> bool:
    """
    Validate that a DICOM series can be converted to enhanced format.

    Args:
        input_source: Either directory path or list of DICOM file paths

    Returns:
        True if series is valid, False otherwise

    Example:
        >>> # Validate from directory
        >>> if validate_dicom_series("./my_series/"):
        ...     print("Series is ready for conversion")
        >>> 
        >>> # Validate from file list
        >>> files = ['/data/ct_001.dcm', '/data/ct_002.dcm']
        >>> if validate_dicom_series(files):
        ...     print("Files are ready for conversion")
    """
    try:
        return convert_to_enhanced_dicom(
            input_source=input_source,
            output_file="dummy.dcm",  # Not used in validate mode
            validate_only=True,
        )
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def batch_convert_by_series(
    file_loader: Iterable[tuple[list[str], str]],
    preserve_series_uid: bool = True,
    compress_htj2k: bool = False,
    **compression_kwargs,
) -> dict:
    """
    Group DICOM files by SeriesInstanceUID and convert each series to enhanced multi-frame.
    
    This function automatically detects all unique DICOM series from the provided files
    and converts each series to a separate enhanced multi-frame file. Useful when you have
    multiple series mixed together.
    
    Output filenames are automatically generated based on metadata:
    - Format: {Modality}_{SeriesInstanceUID}.dcm
    - Examples:
        - CT_1.2.826.0.1.3680043.8.274.1.1.8323329.686521.1629744176.620266.dcm
        - MR_1.2.840.113619.2.55.3.123456789.101.20231127.143052.0.dcm
        - PT_1.3.12.2.1107.5.1.4.12345.30000023110710323456789.dcm
    
    Unsupported modalities (e.g., MG, US, XA) cannot be converted to enhanced multi-frame
    format, but are still processed:
    - If compress_htj2k=True: Transcoded to HTJ2K (compressed, original format preserved)
    - If compress_htj2k=False: Copied without modification
    Original subdirectory structure is preserved in both cases.
    
    Args:
        file_loader:
            Iterable of (input_files, output_dir) tuples, where:
                - input_files: List[str] of input DICOM file paths to process
                - output_dir: str output directory path for this batch
            
            Each yielded tuple specifies a batch of files to scan and the output directory.
            Files from all batches will be grouped by SeriesInstanceUID before conversion.
            
            Example:
                >>> # Simple usage with one batch
                >>> file_loader = [
                ...     (['/data/ct_001.dcm', '/data/ct_002.dcm', '/data/mr_001.dcm'], '/output')
                ... ]
                >>> stats = batch_convert_by_series(file_loader)
                
                >>> # Multiple batches from different sources
                >>> file_loader = [
                ...     (['/data1/file1.dcm', '/data1/file2.dcm'], '/output'),
                ...     (['/data2/file3.dcm', '/data2/file4.dcm'], '/output'),
                ... ]
                >>> stats = batch_convert_by_series(file_loader)
        preserve_series_uid: If True, preserve original SeriesInstanceUID
        compress_htj2k: If True, compress output with HTJ2K
        **compression_kwargs: Additional HTJ2K compression arguments (if compress_htj2k=True)
    
    Returns:
        Dictionary with conversion statistics:
        - 'total_series_input': Total number of unique series found
        - 'total_series_output': Total number of series in output
        - 'total_frames_input': Total number of frames (files) processed
        - 'total_frames_output': Total number of frames in output
        - 'total_size_input_mb': Total size of all input files in MB
        - 'total_size_output_mb': Total size of all output files in MB
        - 'converted_to_multiframe': Number of series converted to enhanced multi-frame
        - 'transcoded_htj2k': Number of series transcoded to HTJ2K (unsupported modalities)
        - 'copied': Number of series copied without compression (unsupported modalities)
        - 'failed': Number of failed conversions
        - 'series_info': List of dicts with per-series information
    
    Example:
        >>> # Collect DICOM files from directory
        >>> from pathlib import Path
        >>> import pydicom
        >>> 
        >>> input_dir = Path('/data/mixed_dicoms')
        >>> input_files = []
        >>> for filepath in input_dir.rglob('*.dcm'):
        ...     try:
        ...         pydicom.dcmread(filepath, stop_before_pixels=True)
        ...         input_files.append(str(filepath))
        ...     except:
        ...         pass  # Skip non-DICOM files
        >>> 
        >>> # Convert all series (uncompressed)
        >>> file_loader = [(input_files, '/output')]
        >>> stats = batch_convert_by_series(file_loader)
        >>> print(f"Processed {stats['total_frames_input']} frames from {stats['total_series_input']} series")
        >>> print(f"Converted {stats['converted_to_multiframe']} series to multi-frame")
        >>> print(f"Compression: {stats['total_size_input_mb'] / stats['total_size_output_mb']:.2f}x")
        
        >>> # Convert with HTJ2K compression
        >>> file_loader = [(input_files, '/output')]
        >>> stats = batch_convert_by_series(
        ...     file_loader=file_loader,
        ...     compress_htj2k=True,
        ...     num_resolutions=6,
        ...     code_block_size=(64, 64),
        ...     progression_order='RPCL'
        ... )
        >>> print(f"HTJ2K compressed: {stats['converted_to_multiframe']} series")
        >>> print(f"Transcoded only: {stats['transcoded_htj2k']} series (unsupported modalities)")
    """
    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info(f"Batch Converting DICOM Series")
    logger.info(f"{'='*80}")
    logger.info(f"  HTJ2K compression: {'Yes' if compress_htj2k else 'No'}")
    logger.info(f"")
    
    # Step 1: Collect all files from file_loader and group by SeriesInstanceUID
    logger.info("Step 1: Scanning files and grouping by SeriesInstanceUID...")
    series_files = {}  # Maps SeriesInstanceUID -> list of file paths
    series_metadata = {}  # Maps SeriesInstanceUID -> metadata dict
    series_output_dirs = {}  # Maps SeriesInstanceUID -> output directory
    output_dirs_set = set()  # Track all output directories
    
    total_files_scanned = 0
    for input_files, output_dir_str in file_loader:
        output_dir = Path(output_dir_str)
        output_dirs_set.add(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for filepath_str in input_files:
            filepath = Path(filepath_str)
            total_files_scanned += 1
            
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                series_uid = getattr(ds, 'SeriesInstanceUID', None)
                
                if series_uid:
                    if series_uid not in series_files:
                        series_files[series_uid] = []
                        series_output_dirs[series_uid] = output_dir
                        # Store metadata from first file
                        series_metadata[series_uid] = {
                            'modality': getattr(ds, 'Modality', 'Unknown'),
                            'series_number': getattr(ds, 'SeriesNumber', 'N/A'),
                            'series_description': getattr(ds, 'SeriesDescription', 'N/A'),
                            'patient_id': getattr(ds, 'PatientID', 'N/A'),
                        }
                    series_files[series_uid].append(filepath)
            except Exception as e:
                logger.debug(f"Skipping file {filepath.name}: {e}")
                continue
    
    total_series = len(series_files)
    logger.info(f"  Scanned {total_files_scanned} files")
    logger.info(f"  Found {total_series} unique series")
    logger.info(f"  Output directories: {len(output_dirs_set)}")
    logger.info(f"")
    
    if total_series == 0:
        logger.warning("No DICOM series found in input files")
        return {
            'total_series_input': 0,
            'total_series_output': 0,
            'total_frames_input': 0,
            'total_frames_output': 0,
            'total_size_input_mb': 0.0,
            'total_size_output_mb': 0.0,
            'converted_to_multiframe': 0,
            'transcoded_htj2k': 0,
            'copied': 0,
            'failed': 0,
            'series_info': []
        }
    
    # Step 2: Convert each series
    stats = {
        'total_series_input': total_series,
        'total_series_output': 0,
        'total_frames_input': 0,
        'total_frames_output': 0,
        'total_size_input_mb': 0.0,
        'total_size_output_mb': 0.0,
        'converted_to_multiframe': 0,
        'transcoded_htj2k': 0,
        'copied': 0,
        'failed': 0,
        'series_info': [],
    }
    
    for idx, (series_uid, file_paths) in enumerate(series_files.items(), 1):
        metadata = series_metadata[series_uid]
        output_dir = series_output_dirs[series_uid]
        num_files = len(file_paths)
        
        # Calculate input size for this series
        series_input_size = sum(fp.stat().st_size for fp in file_paths)
        series_input_size_mb = series_input_size / (1024 * 1024)
        
        # Always count input
        stats['total_size_input_mb'] += series_input_size_mb
        stats['total_frames_input'] += num_files
        
        logger.info(f"")
        logger.info(f"{'-'*80}")
        logger.info(f"Series {idx}/{total_series}")
        logger.info(f"{'-'*80}")
        logger.info(f"  SeriesInstanceUID: {series_uid}")
        logger.info(f"  Modality:          {metadata['modality']}")
        logger.info(f"  SeriesNumber:      {metadata['series_number']}")
        logger.info(f"  SeriesDescription: {metadata['series_description']}")
        logger.info(f"  PatientID:         {metadata['patient_id']}")
        logger.info(f"  Number of files:   {num_files}")
        logger.info(f"  Input size:        {series_input_size_mb:.2f} MB")
        logger.info(f"  Output directory:  {output_dir}")
        logger.info(f"")
        
        # Check if modality is supported for enhanced multi-frame conversion
        if metadata['modality'] not in SUPPORTED_MODALITIES:
            logger.info(f"  Unsupported modality for enhanced conversion: {metadata['modality']}")
            logger.info(f"  Supported for enhanced conversion: {', '.join(SUPPORTED_MODALITIES)}")
            
            if compress_htj2k:
                # Transcode to HTJ2K even though we can't convert to enhanced format
                logger.info(f"  Transcoding to HTJ2K (preserving original format)...")
                
                success, transcoded_size_mb = _transcode_files_to_htj2k(
                    file_paths, output_dir, compression_kwargs
                )
                
                if success:
                    logger.info(f"  Transcoded {num_files} files with HTJ2K")
                    logger.info(f"    Input size:  {series_input_size_mb:.2f} MB")
                    logger.info(f"    Output size: {transcoded_size_mb:.2f} MB")
                    if series_input_size_mb > 0:
                        compression = series_input_size_mb / transcoded_size_mb
                        logger.info(f"    Compression: {compression:.2f}x")
                    
                    # Update transcoded HTJ2K statistics
                    stats['transcoded_htj2k'] += 1
                    stats['total_series_output'] += 1
                    stats['total_frames_output'] += num_files
                    stats['total_size_output_mb'] += transcoded_size_mb
                    
                    stats['series_info'].append({
                        'series_uid': series_uid,
                        'status': 'transcoded_htj2k',
                        'reason': f"Unsupported modality: {metadata['modality']} (HTJ2K compressed)",
                        'num_files': num_files,
                        'input_size_mb': series_input_size_mb,
                        'output_size_mb': transcoded_size_mb,
                    })
                else:
                    # Fall back to copying
                    logger.info(f"  Falling back to copying files without compression...")
                    copied_count = _copy_files(file_paths, output_dir)
                    
                    logger.info(f"  Copied {copied_count}/{num_files} files")
                    stats['copied'] += 1
                    stats['total_series_output'] += 1
                    stats['total_frames_output'] += num_files
                    stats['total_size_output_mb'] += series_input_size_mb
                    
                    stats['series_info'].append({
                        'series_uid': series_uid,
                        'status': 'copied',
                        'reason': f"Unsupported modality: {metadata['modality']} (copied, transcode failed)",
                        'num_files': num_files,
                    })
            else:
                # No compression - just copy files
                logger.info(f"  Copying files to output directory...")
                copied_count = _copy_files(file_paths, output_dir)
                
                logger.info(f"  Copied {copied_count}/{num_files} files")
                
                # Update copied statistics
                stats['copied'] += 1
                stats['total_series_output'] += 1
                stats['total_frames_output'] += num_files
                stats['total_size_output_mb'] += series_input_size_mb
                
                stats['series_info'].append({
                    'series_uid': series_uid,
                    'status': 'copied',
                    'reason': f"Unsupported modality: {metadata['modality']} (copied)",
                    'num_files': num_files,
                })
            
            continue
        
        # Create temporary directory for this series
        with tempfile.TemporaryDirectory() as temp_series_dir:
            temp_series_path = Path(temp_series_dir)
            
            # Copy files to temporary directory
            for file_path in file_paths:
                shutil.copy(file_path, temp_series_path / file_path.name)
            
            # Generate output filename: {Modality}_{SeriesInstanceUID}.dcm
            output_filename = f"{metadata['modality']}_{series_uid}.dcm"
            output_file = output_dir / output_filename
            logger.info(f"  Output filename: {output_filename}")
            
            # Convert the series
            try:
                if compress_htj2k:
                    success = convert_and_convert_to_htj2k(
                        input_source=temp_series_path,
                        output_file=output_file,
                        preserve_series_uid=preserve_series_uid,
                        **compression_kwargs
                    )
                else:
                    success = convert_to_enhanced_dicom(
                        input_source=temp_series_path,
                        output_file=output_file,
                        preserve_series_uid=preserve_series_uid,
                    )
                
                if success:
                    # Get output file size
                    output_size = output_file.stat().st_size
                    output_size_mb = output_size / (1024 * 1024)
                    
                    # Update conversion statistics
                    stats['converted_to_multiframe'] += 1
                    stats['total_series_output'] += 1
                    stats['total_frames_output'] += num_files  # Frames are combined into 1 file
                    stats['total_size_output_mb'] += output_size_mb
                    
                    stats['series_info'].append({
                        'series_uid': series_uid,
                        'status': 'success',
                        'output_file': str(output_file),
                        'num_frames': num_files,
                        'input_size_mb': series_input_size_mb,
                        'output_size_mb': output_size_mb,
                    })
                else:
                    stats['failed'] += 1
                    stats['series_info'].append({
                        'series_uid': series_uid,
                        'status': 'failed',
                        'reason': 'Conversion returned False'
                    })
                    
            except Exception as e:
                logger.error(f"  Failed to convert series: {e}")
                stats['failed'] += 1
                stats['series_info'].append({
                    'series_uid': series_uid,
                    'status': 'failed',
                    'reason': str(e)
                })
    
    # Calculate overall compression statistics
    if stats['total_size_input_mb'] > 0 and stats['total_size_output_mb'] > 0:
        overall_compression = stats['total_size_input_mb'] / stats['total_size_output_mb']
        size_reduction_pct = ((stats['total_size_input_mb'] - stats['total_size_output_mb']) / 
                              stats['total_size_input_mb']) * 100
    else:
        overall_compression = 0.0
        size_reduction_pct = 0.0
    
    # Print comprehensive summary
    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info(f"Batch Conversion Summary")
    logger.info(f"{'='*80}")
    logger.info(f"")
    logger.info(f"  Total series (input):   {stats['total_series_input']}")
    logger.info(f"  Total series (output):  {stats['total_series_output']}")
    logger.info(f"  Total frames (input):   {stats['total_frames_input']}")
    logger.info(f"  Total frames (output):  {stats['total_frames_output']}")
    logger.info(f"  Total size (input):     {stats['total_size_input_mb']:.2f} MB")
    logger.info(f"  Total size (output):    {stats['total_size_output_mb']:.2f} MB")
    if overall_compression > 0:
        logger.info(f"  Compression ratio:      {overall_compression:.2f}x")
        logger.info(f"  Size reduction:         {size_reduction_pct:.1f}%")
    logger.info(f"")
    logger.info(f"  Details:")
    if compress_htj2k:
        logger.info(f"    Converted to multi-frame + HTJ2K: {stats['converted_to_multiframe']}")
        logger.info(f"    Transcoded to HTJ2K only:       {stats['transcoded_htj2k']} (unsupported modalities)")
    else:
        logger.info(f"    Converted to multi-frame: {stats['converted_to_multiframe']}")
        logger.info(f"    Transcoded to HTJ2K:    {stats['transcoded_htj2k']}")
    logger.info(f"    Copied:                 {stats['copied']}")
    logger.info(f"    Failed:                 {stats['failed']}")
    logger.info(f"")
    if len(output_dirs_set) == 1:
        logger.info(f"  Output directory: {list(output_dirs_set)[0]}")
    else:
        logger.info(f"  Output directories: {len(output_dirs_set)}")
        for out_dir in sorted(output_dirs_set):
            logger.info(f"    - {out_dir}")
    logger.info(f"{'='*80}")
    logger.info(f"")
    
    return stats


def convert_and_convert_to_htj2k(
    input_source: Union[str, Path, List[Union[str, Path]]],
    output_file: Union[str, Path],
    preserve_series_uid: bool = True,
    **compression_kwargs,
) -> bool:
    """
    Convert legacy DICOM series to enhanced multi-frame format and compress with HTJ2K.
    
    This is a convenience function that combines multi-frame conversion and HTJ2K compression
    in one step. It creates an uncompressed enhanced DICOM first, then compresses it using
    HTJ2K (High-Throughput JPEG2000).
    
    Args:
        input_source: Either:
            - Directory path containing legacy DICOM files
            - List of DICOM file paths to convert as a single series
        output_file: Path for output HTJ2K compressed enhanced DICOM file
        preserve_series_uid: If True, preserve original SeriesInstanceUID
        **compression_kwargs: Additional arguments for HTJ2K compression:
            - num_resolutions (int): Number of wavelet decomposition levels (default: 6)
            - code_block_size (tuple): Code block size (default: (64, 64))
            - progression_order (str): Progression order (default: "RPCL")
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> # Convert to multi-frame and compress with HTJ2K from directory
        >>> convert_and_convert_to_htj2k(
        ...     input_source="./legacy_ct/",
        ...     output_file="./enhanced_htj2k.dcm",
        ...     num_resolutions=6,
        ...     progression_order="RPCL"
        ... )
        >>> 
        >>> # Convert from file list
        >>> files = ['/data/ct_001.dcm', '/data/ct_002.dcm', '/data/ct_003.dcm']
        >>> convert_and_convert_to_htj2k(
        ...     input_source=files,
        ...     output_file="./enhanced_htj2k.dcm"
        ... )
    """
    output_file = Path(output_file)
    
    # Import here to avoid circular dependency
    try:
        from monailabel.datastore.utils.convert_htj2k import transcode_dicom_to_htj2k
    except ImportError as e:
        logger.error(f"HTJ2K compression requires convert_htj2k module: {e}")
        return False
    
    # Calculate original files size for statistics
    original_size_bytes = 0
    num_files = 0
    
    # Get list of file paths
    if isinstance(input_source, (list, tuple)):
        file_paths = [Path(f) for f in input_source]
    else:
        input_dir = Path(input_source)
        if not input_dir.is_dir():
            logger.error(f"Input path is not a directory: {input_dir}")
            return False
        file_paths = [f for f in input_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    
    # Calculate total input size
    for filepath in file_paths:
        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            original_size_bytes += filepath.stat().st_size
            num_files += 1
        except Exception:
            continue
    
    original_size_mb = original_size_bytes / (1024 * 1024)
        
    # Step 1: Create uncompressed enhanced DICOM in temp file
    with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp:
        temp_file = tmp.name
        
    try:
        logger.info("Step 1/2: Creating enhanced multi-frame DICOM (uncompressed)...")
        success = convert_to_enhanced_dicom(
            input_source=input_source,
            output_file=temp_file,
            preserve_series_uid=preserve_series_uid,
            show_stats=False,  # Suppress intermediate statistics
        )
        
        if not success:
            logger.error("Failed to convert to enhanced multi-frame DICOM")
            return False
        
        # Get intermediate uncompressed size
        temp_size_bytes = Path(temp_file).stat().st_size
        temp_size_mb = temp_size_bytes / (1024 * 1024)
        
        # Step 2: Compress with HTJ2K
        logger.info("Step 2/2: Compressing with HTJ2K...")
        
        # Extract HTJ2K parameters
        num_resolutions = compression_kwargs.get('num_resolutions', 6)
        code_block_size = compression_kwargs.get('code_block_size', (64, 64))
        progression_order = compression_kwargs.get('progression_order', 'RPCL')
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Compress the enhanced DICOM
        # transcode_dicom_to_htj2k expects a file_loader iterable
        file_loader = [([temp_file], [str(output_file)])]
        transcode_dicom_to_htj2k(
            file_loader=file_loader,
            num_resolutions=num_resolutions,
            code_block_size=code_block_size,
            progression_order=progression_order,
        )
        
        # Check if output file was created successfully
        if output_file.exists():
            output_size_bytes = output_file.stat().st_size
            output_size_mb = output_size_bytes / (1024 * 1024)
            
            # Get image metadata from the output file
            try:
                output_ds = pydicom.dcmread(output_file, stop_before_pixels=True)
                num_frames = getattr(output_ds, 'NumberOfFrames', num_files)
                rows = getattr(output_ds, 'Rows', 'N/A')
                columns = getattr(output_ds, 'Columns', 'N/A')
            except Exception:
                num_frames = num_files
                rows = 'N/A'
                columns = 'N/A'
            
            # Calculate compression statistics
            if original_size_bytes > 0:
                overall_compression_ratio = original_size_bytes / output_size_bytes
                size_reduction_pct = ((original_size_bytes - output_size_bytes) / original_size_bytes) * 100
                htj2k_compression_ratio = temp_size_bytes / output_size_bytes
            else:
                overall_compression_ratio = 0.0
                size_reduction_pct = 0.0
                htj2k_compression_ratio = 0.0
            
            # Display comprehensive statistics at the end
            logger.info(f"")
            logger.info(f"✓ Successfully created HTJ2K compressed enhanced DICOM: {output_file}")
            logger.info(f"")
            logger.info(f"  Conversion Statistics:")
            logger.info(f"    Original files:        {num_files} files, {original_size_mb:.2f} MB")
            logger.info(f"    Uncompressed enhanced: 1 file, {temp_size_mb:.2f} MB")
            logger.info(f"    HTJ2K compressed:      1 file, {output_size_mb:.2f} MB")
            logger.info(f"")
            if original_size_bytes > 0:
                logger.info(f"  Compression Performance:")
                logger.info(f"    Overall size reduction: {size_reduction_pct:.1f}% smaller")
                logger.info(f"    Overall compression:    {overall_compression_ratio:.2f}x")
                logger.info(f"    HTJ2K compression:      {htj2k_compression_ratio:.2f}x")
                logger.info(f"")
            logger.info(f"  Image Information:")
            logger.info(f"    Frames:                 {num_frames}")
            if rows != 'N/A' and columns != 'N/A':
                logger.info(f"    Dimensions:             {rows}x{columns}")
            logger.info(f"")
            
            return True
        else:
            logger.error(f"HTJ2K compression failed - output file not created")
            return False
            
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    # Example CLI usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert legacy DICOM series to enhanced multi-frame format"
    )
    parser.add_argument(
        "input", type=str,
        help="Input directory containing legacy DICOM files"
    )
    parser.add_argument(
        "-o", "--output", type=str,
        help="Output file path for enhanced DICOM (required unless --validate-only)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate series without creating output"
    )
    
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch mode: group files by SeriesInstanceUID and convert each series separately"
    )

    parser.add_argument(
        "--htj2k", action="store_true",
        help="Compress the enhanced DICOM file with HTJ2K"
    )

    parser.add_argument(
        "--num-resolutions", type=int, default=6,
        help="Number of wavelet decomposition levels (default: 6)"
    )

    parser.add_argument(
        "--code-block-size", type=tuple, default=(64, 64),
        help="Code block size (default: (64, 64))"
    )

    parser.add_argument(
        "--progression-order", type=str, default="RPCL",
        help="Progression order (default: RPCL)"
    )

    parser.add_argument(
        "--preserve-series-uid", action="store_true",
        help="Preserve the original SeriesInstanceUID"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s"
    )
    
    # Run conversion
    try:
        if args.batch:
            # Batch mode: group by series and convert each
            if not args.output:
                parser.error("--output is required for batch mode (use as output directory)")
            
            # Scan input directory for DICOM files
            input_dir = Path(args.input)
            input_files = []
            for filepath in input_dir.rglob('*'):
                if filepath.is_file() and not filepath.name.startswith('.'):
                    try:
                        # Quick check if it's a DICOM file
                        pydicom.dcmread(filepath, stop_before_pixels=True)
                        input_files.append(str(filepath))
                    except:
                        pass  # Skip non-DICOM files
            
            # Create file_loader with single batch
            file_loader = [(input_files, args.output)]
            
            stats = batch_convert_by_series(
                file_loader=file_loader,
                preserve_series_uid=args.preserve_series_uid,
                compress_htj2k=args.htj2k,
                num_resolutions=args.num_resolutions,
                code_block_size=args.code_block_size,
                progression_order=args.progression_order,
            )
            exit(0 if stats['failed'] == 0 else 1)
        else:
            # Single series mode
            if not args.validate_only and not args.output:
                parser.error("--output is required unless --validate-only is specified")

            if args.htj2k:
                success = convert_and_convert_to_htj2k(
                    input_source=args.input,
                    output_file=args.output,
                    preserve_series_uid=args.preserve_series_uid,
                    num_resolutions=args.num_resolutions,
                    code_block_size=args.code_block_size,
                    progression_order=args.progression_order,
                )
            else:
                success = convert_to_enhanced_dicom(
                    input_source=args.input,
                    output_file=args.output or "dummy.dcm",
                    validate_only=args.validate_only,
                    preserve_series_uid=args.preserve_series_uid,
                )
            exit(0 if success else 1)
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        exit(1)

