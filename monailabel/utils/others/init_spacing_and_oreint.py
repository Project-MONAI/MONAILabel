import os
import pathlib as Path
from typing import List, Union

import SimpleITK as sitk


def get_file_paths(
    folder: Union[str, Path], extension_tuple=(".nrrd"), pattern: str = None, ignore_folders: Union[List, str] = []
) -> Union[List[str], List[Path]]:
    """
    Given a directory return all image paths within all sibdirectories.

    Parameters
    ----------
    folder
        Where to look for images
    extension_tuple
        Select only images with these extensions
    pattern
        Do a simple `pattern in filename` filter on filenames
    ignore_folders
        do not look in folder with these names

    Notes
    -----
    Lama is currently using a mixture of Paths or str to represent filepaths. Will move all to Path.
    For now, return same type as folder input

    Do not include hidden filenames
    """
    paths = []

    if isinstance(ignore_folders, str):
        ignore_folders = [ignore_folders]

    for root, subfolders, files in os.walk(folder):
        for f in ignore_folders:
            if f in subfolders:
                subfolders.remove(f)

        for filename in files:
            if filename.lower().endswith(extension_tuple) and not filename.startswith("."):
                if pattern:
                    if pattern and pattern not in filename:
                        continue

                paths.append(os.path.abspath(os.path.join(root, filename)))

    if isinstance(folder, str):
        return paths
    else:
        return [Path(x) for x in paths]


def initialise_spacing(target_dir):
    """
    output from MPI2/LAMA removes direction and orientation information, leaving them to be blank within headers.
    This function should initialise direction and orientation information
    """
    volpaths = get_file_paths(target_dir)

    for i, path in enumerate(volpaths):
        vol = sitk.ReadImage(path)
        # pa = sitk.PermuteAxesImageFilter()
        # pa.SetOrder([1,0,2])
        # flipped_vol = pa.Execute(vol)

        # sets Direction and Origin
        vol.SetDirection([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
        vol.SetOrigin([0.0, 0.0, 0.0])
        sitk.WriteImage(vol, path, True)
