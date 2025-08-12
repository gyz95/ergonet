from .ergonet_batch import ErgoNetBatchProcessor
from .ergonet_single import ErgoNetSingleProcessor
from .ergonet_helper import ErgoNetHelper

__version__ = "1.0.0"
__all__ = ['ErgoNetBatchProcessor', 'ErgoNetSingleProcessor','ErgoNetHelper']

import os
import zipfile
from pathlib import Path
import gdown

def _download_and_unzip_gdrive(file_id, folder_name, zip_filename):
    # Always store in the installed package's directory
    PKG_ROOT = Path(__file__).resolve().parent
    project_dir = PKG_ROOT / folder_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Skip if folder already has content
    if any(project_dir.iterdir()):
        print(f"[ergonet_message] {folder_name}/ already populated. Skipping download.")
        return

    zip_path = project_dir / zip_filename
    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

    print(f"[ergonet_message] Downloading {zip_filename} to {zip_path}")
    try:
        gdown.download(gdrive_url, str(zip_path), quiet=False)
    except Exception as e:
        print(f"[ergonet_message] Failed to download {zip_filename}: {e}")
        return

    print(f"[ergonet_message] Unzipping {zip_filename} into {project_dir}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = [m for m in zip_ref.namelist() if not m.startswith('__MACOSX/')]
            root_dirs = set(p.split('/')[0] for p in members if '/' in p)

            # If the zip has a single root folder, strip it
            if len(root_dirs) == 1:
                root_prefix = next(iter(root_dirs)) + '/'
                for member in members:
                    if member.startswith(root_prefix):
                        relative_path = member[len(root_prefix):]
                        if relative_path == '' or relative_path.endswith('/'):
                            continue
                        target_path = project_dir / relative_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                            target.write(source.read())
            else:
                # Extract all but MACOSX
                for member in members:
                    if member.endswith('/'):
                        continue
                    target_path = project_dir / member
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())

        zip_path.unlink()  # remove zip after extraction
        print(f"[ergonet_message] Done unzipping and cleanup.")
    except zipfile.BadZipFile as e:
        print(f"[ergonet_message] Failed to unzip {zip_filename}: {e}")


# ========== Download Tasks ==========
_download_and_unzip_gdrive(
    file_id="1dx6x3MXsNl6JGT0R-cZzaVwwZhsnZc-Y",
    folder_name="smpl_data",
    zip_filename="smpl_data.zip"
)

_download_and_unzip_gdrive(
    file_id="1TW0Got_zFU2Z2Fmxbcis8LUN77yFF9sW",
    folder_name="checkpoints",
    zip_filename="checkpoints.zip"
)