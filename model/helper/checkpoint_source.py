from pathlib import Path

from huggingface_hub import hf_hub_download

def resolve_checkpoint_source(ref: str) -> Path:
    """
    Local path if it exists on disk, else '<owner>/<repo>/<run_name>/<filename>'
    on the HF Hub -- repo_id is always exactly the first two '/'-separated segments,
    everything after that is the path within the repo."""
    local_path = Path(ref)
    if local_path.exists():
        return local_path
    
    parts = ref.split("/")
    repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
    
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))
