from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/home/research1/.cache/huggingface/lerobot/physical-intelligence/Piper_test",
    repo_id="RainyBot/PiPER_test",
    repo_type="dataset",
)
