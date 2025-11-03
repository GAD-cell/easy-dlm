from huggingface_hub import create_repo, upload_folder

local_folder = "/workspace/easy-dlm/a2d_output/checkpoint-7000"

new_repo_id = "GAD-cell/3M-7K-from_pretrained"

create_repo(repo_id=new_repo_id, repo_type="model", private=True, exist_ok=True)

upload_folder(
    folder_path=local_folder,
    repo_id=new_repo_id,
    repo_type="model",
)

