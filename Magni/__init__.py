def os_change_folder(script_folder_name: str, old_folder_name: str, new_folder_name: str) -> str:
    data_folder_name = new_folder_name.join(script_folder_name.rsplit(old_folder_name, 1))

    return data_folder_name