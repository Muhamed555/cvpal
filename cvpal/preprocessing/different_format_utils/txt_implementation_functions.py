import os
import shutil
import yaml
from collections import defaultdict
import pandas as pd


def merge_data_txt(path, paths, folder_name_provided, base_storage_path, parental_reference):
    """
    Merges data from multiple folders into one unified folder structure and aligns labels.

    Parameters:
    path (str): Path of the primary folder.
    paths (list): A list of paths to other folders.
    folder_name_provided (str): The name of the output folder.
    base_storage_path (str): The base path where the output folder will be stored. If None, uses the current directory.
    parental_reference (bool): If False, combine all the data into a single folder structure. If True, only add labels that match parental labels.

    Returns:
    str: The path of the unified output folder.
    dict: A dictionary of excluded images and their label files.
    """

    def create_folder_structure(base_path):
        for split in ['train', 'test', 'valid']:
            os.makedirs(os.path.join(base_path, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(base_path, split, 'labels'), exist_ok=True)

    def copy_files_and_fix_labels(src_paths, dest_path, label_map, parental_labels):
        excluded_images = {}
        for src_path in src_paths:
            for split in ['train', 'test', 'valid']:
                src_images = os.path.join(src_path, split, 'images')
                src_labels = os.path.join(src_path, split, 'labels')

                dest_images = os.path.join(dest_path, split, 'images')
                dest_labels = os.path.join(dest_path, split, 'labels')

                print(f"Copying images from {src_images} to {dest_images}")
                print(f"Copying labels from {src_labels} to {dest_labels}")

                # Copy images
                for file_name in os.listdir(src_images):
                    src_image_file = os.path.join(src_images, file_name)
                    dest_image_file = os.path.join(dest_images, file_name)
                    src_label_file = os.path.join(src_labels,
                                                  file_name.replace('.jpg', '.txt'))  # Assuming images are .jpg

                    # Read and filter labels
                    if os.path.exists(src_label_file):
                        with open(src_label_file, 'r') as f:
                            lines = f.readlines()

                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            class_id = int(parts[0])

                            if parental_reference:
                                if class_id in label_map:
                                    new_class_id = label_map[class_id]
                                    parts[0] = str(new_class_id)
                                    new_lines.append(' '.join(parts) + '\n')
                            else:
                                new_class_id = label_map[class_id]
                                parts[0] = str(new_class_id)
                                new_lines.append(' '.join(parts) + '\n')

                        if new_lines:
                            shutil.copy(src_image_file, dest_images)
                            with open(os.path.join(dest_labels, file_name.replace('.jpg', '.txt')), 'w') as f:
                                f.writelines(new_lines)
                        else:
                            excluded_images[src_image_file] = src_label_file
        return excluded_images

    def read_yaml(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def build_label_map(parent_labels, other_labels):
        label_map = {}
        for i, label in enumerate(other_labels):
            if label in parent_labels:
                label_map[i] = parent_labels.index(label)
            elif not parental_reference:
                # Find the next available label index
                next_index = len(parent_labels)
                parent_labels.append(label)
                label_map[i] = next_index
        return label_map

    # Determine the base storage path
    if base_storage_path is None:
        base_storage_path = os.getcwd()

    # Create the unified folder structure
    unified_path = os.path.join(base_storage_path, folder_name_provided)
    create_folder_structure(unified_path)

    # Read parental YAML file and create the labels dictionary
    parental_yaml_path = os.path.join(path, 'data.yaml')  # Assuming YAML file is named 'dataset.yaml'
    parental_data = read_yaml(parental_yaml_path)
    parental_labels = parental_data['names']

    # Collect excluded images from each additional folder
    excluded_images_total = {}

    # Process each additional folder
    for i, other_path in enumerate(paths):
        other_yaml_path = os.path.join(other_path, 'data.yaml')  # Assuming YAML file is named 'dataset.yaml'
        other_data = read_yaml(other_yaml_path)
        other_labels = other_data['names']

        # Build the label map for the current folder
        label_map = build_label_map(parental_labels, other_labels)

        # Copy files and fix labels
        excluded_images = copy_files_and_fix_labels([other_path], unified_path, label_map, parental_labels)
        excluded_images_total.update(excluded_images)

    # Copy files from the primary folder and fix labels
    primary_label_map = {i: i for i in range(len(parental_labels))}
    excluded_images = copy_files_and_fix_labels([path], unified_path, primary_label_map, parental_labels)
    excluded_images_total.update(excluded_images)

    # Create the output YAML file with the updated labels
    yaml_data = {
        'train': os.path.join(unified_path, 'train'),
        'test': os.path.join(unified_path, 'test'),
        'val': os.path.join(unified_path, 'val'),
        'nc': len(parental_labels),
        'names': parental_labels
    }

    with open(os.path.join(unified_path, 'data.yaml'), 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)

    # Return the unified path and the excluded images dictionary
    return unified_path, excluded_images_total

def remove_label_from_txt_dataset(dataset_path, label_to_remove):
    """
    Removes a specified label from the dataset and reindexes the remaining labels.

    Parameters:
    dataset_path (str): The path to the dataset directory containing train, test, valid folders and data.yaml file.
    label_to_remove (str or int): The label name or index to be removed from the dataset.
    """

    def read_yaml(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def write_yaml(yaml_path, data):
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data, f)

    def update_label_indices_in_txt(txt_path, index_to_remove, reindex_map):
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id != index_to_remove:
                new_class_id = reindex_map[class_id]
                new_line = ' '.join([str(new_class_id)] + parts[1:])
                new_lines.append(new_line)

        with open(txt_path, 'w') as f:
            for line in new_lines:
                f.write(line + '\n')

    # Read the data.yaml file
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    data = read_yaml(yaml_path)

    # Build the dictionary of labels and indices
    labels_dict = {name: idx for idx, name in enumerate(data['names'])}

    # Determine the index to remove
    if isinstance(label_to_remove, str):
        index_to_remove = labels_dict.get(label_to_remove)
        if index_to_remove is None:
            raise ValueError(f"Label name '{label_to_remove}' not found in the dataset.")
    elif isinstance(label_to_remove, int):
        if label_to_remove < 0 or label_to_remove >= len(data['names']):
            raise ValueError(f"Label index '{label_to_remove}' is out of range.")
        index_to_remove = label_to_remove
    else:
        raise ValueError("label_to_remove must be a string (label name) or an integer (label index).")

    # Remove the label from the labels dictionary
    label_name_to_remove = data['names'][index_to_remove]
    del labels_dict[label_name_to_remove]

    # Remove the label from the data.yaml file
    data['names'].remove(label_name_to_remove)
    data['nc'] -= 1

    # Create a reindex map for the remaining labels
    reindex_map = {}
    for new_idx, name in enumerate(data['names']):
        old_idx = labels_dict[name]
        reindex_map[old_idx] = new_idx

    # Paths to train, test, valid folders
    splits = ['train', 'test', 'valid']

    # Update the indices in the txt files in each folder
    for split in splits:
        labels_folder = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(labels_folder):
            continue
        for txt_file in os.listdir(labels_folder):
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(labels_folder, txt_file)
                update_label_indices_in_txt(txt_path, index_to_remove, reindex_map)

    # Write the updated data.yaml file
    write_yaml(yaml_path, data)
    print(f"Label '{label_name_to_remove}' (index {index_to_remove}) removed successfully.")


def count_labels_in_txt_dataset(dataset_path):
    """
    Counts the occurrences of each label in the train, valid, and test folders of a dataset.

    Parameters:
    dataset_path (str): The path to the dataset directory containing train, test, valid folders and data.yaml file.

    Returns:
    dict: A dictionary with counts of each label in train, valid, and test folders.
    """

    def read_yaml(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def count_labels_in_folder(folder_path, labels_dict):
        label_counts = {label: 0 for label in labels_dict.values()}
        unknown_label_counts = defaultdict(int)

        labels_folder = os.path.join(folder_path, 'labels')
        if not os.path.exists(labels_folder):
            return label_counts, unknown_label_counts

        for txt_file in os.listdir(labels_folder):
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(labels_folder, txt_file)
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    if class_id in labels_dict:
                        label_name = labels_dict[class_id]
                        label_counts[label_name] += 1
                    else:
                        unknown_label_counts[class_id] += 1
                        print(f"Warning: class_id {class_id} not found in labels_dict in file {txt_path}")

        return label_counts, unknown_label_counts

    # Read the data.yaml file
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    data = read_yaml(yaml_path)

    # Build the dictionary of label indices and names
    labels_dict = {idx: name.strip() for idx, name in enumerate(data['names'])}

    # Paths to train, test, valid folders
    splits = ['train', 'test', 'valid']
    label_counts = {}
    unknown_counts = {}

    # Count labels in each folder
    for split in splits:
        folder_path = os.path.join(dataset_path, split)
        if os.path.exists(folder_path):
            counts, unknowns = count_labels_in_folder(folder_path, labels_dict)
            label_counts[split] = counts
            if unknowns:
                unknown_counts[split] = unknowns

    if unknown_counts:
        print(f"Summary of unknown class IDs found: {unknown_counts}")

    return label_counts


def replace_labels_in_txt_yaml(dataset_path, labels_dict):
    """
    Replaces old labels with new ones in the YAML file.

    Parameters:
    dataset_path (str): The path to the dataset directory containing the data.yaml file.
    labels_dict (dict): A dictionary where keys are old labels and values are new labels.
    """

    def read_yaml(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def write_yaml(yaml_path, data):
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data, f)

    # Read the data.yaml file
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    data = read_yaml(yaml_path)

    # Replace labels in the data.yaml file
    for old_label, new_label in labels_dict.items():
        if old_label in data['names']:
            index = data['names'].index(old_label)
            data['names'][index] = new_label
        else:
            print(f"Error: Label '{old_label}' not found in the dataset.")
            return

    # Write the updated data.yaml file
    write_yaml(yaml_path, data)
    print("Labels updated successfully.")


def find_images_with_label_in_txt_type(dataset_path, label, exclusive=False):
    """
    Finds images and txt files that contain the specified label.

    Parameters:
    dataset_path (str): The path to the dataset directory containing train, test, valid folders and data.yaml file.
    label (str or int): The label name or index to find in the dataset.
    exclusive (bool): If True, include only the images and files that contain this label only. If False, include them whether they are individual or with other labels.

    Returns:
    list: A list of tuples containing paths of images and corresponding txt files.
    """

    def read_yaml(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    # Read the data.yaml file
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    data = read_yaml(yaml_path)

    # Build the dictionary of labels and indices
    labels_dict = {name: idx for idx, name in enumerate(data['names'])}

    # Determine the index of the label to find
    if isinstance(label, str):
        index_to_find = labels_dict.get(label)
        if index_to_find is None:
            raise ValueError(f"Label name '{label}' not found in the dataset.")
    elif isinstance(label, int):
        if label < 0 or label >= len(data['names']):
            raise ValueError(f"Label index '{label}' is out of range.")
        index_to_find = label
    else:
        raise ValueError("label must be a string (label name) or an integer (label index).")

    # Paths to train, test, valid folders
    splits = ['train', 'test', 'valid']
    result = []

    # Find the images and txt files with the specified label
    for split in splits:
        images_folder = os.path.join(dataset_path, split, 'images')
        labels_folder = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(labels_folder):
            continue
        for txt_file in os.listdir(labels_folder):
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(labels_folder, txt_file)
                with open(txt_path, 'r') as f:
                    lines = f.readlines()

                has_label = False
                is_exclusive = True
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    if class_id == index_to_find:
                        has_label = True
                    else:
                        is_exclusive = False

                if has_label and (not exclusive or is_exclusive):
                    img_path = os.path.join(images_folder, txt_file.replace('.txt', '.jpg'))
                    result.append((img_path, txt_path))

    return result


def report(dataset_path):
    """
    Generates a report of the dataset.

    Parameters:
    dataset_path (str): The path to the dataset directory containing train, test, valid folders and data.yaml file.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the report with columns:
                  - image_path
                  - label_path
                  - num_of_labels
                  - labels
                  - directory
    """

    def read_yaml(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    # Read the data.yaml file
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    data = read_yaml(yaml_path)

    # Build the dictionary of labels and indices
    labels_dict = {idx: name for idx, name in enumerate(data['names'])}

    # Paths to train, test, valid folders
    splits = ['train', 'test', 'valid']
    report_data = []

    # Generate the report data
    for split in splits:
        images_folder = os.path.join(dataset_path, split, 'images')
        labels_folder = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(labels_folder):
            continue
        for txt_file in os.listdir(labels_folder):
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(labels_folder, txt_file)
                img_path = os.path.join(images_folder, txt_file.replace('.txt', '.jpg'))

                with open(txt_path, 'r') as f:
                    lines = f.readlines()

                labels_in_image = [labels_dict[int(line.strip().split()[0])] for line in lines]
                num_of_labels = len(labels_in_image)

                report_data.append({
                    'image_path': img_path,
                    'label_path': txt_path,
                    'num_of_labels': num_of_labels,
                    'labels': labels_in_image,
                    'directory': split
                })

    # Create a pandas DataFrame from the report data
    df_report = pd.DataFrame(report_data)

    return df_report
