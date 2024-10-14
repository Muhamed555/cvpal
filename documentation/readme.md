# CvPal

`CvPal` is a versatile tool for handling and manipulating data, particularly designed for various formats such as text and COCO JSON, XML..etc, Below are the details of each method available in the `CvPal` class.

## Usage


### `read_data`
Reads the data and sets the path and data type.

#### Parameters:
- `path` (str): The path to the data.
- `data_type` (str): The type of data ('txt' or 'coco_json').

### `merge`
Merges data from multiple paths and optionally creates a YAML file.

#### Parameters:
- `paths` (list): A list containing multiple paths to the data.
- `output_folder_name` (str): The name of the output folder. Default is "output".
- `base_storage_path` (str): The base storage path. Default is None.
- `parental_reference` (bool): If True, includes parental references. Default is False.

#### Returns:
- `str`: The path to the new dataset.

### `delete_label`
Deletes a certain label from the dataset.

#### Parameters:
- `label_to_delete` (str/int): The label to be deleted.

### `count_labels`
Counts the occurrence of each label in the dataset.

#### Returns:
- `dict`: A dictionary with labels as keys and their occurrences as values.

### `replace_label`
Replaces a label in the dataset based on a dictionary of replacements.

#### Parameters:
- `replacement_dict` (dict): A dictionary with old labels as keys and new labels as values.

### `find_files_with_label`
Finds files that contain a certain label.

#### Parameters:
- `label` (str): The label to find.
- `exclusive` (bool): If True, finds files that contain only the specified label. Default is False.

#### Returns:
- `list`: A list of tubles containing both the image and its corresponding label file.

### `report`
Creates a DataFrame that contains columns for image path, label, and label path.

#### Returns:
- `pandas.DataFrame`: A DataFrame containing the report.

## Example

```python
from cvPal import CvPal

# Initialize cvPal
cv = CvPal()

# Read data
cv.read_data(path="path/to/data", data_type="txt")

# Merge data
cv.merge(paths=["path1", "path2"], output_folder_name="merged_output")

# Delete a label
cv.delete_label(label_to_delete="old_label")

# Count labels
label_counts = cv.count_labels()
print(label_counts)

# Replace labels
cv.replace_label(replacement_dict={"old_label": "new_label"})

# Find files with a specific label
files = cv.find_files_with_label(label="label_to_find")
print(files)

# Generate report
report_df = cv.report()
print(report_df)

