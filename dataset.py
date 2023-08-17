import os
import json
import random
import shutil
import string
from tqdm import tqdm
import cv2


class CVDataset:
    """
    A class to work with a dataset of UAV images and annotations.
    """

    def __init__(self, data_folder, json_filename=None):
        """
        Initialize the UAVDataset object.

        Args:
            data_folder (str): The path to the folder containing the dataset (data.json or json_filename) or full path to json.
            json_filename (str, optional): The name of the JSON file with dataset annotations.
            Defaults to 'data.json'.
        """
        if json_filename is None and data_folder.endswith('.json'):
            data_folder, json_filename = data_folder.replace("\\", "/").rsplit('/', 1)

        if not json_filename.endswith('.json'):
            json_filename = json_filename + ".json"
        self.class_name2id = {}
        self.id2class_name = {}
        self.data_folder = data_folder
        self.json_filename = json_filename
        self.data = self._load_data()

        # self.all_frames_data = self._extract_frames_data()

    def get_abs_img_path(self, relative_path):
        return os.path.join(self.data_folder, relative_path).replace('\\', '/')

    def _load_data(self):
        """
        Load the dataset annotations from the JSON file.

        Returns:
            dict: A dictionary containing the dataset annotations.
        """
        json_path = os.path.join(self.data_folder, self.json_filename)
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        for sequence_id, sequence_data in data.items():
            frames_data = sequence_data['frames']
            for image_id, image_data in frames_data.items():
                for obj in image_data['objects']:
                    cls_name = obj['cls']
                    if cls_name not in self.class_name2id:
                        cls_id = len(self.class_name2id)
                        self.class_name2id[cls_name] = cls_id
                        self.id2class_name[cls_id] = cls_name
        return data

    def get_sequence_names(self, sample_type=None):
        """
        Get sequence names based on the specified sample_type.

        Args:
            sample_type (str or list, optional): The sample type to filter sequence names by.
                If None, all sequence names will be returned. If a string, sequence names with
                the specified sample type will be returned. If a list, sequence names from
                the list will be returned. Defaults to None.

        Returns:
            list: A list of sequence names based on the sample_type filter.
        """
        all_sequence_names = list(self.data.keys())

        if sample_type is None:
            return all_sequence_names
        elif isinstance(sample_type, str):
            return [seq_name for seq_name in all_sequence_names if self.data[seq_name]['sample_type'] == sample_type]
        elif isinstance(sample_type, list):
            return [seq_name for seq_name in all_sequence_names if self.data[seq_name]['sample_type'] in sample_type]
        else:
            raise ValueError("Invalid sample_type parameter")

    def print_dataset_stats(self):
        """
        Print statistics about the dataset, including class information, object counts, image resolutions, etc.
        """
        num_sequences = len(self.get_sequence_names())
        num_images = sum(len(seq_data['frames']) for seq_data in self.data.values())
        num_classes = len(self.class_name2id)
        class_counts = {cls_name: {'count': 0, 'id': cls_id} for cls_name, cls_id in
                        self.class_name2id.items()}
        sample_type_count = {}
        bbox_areas = []
        bbox_ratios = []
        resolutions = {}
        w_bbox_list = []
        h_bbox_list = []
        area_bbox_list = []

        for sequence_id, sequence_data in self.data.items():
            frames_data = sequence_data['frames']
            sample_type = sequence_data['sample_type']
            sample_type_count[sample_type] = \
                sample_type_count.setdefault(sample_type, 0) + len(frames_data.keys())
            for image_id, image_data in frames_data.items():
                img_width = image_data['w']
                img_height = image_data['h']
                resolutions[(img_width, img_height)] = \
                    resolutions.setdefault((img_width, img_height), 0) + 1

                for obj_info in image_data['objects']:
                    class_name = obj_info['cls']
                    # class_id = self.class_name2id[class_name]
                    class_counts[class_name]['count'] += 1

                    bbox = obj_info['bb']
                    bbox_width = bbox[2] - bbox[0]
                    bbox_height = bbox[3] - bbox[1]
                    bbox_area = bbox_width * bbox_height
                    bbox_ratio = bbox_width / bbox_height if bbox_height > 0 else 0

                    bbox_areas.append(bbox_area)
                    bbox_ratios.append(bbox_ratio)
                    w_bbox_list.append(bbox_width)
                    h_bbox_list.append(bbox_height)
                    area_bbox_list.append(bbox_area)

        print("Dataset Statistics:")
        print(f"Number of sequences: {num_sequences}")
        print(f"Number of images: {num_images}")
        print(f"Number of classes: {num_classes}")
        print("\nClass counts:")
        for cls_name, cls_data in class_counts.items():
            print(f"- Class '{cls_name}' (ID {cls_data['id']}): {cls_data['count']} objects")
        for sample_name, count_sample in sample_type_count.items():
            print(f"{sample_name} : {count_sample} images")

        print("\nImage resolutions:")
        for i, (res, count) in enumerate(resolutions.items()):
            if i > 9:
                print(f'and other {len(resolutions) - 10} resolution ...')
                break
            print(f"- {res[0]} x {res[1]}: {count} images")
        print(f"\nAverage bbox area: {sum(bbox_areas) / len(bbox_areas):.2f} square pixels")
        print(f"Average bbox aspect ratio: {sum(bbox_ratios) / len(bbox_ratios):.2f}")
        print(f"Min bbox width: {min(w_bbox_list):.2f}")
        print(f"Max bbox width: {max(w_bbox_list):.2f}")
        print(f"Average bbox width: {sum(w_bbox_list) / len(w_bbox_list):.2f}")
        print(f"Min bbox height: {min(h_bbox_list):.2f}")
        print(f"Max bbox height: {max(h_bbox_list):.2f}")
        print(f"Average bbox height: {sum(h_bbox_list) / len(h_bbox_list):.2f}")
        print(f"Min bbox area: {min(area_bbox_list):.2f}")
        print(f"Max bbox area: {max(area_bbox_list):.2f}")
        print(f"Average bbox area: {sum(area_bbox_list) / len(area_bbox_list):.2f}")

    def draw_annotations_on_image(self, img, annotations, bbox_color=(0, 0, 255),
                                  text_color=(255, 255, 255), font_scale=0.5, thickness=1):
        """
        Draw bounding boxes and class names on an image.

        Args:
            img (numpy.ndarray): The input image.
            annotations (list): List of annotation dictionaries containing 'bb', 'cls', and 'id' keys.
            bbox_color (tuple, optional): Color of the bounding box. Defaults to (0, 0, 255) (red).
            text_color (tuple, optional): Color of the text. Defaults to (255, 255, 255) (white).
            font_scale (float, optional): Scale of the font for the class name. Defaults to 0.5.
            thickness (int, optional): Thickness of lines for the bounding box and text. Defaults to 2.
        """
        frame_data = {}
        if 'objects' in annotations:
            frame_data = annotations
            annotations = annotations['objects']

        for annotation in annotations:
            bbox = annotation['bb']
            class_name = annotation['cls']
            track_id = annotation['id']

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          bbox_color, thickness)
            text = f"{class_name} (ID: {track_id})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.putText(img, text, (int(bbox[0]), int(bbox[1]) - text_size[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        if frame_data:
            if frame_data.get('bad', False):
                cv2.putText(img, 'BAD', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.2, (0, 0, 255),
                            thickness + 1)
        return img

    def view_dataset_annotations(self, step=1):
        """
        View dataset annotations interactively using OpenCV window.

        Args:
            step (int, optional): Step size for navigation. Defaults to 1.

        HotHey:
            f d : forward/backword frame
            + - : step up/down 1
            s : save
            b g : bad/unbad current frame
            q Esc : quit

        """
        sequence_names = self.get_sequence_names()
        sequence_idx = 0
        image_idx = 0

        while True:
            sequence_name = sequence_names[sequence_idx]
            images_data = self.data[sequence_name]['frames']
            image_ids = list(images_data.keys())
            num_images = len(image_ids)

            if image_idx >= num_images:
                image_idx = 0 if sequence_idx + 1 < len(sequence_names) else num_images - 1
                sequence_idx = min(sequence_idx + 1, len(sequence_names) - 1)
            if image_idx < 0:
                image_idx = num_images - 1 if sequence_idx > 0 else 0
                sequence_idx = max(sequence_idx - 1, 0)
            if image_idx > num_images - 1:
                print('error/ ', image_idx, num_images, sequence_idx)
            image_data = images_data[image_ids[image_idx]]
            img_path = image_data['img_path']
            # annotations = image_data['objects']
            img = cv2.imread(self.get_abs_img_path(img_path))
            img = self.draw_annotations_on_image(img, image_data)

            cv2.imshow('hot key - f d + -', img)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q') or key == 27:  # q or Esc
                break
            elif key == ord('f'):  # Next image
                image_idx = image_idx + step
            elif key == ord('d'):  # Previous image
                image_idx = image_idx - step
            elif key == ord('+'):  # Increase step
                step = max(step + 1, 1)
            elif key == ord('-'):  # Decrease step
                step = max(step - 1, 1)
            elif key == ord('b'):  # Mark frame as bad
                self.data[sequence_name]['frames'][image_ids[image_idx]]['bad'] = True
            elif key == ord('g'):  # Remove bad frame mark
                if 'bad' in self.data[sequence_name]['frames'][image_ids[image_idx]]:
                    del self.data[sequence_name]['frames'][image_ids[image_idx]]['bad']
            elif key == ord('s'):  # save new json
                self.save_data_to_json()

        cv2.destroyAllWindows()

    def save_data_to_json(self, output_file=None):
        """
        Save dataset data to a JSON file.

        Args:
            output_file (str): Path to the output JSON file.
        """
        if output_file is None:
            output_file = os.path.join(self.data_folder, 'new_data.json')
            while os.path.exists(output_file):
                random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
                output_file = os.path.join(self.data_folder, f'new_data_{random_suffix}.json')

        with open(output_file, 'w') as json_file:
            json.dump(self.data, json_file, indent=4)

    def _create_uniq_file_path(self, file_path):
        """
        If a file already exists at the specified path, a new unique path will be created,
        adding random characters to the name if the file already exists. Returns the path to the file.

        Args:
            file_path (str): The original file path.

        Returns:
            str: A unique file path.
        """
        if os.path.exists(file_path):
            file_path_new, ext = os.path.splitext(file_path)
            random_suffix = random.choice(string.ascii_letters + string.digits)
            file_path_new = f"{file_path_new}{random_suffix}{ext}"
            return self._create_uniq_file_path(file_path_new)
        return file_path

    def remove_invalid_objects(self):
        """
        Remove objects with invalid coordinates (less than 0) from the dataset.
        """
        for sequence_name, sequence_data in self.data.items():
            frames_data = sequence_data['frames']

            for image_id, image_data in frames_data.items():
                if 'objects' in image_data:
                    valid_objects = []
                    for obj_info in image_data['objects']:
                        bbox = obj_info['bb']
                        if bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] >= 0 and bbox[3] >= 0:
                            valid_objects.append(obj_info)
                    image_data['objects'] = valid_objects

        print("Removed invalid objects from the dataset.")
        return self

    def create_yolo_dataset(self, output_folder, step=1, cls2id=None, rewrite_img=False):
        """
        Create a YOLO-format dataset by generating annotation files and copying images.

        Args:
            output_folder (str): Path to the output folder for the YOLO dataset.
        """

        if cls2id is None:
            cls2id = self.class_name2id

        sample_type_list = list(set(sequence_data['sample_type'] for sequence_data in self.data.values()))
        for stype in sample_type_list:
            labels_path = os.path.join(output_folder, stype, 'labels')
            images_path = os.path.join(output_folder, stype, 'images')
            os.makedirs(labels_path, exist_ok=True)
            os.makedirs(images_path, exist_ok=True)

        tamplate_image_path = os.path.join(output_folder, '{sample_type}', 'images', '{file_name}')
        tamplate_label_path = os.path.join(output_folder, '{sample_type}', 'labels', '{file_name}')


        for sequence_name, sequence_data in tqdm(self.data.items()):
            frames_data = sequence_data['frames']
            sample_type = sequence_data['sample_type']

            # sequence_output_folder = os.path.join(output_folder, sequence_name)
            # os.makedirs(sequence_output_folder, exist_ok=True)

            for i, (image_id, image_data) in enumerate(frames_data.items()):
                if i % step:
                    # processing each step frame
                    continue
                img_path = image_data['img_path']
                abs_img_path = self.get_abs_img_path(img_path)
                img_filename = os.path.basename(abs_img_path)
                to_image_file_path = tamplate_image_path.format(sample_type=sample_type,
                                                                file_name=img_filename)
                to_image_file_path = self._create_uniq_file_path(to_image_file_path)
                # имя файла могло изменится
                img_filename = os.path.basename(to_image_file_path)
                annotation_filename = img_filename.rsplit('.', 1)[0] + '.txt'
                to_annotation_file_path = tamplate_label_path.format(sample_type=sample_type,
                                                                file_name=annotation_filename)

                img_width = image_data['w']
                img_height = image_data['h']

                annotations = image_data.get('objects', [])


                with open(to_annotation_file_path, 'w') as annotation_file:
                    for annotation in annotations:
                        class_name = annotation['cls']
                        class_id = cls2id[class_name]
                        bbox = annotation['bb']

                        x_center = (bbox[0] + bbox[2]) / (2 * img_width)
                        y_center = (bbox[1] + bbox[3]) / (2 * img_height)
                        width = (bbox[2] - bbox[0]) / img_width
                        height = (bbox[3] - bbox[1]) / img_height

                        annotation_line = f"{class_id} {round(max(0, x_center), 10)} {round(max(0, y_center), 10)} {round(width, 10)} {round(height, 10)}\n"
                        annotation_file.write(annotation_line)
                if abs_img_path.endswith('.png') or rewrite_img:
                    cv2.imwrite(to_image_file_path.rsplit('.',1)[0] + '.jpg', cv2.imread(abs_img_path))
                else:
                    shutil.copy(abs_img_path, to_image_file_path)

        print(f"Created YOLO dataset for sequence in '{output_folder}'")


if __name__ == "__main__":
    # Create an instance of CVDataset
    print()
    # path = r"C:\workspace\dataset\data.json"
    # dataset = CVDataset(path)
    # dataset.print_dataset_stats()

    # f, d: forward / backword
    # + -: step up / down
    # s: save
    # b, g: bad / unbad current frame
    # q, Esc: quit

    # dataset.view_dataset_annotations()








