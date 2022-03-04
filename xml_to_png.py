import os
from xml.dom import minidom
import xml
import gc
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import skimage.io as io
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')

###################################input path#######################################
img_path = "/home/pkwmay/202103_Hallym_leenuri/dataset/01_seg_train_dataset/image/"
xml_path = "/home/pkwmay/202103_Hallym_leenuri/dataset/01_seg_train_dataset/label/"
dst_path = "/home/pkwmay/202103_Hallym_leenuri/dataset/01_seg_train_dataset/mask/"
####################################################################################

def parse_xml(xml_file_path):
    """read xml file and return a parsed object from it

    Args:
        xml_file_path (str): path to the xml file
    Returns:
        parsed object from the xml file

    Example:
        > > > parse_xml('1035134_.xml')
        <parsed object>
    """
    
    xml_content = open(xml_file_path, "r")
    xml_doc = minidom.parse(xml_content)
    xml_content.close()

    return xml_doc


def create_mask_from_xml(xml_file, wsi_dimension, save, save_dir=None):
    """Takes the xml object, parse, get annotation information for each class.
    When new class is detected, new label mask image is created for the information to be drawn.
    Through this function, there will be n-number, number of classes in the xml file, plus an additional
    mask image that represents all the classes, as a total mask image. Will return a tuple of lists,
    one with names of the classes, and the other with mask images of the classes.

    Args:
        xml_file (str): path to the xml file
        wsi_dimension ((int, int)): dimension for the new masks to be created. Same with the dimension of the WSI.
        save (bool): toggle for debugging mode. If true, will save resulting mask images
        save_dir(str): if save is true, give a directory path to locate where to save the resulting images.

    Returns:
        List<str>: list of names of the classes with last class 'TOTAL'
        List<PIL.Image>: list of mask images of the classes with las class 'TOTAL' mask image

    Example:
        > > > create_mask_from_xml('1035134_.xml', (10234, 4547))
        (List<str>, List<PIL.Image>)
    """

    height, width = wsi_dimension[1], wsi_dimension[0]
    parsed_xml = parse_xml(xml_file)
    cls_lst = []
    cls_img_lst = []
    total_img = Image.new("L", (height, width), 0)

    for i, annotation in enumerate(parsed_xml.getElementsByTagName("Annotation")):
        cls = annotation.getAttribute("class")

        if not (cls in cls_lst):
            cls_lst.append(cls)
            cls_img_lst.append(Image.new("L", (height, width), 0))

        for coordinates in annotation.childNodes:
            list_of_coordinates = []
            
            if isinstance(coordinates, minidom.Text):
                continue

            if coordinates.tagName == "Coordinates":
                for coordinate in coordinates.childNodes:
                    if isinstance(coordinate, minidom.Element):
                        if coordinate.tagName == "Coordinate":
                            x = float(coordinate.getAttribute("x"))
                            y = float(coordinate.getAttribute("y"))
                            list_of_coordinates.append((x, y))

            idx = cls_lst.index(cls)

            try:
                if len(list_of_coordinates) != 0:
                    ImageDraw.Draw(cls_img_lst[idx]).polygon(
                        list_of_coordinates, outline=1, fill=255)
                    ImageDraw.Draw(total_img).polygon(
                        list_of_coordinates, outline=1, fill=255)
            except:
                pass

            gc.collect()

    cls_img_lst.append(total_img)
    cls_lst.append("TOTAL")

    if save:
        for name, img in zip(cls_lst, cls_img_lst):
            img.save(os.path.join(save_dir, name[:-4] + name + "png"))

    return cls_lst, cls_img_lst

def get_all_files_from_directory(path):
    files = os.listdir(path)
    
    return files



files = get_all_files_from_directory(img_path)

for k, file_name in tqdm(enumerate(files)):
    # print("[", k, "/", len(files), "]", file_name)
    dst_fn = os.path.join(dst_path, file_name[:-4] + ".png")

    if os.path.exists(dst_fn):
        # print("File exist")
        continue

    if file_name[:-4] == "validation_data_03":
        continue

    wsi_fn = os.path.join(img_path, file_name)
    xml_fn = os.path.join(xml_path, file_name[:-4] + ".xml")
    ops_img = cv2.imread(wsi_fn)
    max_dim = ops_img.shape

    try:
        class_name_list, class_mask_list = create_mask_from_xml(xml_fn, max_dim, False)
    except (xml.parsers.expat.ExpatError) as e:
        print(e)
        continue
    
    for mask in class_mask_list:
        mask = np.asarray(mask)
        io.imsave(dst_fn, mask)
