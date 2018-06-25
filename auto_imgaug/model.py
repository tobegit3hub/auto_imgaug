import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


class AutoImgaugOperation(object):
  def __init__(self, operation_name="Fliplr", magnitude=1.0, probability=1.0):
    # translation, rotation, or shearing,
    # 16 operators: ShearX/Y, TranslateX/Y, Rotate, AutoContrast, Invert, Equalize, Solarize, Posterize, Contrast, Color, Brightness, Sharpness, Cutout, Sample Pairing
    self.operation_name = operation_name
    # 10 values: [0, 10]
    self.magnitude = magnitude
    # 11 values: [0, 10],
    self.probability = probability
    self.imgaug_operation = self.normalized_to_imgaug_operation(
        self.operation_name, self.magnitude)

  def __str__(self):
    return "operation: {}, magnitude: {}, probability: {}, imgaug_operation: {}".format(
        self.operation_name, self.magnitude, self.probability,
        self.imgaug_operation)

  def normalized_to_imgaug_operation(self, operation_name, magnitude):
    # TODO: Make this to static method
    imgaug_operation = None

    if operation_name == "Fliplr":
      imgaug_magnitude = 0.5
      imgaug_operation = iaa.Fliplr(imgaug_magnitude)
    elif operation_name == "Crop":
      imgaug_magnitude = 0.5
      imgaug_operation = iaa.Crop(percent=(0, 0.1))

    return imgaug_operation

  def process(self, input_ndarray):
    seq = iaa.Sequential([self.imgaug_operation], random_order=True)
    output_ndarray = seq.augment_images(input_ndarray)
    return output_ndarray


class AutoImgaugPolicy(object):
  def __init__(self):
    self.num_of_operations = 3
    self.operations = []

  def __str__(self):
    return "operations: {}".format(self.operations)

  def init_with_params(self, params):
    operation1 = AutoImgaugOperation(params["operation_name1"],
                                     params["magnitude1"],
                                     params["probability1"])
    operation2 = AutoImgaugOperation(params["operation_name2"],
                                     params["magnitude2"],
                                     params["probability2"])
    operation3 = AutoImgaugOperation(params["operation_name3"],
                                     params["magnitude3"],
                                     params["probability3"])

    self.operations = [operation1, operation2, operation3]

  def init_with_default_operations(self):
    operation1 = AutoImgaugOperation("Fliplr", 5, 0.7)
    operation2 = AutoImgaugOperation("Crop", 7, 0.2)
    operation3 = AutoImgaugOperation("Fliplr", 9, 0.8)

    self.operations = [operation1, operation2, operation3]

  def process(self, input_ndarray):
    """  
    output_ndarray = input_ndarray
    for operation in self.operations:
      output_ndarray = operation.process(output_ndarray)

    return output_ndarray
    """

    imgaug_sequential_list = []

    for operation in self.operations:
      imgaug_sequential_list.append(operation.imgaug_operation)

    imgaug_sequential = iaa.Sequential(
        imgaug_sequential_list, random_order=False)

    output_ndarray = imgaug_sequential.augment_images(input_ndarray)

    return output_ndarray
