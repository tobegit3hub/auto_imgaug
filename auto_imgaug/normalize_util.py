from imgaug import augmenters as iaa


class NormalizedUtil(object):
  @staticmethod
  def normalized_operation_to_imgaug_operation(operation_name, magnitude):

    imgaug_operation = None

    if operation_name == "Fliplr":
      imgaug_magnitude = 0.5
      imgaug_operation = iaa.Fliplr(imgaug_magnitude)
    elif operation_name == "Crop":
      imgaug_magnitude = 0.5
      imgaug_operation = iaa.Crop(percent=(0, 0.1))

    return imgaug_operation
