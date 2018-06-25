from imgaug import augmenters as iaa


class NormalizedUtil(object):

  @staticmethod
  def normalized_operation_to_imgaug_operation(name, magnitude):

    imgaug_operation = None

    if name == "Fliplr":
      imgaug_magnitude = 0.5
      imgaug_operation = iaa.Fliplr(imgaug_magnitude)
    elif name == "Crop":
      imgaug_magnitude = 0.5
      imgaug_operation = iaa.Crop(percent=(0, 0.1))
    elif name == "GaussianBlur":
      imgaug_operation = iaa.GaussianBlur(sigma=(0, 0.5))
    elif name == "ContrastNormalization":
      imgaug_operation = iaa.ContrastNormalization((0.75, 1.5))
    elif name == "AdditiveGaussianNoise":
      imgaug_operation =  iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
    elif name == "Multiply":
      imgaug_operation = iaa.Multiply((0.8, 1.2), per_channel=0.2),

    return imgaug_operation
