import dct_transformer


class MultiResolution:
    def __init__(self):
        pass

    def __call__(self, pic):
        multiresolution_img = dct_transformer.transfom_to_multiple_images(pic)
        return multiresolution_img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"