class ErrorBase(Exception):
    '''
        Base eval error
    '''

    def __init__(self, code=599, msg="Unknown Error"):
        Exception.__init__(self)
        self.code = code
        self.message = msg

    def __str__(self):
        return self.message


class ErrorConfig(ErrorBase):
    '''
        Config Error
    '''

    def __init__(self, value):
        self.value = value
        msg = "no " + str(self.value) + " config set"
        ErrorBase.__init__(self, 400, msg)


class ErrorFileNotExist(ErrorBase):
    '''
        Error File Not Exist
    '''

    def __init__(self, name):
        self.name = name
        msg = "file not exist: {}".format(self.name)
        ErrorBase.__init__(self, 400, msg)


class ErrorCV2ImageRead(ErrorBase):
    '''
        CV2 Image Read Fail
    '''

    def __init__(self, image):
        self.image = image
        msg = "cv2 load {} failed".format(self.image)
        ErrorBase.__init__(self, 400, msg)


class ErrorImageTooSmall(ErrorBase):
    '''
        Image too small
    '''

    def __init__(self, image):
        self.image = image
        msg = "image {} is too small, should be more than 32x32".format(
            self.image)
        ErrorBase.__init__(self, 400, msg)


class ErrorImageNdim(ErrorBase):
    '''
        Image with invalid ndim
    '''

    def __init__(self, image):
        self.image = image
        msg = "image {} with invalid ndim, should be 3".format(self.image)
        ErrorBase.__init__(self, 400, msg)


class ErrorImageTooLarge(ErrorBase):
    '''
        Image too large
    '''

    def __init__(self, image):
        self.image = image
        msg = "image {} is too large, should be in 4999x4999, less than 10MB".format(
            self.image)
        ErrorBase.__init__(self, 400, msg)


class ErrorInvalidPTS(ErrorBase):
    '''
        Invalid PTS
        pts: a list [[xl, yt],[xr, yt], [xr, yb], [xl, yb], ...]
             or a tuple ([xl, yt],[xr, yt], [xr, yb], [xl, yb], ...)
    '''

    def __init__(self, pts):
        self.pts = pts
        msg = "invalid pts {}".format(self.pts)
        ErrorBase.__init__(self, 400, msg)


class ErrorOutOfBatchSize(ErrorBase):
    '''
        eval out of batch_size
        batch_size: forward batch_size of network
    '''

    def __init__(self, batch_size):
        self.batch_size = batch_size
        msg = "eval request out of batch_size {}".format(self.batch_size)
        ErrorBase.__init__(self, 400, msg)


class ErrorTransformImage(ErrorBase):
    '''
        tramsform image to net numpy failed
    '''

    def __init__(self, image):
        self.image = image
        msg = "fail to transform image {}".format(self.image)
        ErrorBase.__init__(self, 400, msg)


class ErrorNoPTS(ErrorBase):
    '''
        no pts provided
    '''

    def __init__(self, image):
        self.image = image
        msg = "no pts provided for image {}".format(self.image)
        ErrorBase.__init__(self, 400, msg)


class ErrorForwardInference(ErrorBase):
    '''
        forward inference failed
    '''

    def __init__(self):
        msg = "forward inference failed"
        ErrorBase.__init__(self, 500, msg)
