
class _DLBaseException(Exception):
    pass

class _TargetTransformBaseException(_DLBaseException):
    pass

class MaximumReapplyError(Exception):
    pass