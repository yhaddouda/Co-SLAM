"""Optional telemetry only. Required power/CUDA/ownership checks never use this helper."""
import os


def read_optional_sysfs(path):
    """Return (text, None) or (None, explicit field/error details).

    Some Orin thermal sysfs nodes return EAGAIN. Buffered TextIO can turn its
    underlying None read into a decoder TypeError. os.read exposes the OSError.
    """
    field = os.fspath(path)
    try:
        descriptor = os.open(field, os.O_RDONLY | os.O_CLOEXEC)
        try:
            raw = os.read(descriptor, 4096)
        finally:
            os.close(descriptor)
        if not isinstance(raw, bytes):
            raise TypeError('optional sysfs read returned ' + type(raw).__name__ + ', expected bytes')
        return raw.decode('utf-8').strip(), None
    except (OSError, UnicodeError, TypeError) as error:
        return None, dict(field=field, status='unavailable', exception_type=type(error).__name__,
                          message=str(error), errno=getattr(error, 'errno', None))
