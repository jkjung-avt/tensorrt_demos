"""writer.py
"""


import subprocess

import cv2


def get_video_writer(name, width, height, fps=30):
    """Get a VideoWriter object for saving output video.

    This function tries to use Jetson's hardware H.264 encoder (omxh264enc)
    if available, in which case the output video would be a MPEG-2 TS file.
    Otherwise, it uses cv2's built-in encoding mechanism and saves a MP4
    file.
    """
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'omxh264dec' in gst_elements:
        filename = name + '.ts'  # Transport Stream
        gst_str = ('appsrc ! videoconvert ! omxh264enc ! mpegtsmux ! '
                   'filesink location=%s') % filename
        return cv2.VideoWriter(
            gst_str, cv2.CAP_GSTREAMER, 0, fps, (width, height))
    else:
        filename = name + '.mp4'  # MP4
        return cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


