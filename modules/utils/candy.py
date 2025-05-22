import inspect

def trace_str(skip_frames=0) -> str:
    """
    返回调用位置的 [文件名:行号]
    :param skip_frames: 跳过栈帧数（用于嵌套调用）
    """
    frame = inspect.currentframe()
    # 回溯到调用者的栈帧
    for _ in range(skip_frames + 1):
        frame = frame.f_back
    # 提取文件名和行号
    filename = frame.f_code.co_filename
    line_no = frame.f_lineno
    return f"{filename}:{line_no}"
def _debug(*args, **kwargs):
    print("DEBUG:", trace_str(1), *args, **kwargs)
