from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'xpu']
__version__ = '2.8.0'
debug = False
cuda: Optional[str] = None
git_version = 'Unknown'
hip: Optional[str] = None
xpu: Optional[str] = None
