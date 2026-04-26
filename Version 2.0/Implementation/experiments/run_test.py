import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model import openai_model
from core.pipeline import SakshiPipeline
