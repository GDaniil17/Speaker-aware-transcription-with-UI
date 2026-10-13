"""Shared test setup."""

import os

# Tests never talk to the Hugging Face Hub: lookups such as download sizes fail
# fast and the code under test takes its offline path.
os.environ["HF_HUB_OFFLINE"] = "1"
