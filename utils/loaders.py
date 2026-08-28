"""Shared convenience loader — thin wrapper so every page doesn't repeat the same three calls.

No extra caching layer here: data_processing.load_raw / run_pipeline are already
st.cache_data, so calling this from multiple pages in the same session is free
after the first call.
"""
from . import data_processing as dp


def get_data():
    raw, data_dict, issues = dp.load_raw()
    result = dp.run_pipeline(raw)
    return raw, data_dict, issues, result
