"""Explicit separate source continuation; preserves failed original attempt."""
from qualify_joint_reference_source import run, OUT


if __name__ == '__main__':
    run(OUT / 'source_continuation_v2')
