# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""pen_art tries the macOS Arial files before falling back to Pillow's bitmap font (apple port, REQ-A-11).

The glyph walls (north: letters, east: numbers) are what the vision sense sees; the font that draws them decides
the wall art. Until 2026-11-04 _font tried four bare font file names (three Windows, one Linux; Pillow's macOS
search matches a bare name letter for letter, so none of them opened on a Mac) and then Pillow's bitmap font, so a
Mac drew the walls with the bitmap font.
"""
import pytest
from PIL import ImageFont

from mimo_bridge import pen_art

MACOS_FONTS = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf")
WINDOWS_FONTS = ("arialbd.ttf", "arial.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf")


def test_font_list_holds_the_macos_names_after_the_windows_names():
    """The two macOS Arial files are in the list by full path, after the four bare names, bold before regular; every
    name is tried before the bitmap fallback (which is not in the list)."""
    assert all(n.startswith("/System/Library/Fonts/Supplemental/") for n in pen_art.FONT_FILES[4:6])
    assert pen_art.FONT_FILES[:4] == WINDOWS_FONTS
    assert tuple(pen_art.FONT_FILES[4:6]) == MACOS_FONTS
    assert all(isinstance(n, str) and n.endswith(".ttf") for n in pen_art.FONT_FILES)


def test_font_tries_every_name_in_order_then_the_bitmap_font(monkeypatch):
    """When no font file opens (every truetype call raises OSError), _font tries every name of FONT_FILES in order
    at the asked size and returns Pillow's bitmap font."""
    tried = []

    def truetype(name, size):
        tried.append((name, size))
        raise OSError("no such font: %s" % name)

    default = object()
    monkeypatch.setattr(ImageFont, "truetype", truetype)
    monkeypatch.setattr(ImageFont, "load_default", lambda: default)
    assert pen_art._font(246) is default
    assert tried == [(name, 246) for name in pen_art.FONT_FILES]
    assert [n for n, _ in tried][4:6] == list(MACOS_FONTS)


def test_font_returns_the_first_macos_file_that_opens(monkeypatch):
    """On a machine where only the macOS Arial Bold file opens (the Windows names fail), _font returns that font
    and never reaches the bitmap fallback."""
    opened = object()

    def truetype(name, size):
        if name == MACOS_FONTS[0]:
            return opened
        raise OSError("no such font: %s" % name)

    monkeypatch.setattr(ImageFont, "truetype", truetype)
    monkeypatch.setattr(ImageFont, "load_default", lambda: pytest.fail("the bitmap fallback was reached"))
    assert pen_art._font(100) is opened
