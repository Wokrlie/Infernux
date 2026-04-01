"""UI system enumerations."""

from enum import IntEnum


class RenderMode(IntEnum):
    """How a UICanvas renders its content."""
    ScreenOverlay = 0     # Rendered on top of everything (screen-space)
    CameraOverlay = 1     # Rendered on top of a specific camera's output


class ScreenAlignH(IntEnum):
    """Horizontal anchor for screen-space UI layout."""
    Left = 0
    Center = 1
    Right = 2


class ScreenAlignV(IntEnum):
    """Vertical anchor for screen-space UI layout."""
    Top = 0
    Center = 1
    Bottom = 2


class TextAlignH(IntEnum):
    """Horizontal text alignment (Figma-style)."""
    Left = 0
    Center = 1
    Right = 2


class TextAlignV(IntEnum):
    """Vertical text alignment (Figma-style)."""
    Top = 0
    Center = 1
    Bottom = 2


class TextOverflow(IntEnum):
    """How text overflows its bounding box."""
    Visible = 0       # Draw beyond box
    Clip = 1          # Clip at box edge (visual only)
    Truncate = 2      # Add '…' when text is too long


class TextResizeMode(IntEnum):
    """How a UIText determines its clipping box size."""
    AutoWidth = 0
    AutoHeight = 1
    FixedSize = 2


class UIScaleMode(IntEnum):
    """How a UICanvas scales UI elements (mirrors Unity CanvasScaler)."""
    ConstantPixelSize = 0       # UI elements keep their pixel size
    ScaleWithScreenSize = 1     # Scale based on a reference resolution
    ConstantPhysicalSize = 2    # UI keeps physical size (future)


class ScreenMatchMode(IntEnum):
    """How to blend width/height when using ScaleWithScreenSize."""
    MatchWidthOrHeight = 0      # Blend between matching width or height
    Expand = 1                  # Canvas grows so nothing is cropped
    Shrink = 2                  # Canvas shrinks so everything fits


class UITransitionType(IntEnum):
    """Visual transition type for interactive elements (UISelectable)."""
    ColorTint = 0      # Tint the target graphic's color
    SpriteSwap = 1     # Swap the target image sprite (future)
    Animation = 2      # Trigger an animator state (future)
    None_ = 3          # No visual feedback
