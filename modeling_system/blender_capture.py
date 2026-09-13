"""Geometry-only framing; native capture belongs to the configured workspace adapter."""
import math

def context_framing(bounds, yaw, aspect=1.2, margin=1.08):
    """Fit the recorded face/bust bounds with a margin at the chosen yaw."""
    low, high = bounds
    center = tuple(((a + b) / 2 for a, b in zip(low, high)))
    half = tuple(((b - a) / 2 for a, b in zip(low, high)))
    angle = math.radians(yaw)
    horizontal = 2 * (abs(math.cos(angle)) * half[0] + abs(math.sin(angle)) * half[1])
    return (center, max(horizontal, 2 * half[2] * aspect) * margin)
