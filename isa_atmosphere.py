def T(h: float) -> float:
    """
    Temperature [K] at geopotential altitude h [m] per ISA Standard Atmosphere 1976.
    Valid range: 0 – 86,000 m. Raises ValueError outside that range.
    """
    # (h_base_m, T_base_K, lapse_rate_K_per_km)
    # Positive lapse rate → temperature drops with altitude.
    LAYERS = (
        (     0,   288.150,  6.5),
        (11_000,   216.650,  0.0),
        (20_000,   216.650, -1.0),
        (32_000,   228.650, -2.8),
        (47_000,   270.650,  0.0),
        (51_000,   270.650,  2.8),
        (71_000,   214.650,  2.0),
        (84_852,   186.946,  0.0),  # Mesopause — upper boundary
    )

    if h < 0:
        raise ValueError(f"Altitude {h} m is below sea level (h must be >= 0)")
    if h > 86_000:
        raise ValueError(f"Altitude {h} m exceeds model ceiling of 86,000 m")

    for i in range(len(LAYERS) - 1):
        h_base, T_base, lapse = LAYERS[i]
        h_top = LAYERS[i + 1][0]
        if h <= h_top:
            return T_base - lapse * (h - h_base) / 1000.0

    # h == 84,852 exactly
    return LAYERS[-1][1]


if __name__ == "__main__":
    checkpoints = [0, 11_000, 20_000, 32_000, 47_000, 51_000, 71_000, 84_852]
    expected_K  = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]

    print(f"{'h (m)':>8}  {'T (K)':>9}  {'expected':>9}  {'delta':>8}")
    for h, exp in zip(checkpoints, expected_K):
        got = T(h)
        print(f"{h:>8}  {got:>9.3f}  {exp:>9.3f}  {got - exp:>+8.4f}")
