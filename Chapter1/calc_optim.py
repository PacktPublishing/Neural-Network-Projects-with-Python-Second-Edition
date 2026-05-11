import points
import numpy as np


if __name__ == '__main__':
    x, y = points.get_points()

    # initial values for slope and intercept - could be anything
    slope = 1.0
    intercept = 0.0

    # steps for the slope and intercept
    steps = [0.1, 0.1]

    prev_d = points.calc_dist(x, y, slope, intercept)
    for i in range(200):
        # update slope and intercept in turns
        if i % 2 == 0:
            d = points.calc_dist(x, y, slope + steps[0], intercept)
            if prev_d < d:
                steps[0] = -steps[0]/2
            else:
                slope += steps[0]
                prev_d = d
        else:
            d = points.calc_dist(x, y, slope, intercept + steps[1])
            if prev_d < d:
                steps[1] = -steps[1]/2
            else:
                intercept += steps[1]
                prev_d = d
        print(f"{i}: {prev_d} -> {d}: slope={slope:.4f}, intercept={intercept:.4f}, steps={steps}")
        if np.max(np.abs(steps)) < 1e-4:
            break
    print(f"final: slope={slope:.4f}, intercept={intercept:.4f}")