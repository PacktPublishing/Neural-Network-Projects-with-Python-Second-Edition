import points


if __name__ == '__main__':
    x, y = points.get_points()

    # calculate the distances for various lines
    for slope, intercept in ((1.0, 0), (1/2, -1/2), (1/3, 1), (0.4, 0.1), (0.4152, -0.0253)):
        dist = points.calc_dist(x, y, slope, intercept)
        print(f"slope = {slope:.4f}, intercept = {intercept:7.4f} -> dist = {dist:8.4f}")
