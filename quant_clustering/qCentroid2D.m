function xq = qCentroid2D(x, C1, C2)
    xq = [nearest_centroid_1d(x(1), C1); nearest_centroid_1d(x(2), C2)];
end