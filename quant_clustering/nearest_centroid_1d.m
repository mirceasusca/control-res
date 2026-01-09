function c = nearest_centroid_1d(val, C)
    [~,i] = min(abs(C - val));
    c = C(i);
end