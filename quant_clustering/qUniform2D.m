function xq = qUniform2D(x, q1, q2)
    xq = [round(x(1)/q1)*q1;round(x(2)/q2)*q2];
end