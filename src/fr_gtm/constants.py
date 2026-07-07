# State vector:
# pos(3) + vel_body(3) + att_quat(4) + omega(3) + act_pos(4) + dryden(8) = 25
XDIM = 25
# Control vector: act_cmd(4) = 4
UDIM = 4
# Disturbance vector: dryden_noise(4) = 4
DDIM = 4
# Tunable parameter vector: wind(5) + uncertainty(28) = 33
PDIM = 33
# Noise vector: sensor_noise(21) = 21
NDIM = 21
# Output vector: gamma(1) + alpha(1) + beta(1) + V(1) + altitude(1) + euler(3) +
#                omega(3) + nz(1) + qbar(1) + act_pos(4) + act_cmd(4) = 21
YDIM = 21
