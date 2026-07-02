# Sensor Error Formulas

This document summarises the error models used by each sensor in the simulator.

## GPS
Position measurement is modelled as 2D Gaussian noise around the true position:

- x_meas = x_true + N(0, σ_gps²)
- y_meas = y_true + N(0, σ_gps²)

where σ_gps is the sensor's base noise parameter. In challenging environments it may increase based on signal quality or range.

## Camera
Camera position error depends on range and viewing angle:

- σ_cam = σ0 + k_r · r + k_θ · |θ|
- x_meas = x_true + N(0, σ_cam²)
- y_meas = y_true + N(0, σ_cam²)

where r is the distance from the camera to the vehicle, and θ is the off-axis viewing angle relative to the camera heading.

## DAS
Signal amplitude decays with distance from the fiber:

- A = W / (r + d0)²

where W is the source intensity (vehicle weight), r is the distance from the fiber, and d0 is a small floor distance to avoid singularity at r = 0.

Signal-to-noise ratio is derived from amplitude:

- SNR = A / σ_noise

Position uncertainty is approximated as:

- σ_DAS = k / sqrt(SNR)

Noise is injected as a Gaussian around the true position or around the projection onto the fiber, depending on the event type.

## Speed
For sensors that also provide a speed measurement, the basic model is:

- v_meas = v_true + N(0, σ_v²)

## Kalman Filter
The filter state vector is:

- X = [x, vx, ax, y, vy, ay]

The motion model is Constant Acceleration:

- x(t+dt) = x(t) + vx·dt + 0.5·ax·dt²
- vx(t+dt) = vx(t) + ax·dt
- ax(t+dt) = ax(t)

The same applies for the y-axis.
