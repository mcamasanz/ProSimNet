"""
ProSimNet control library — time-variable and state-feedback BC signals.

Modules
-------
signals     Time-variable signal primitives: ramp, step, pulse, piecewise, sine.
controllers Feedback and feedforward controllers: onoff, proportional, feedforward.

All signal primitives and controllers return callables compatible with
``src.utils.signals.resolve(signal, t, snap)``.

Quick start::

    from src.control.signals import ramp, step
    from src.control.controllers import proportional

    # Ramp T_wall from 700 K to 1000 K over the first 600 s
    tbc_cfg["T_wall"] = ramp(t_start=0.0, slope=(1000-700)/600,
                             value_init=700.0, value_max=1000.0)

    # Control T_wall proportionally to keep Ts_mean at 900 K
    tbc_cfg["T_wall"] = proportional(
        setpoint=900.0, gain=5.0,
        channel_in=lambda snap: snap.get("Ts_mean", 900.0),
        output_bias=1073.15, output_min=700.0, output_max=1200.0,
    )
"""
