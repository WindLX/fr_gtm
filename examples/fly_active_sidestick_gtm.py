"""Fly GTM with ActiveSidestick elevator/aileron and T16000 rudder/throttle."""

from .fly_stick_gtm import main

if __name__ == "__main__":
    raise SystemExit(main(active_sidestick_default=True))
