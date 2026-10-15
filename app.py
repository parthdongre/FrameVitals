"""Source-checkout compatibility entrypoint; installed API lives in framevitals.webapp."""
import sys
from framevitals import webapp

if __name__ == '__main__':
    webapp.serve()
else:
    # Preserve historical imports and monkeypatch semantics for existing integrations.
    sys.modules[__name__] = webapp
