"""logging integration for WMF output

The preferred method of logging in WMF infrastructure is to output
messages to stdout/stderr to be picked up by rsyslog. Structured events
(json) needs to include an `@cee` prefix so rsyslog knows how to
handle them.

Reference: https://wikitech.wikimedia.org/wiki/Logstash/Interface
"""
import logging
from pythonjsonlogger.jsonlogger import JsonFormatter


class StructuredLoggingHandler(logging.StreamHandler):
    def __init__(self, fmt, rsyslog=False):
        super(StructuredLoggingHandler, self).__init__()
        prefix = '@cee' if rsyslog else ''
        self.formatter = JsonFormatter(fmt=fmt, prefix=prefix)
