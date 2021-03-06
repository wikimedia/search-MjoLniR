"""logging integration for WMF output

The preferred method of logging in WMF infrastructure is to output
messages to stdout/stderr to be picked up by rsyslog. Structured events
(json) needs to include an `@cee` prefix so rsyslog knows how to
handle them.

Reference: https://wikitech.wikimedia.org/wiki/Logstash/Interface
"""
import logging
from pythonjsonlogger.jsonlogger import JsonFormatter


class CustomJsonFormatter(JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['level'] = record.levelname.upper()


class StructuredLoggingHandler(logging.StreamHandler):
    def __init__(self, fmt, rsyslog=False):
        super(StructuredLoggingHandler, self).__init__()
        prefix = '@cee: ' if rsyslog else ''
        self.formatter = CustomJsonFormatter(fmt=fmt, prefix=prefix)
