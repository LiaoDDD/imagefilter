# logging_config.py
import logging, json, sys, time, logging.handlers, os

class JsonFormatter(logging.Formatter):
    def format(self, rec):
        return json.dumps({
            "ts":  time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(rec.created)),
            "lvl": rec.levelname,
            "msg": rec.getMessage(),
            "mod": rec.name
        }, ensure_ascii=False)

def init_logger(path="/app/data/logs"):
    if os.path.isdir(path):
        path = os.path.join(path, "service.log")
    elif path.endswith("/"):
        path = os.path.join(path, "service.log")
    lg = logging.getLogger("svc"); lg.setLevel(logging.INFO)
    h1 = logging.StreamHandler(sys.stdout)
    h1.setFormatter(JsonFormatter())
    h2 = logging.handlers.RotatingFileHandler(
        path, maxBytes=50_000_000, backupCount=3, encoding="utf-8")
    h2.setFormatter(JsonFormatter())
    lg.addHandler(h1); lg.addHandler(h2)
    logging.getLogger("aiohttp.client").setLevel(logging.DEBUG)
    return lg
