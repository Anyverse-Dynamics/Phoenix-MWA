import argparse
import json
from pathlib import Path

from prometheus_env.websocket_policy import packb,unpackb
from .config import read_config


def make_server(interface,host="127.0.0.1",port=8009):
    from websockets.sync.server import serve
    def handle(connection):
        connection.send(packb(interface.metadata))
        for raw in connection:
            request_id = None
            try:
                if not isinstance(raw,bytes):
                    raise ValueError("Binary MessagePack required")
                request = unpackb(raw)
                if isinstance(request,dict):request_id=request.get("request_id")
                reply = interface.infer(request)
            except Exception as error:
                reply = dict(type="inference_result",status="error",ok=False,request_id=request_id,error=str(error))
            connection.send(packb(reply))
    return serve(handle,host,port,compression=None,max_size=64*1024*1024,ping_interval=None)


def main():
    p=argparse.ArgumentParser(description="Phoenix H32 model-only service")
    p.add_argument("--config",default="configs/phoenix.yaml")
    p.add_argument("--check-config",action="store_true")
    args=p.parse_args()
    cfg=read_config(args.config)
    from .model import load_interface
    interface=load_interface(cfg,check_only=args.check_config)
    print(json.dumps(interface.metadata if not args.check_config else interface,indent=2),flush=True)
    if args.check_config:return
    with make_server(interface,host=cfg["host"],port=cfg["port"]) as server:
        server.serve_forever()


if __name__=="__main__":main()
