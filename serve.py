"""Serve the self-contained artifact locally. No network/CDN dependencies."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from functools import partial

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    handler=partial(SimpleHTTPRequestHandler,directory=str(root/'web'))
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler)
    print(f'Retractable Imagination: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
