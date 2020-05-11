from aiohttp import web
import aiohttp_jinja2
import argparse
import asyncio
import jinja_app_loader

from worker import Worker


@aiohttp_jinja2.template("index.html")
async def index(request):
    return dict()


async def handle(request):
    payload = await request.json()
    query = payload.get("query")
    rels = []
    if query:
        rels = request.app.worker.process(query)
    return web.json_response(rels)


async def init_app(loop, args):
    app = web.Application(loop=loop)

    aiohttp_jinja2.setup(
        app, loader=jinja_app_loader.Loader(),
        auto_reload=True, context_processors=[]
    )
    app.router.add_get("/", index)
    app.router.add_post("/handle", handle)
    app.router.add_static("/static", path="./static", name="static")
    app.router.add_static("/videos", path=args.videos, name="videos")

    app.worker = Worker(args.index, args.textual, args.glove)
    return app


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Listening port, fit current docker-compose by default.",
    )
    p.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Listening host, use default for visibility from external hosts",
    )
    p.add_argument(
        "--textual",
        type=str,
        default='./textual.zip',
        help="Path to scripted textual model"
    )
    p.add_argument(
        "--index",
        type=str,
        default='./index-test.pkl',
        help="Path to video-index"
    )
    p.add_argument(
        "--glove",
        type=str,
        default='../glove_pretrained',
        help="Path to folder with glove"
    )
    p.add_argument(
        "--videos",
        type=str,
        default="../videos",
        help="Path to local video library",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(init_app(loop, args))
    web.run_app(app, port=args.port, host=args.host)
