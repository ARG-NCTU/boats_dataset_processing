#!/usr/bin/env python3
"""
STAMP API Server Launcher
=========================

Usage:
    python run_server.py              # Run on localhost:8000
    python run_server.py --host 0.0.0.0 --port 8080
    python run_server.py --reload     # Auto-reload on code changes
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="STAMP API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0 for all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, recommended for GPU)",
    )
    
    args = parser.parse_args()
    
    print(f"""
    ╔═══════════════════════════════════════════╗
    ║         STAMP API Server                  ║
    ╠═══════════════════════════════════════════╣
    ║  Host: {args.host:<15}                    ║
    ║  Port: {args.port:<15}                    ║
    ║  URL:  http://{args.host}:{args.port:<10}           ║
    ║  Docs: http://{args.host}:{args.port}/docs          ║
    ╚═══════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
