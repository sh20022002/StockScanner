"""
Opt-in HTTP Basic Auth gate for the monitor station.

Off by default — server/run.py already warns that the app has no
authentication and shouldn't be bound to 0.0.0.0 outside a trusted network.
Set both SMARTRAID_USER and SMARTRAID_PASSWORD to turn this on before
exposing the app beyond localhost; leaving either unset keeps local dev
frictionless, same as EXCHANGE_API_KEY elsewhere in this codebase.

This is a shared-password gate for a handful of trusted people, not a real
multi-user auth system — there's one username/password for the whole app,
same as everyone hitting the same in-process scanner state.
"""
import base64
import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


def verify_basic_auth(header_value: str | None, expected_user: str,
                      expected_password: str) -> bool:
    """
    True iff header_value is a well-formed 'Basic <base64(user:password)>'
    Authorization header whose credentials match expected_user/expected_password.

    Uses secrets.compare_digest for the comparison so a correct username
    can't be distinguished from an incorrect one by response timing.

    Pulled out as a pure function so it's testable without spinning up the
    ASGI middleware — see server/tests/test_web.py.
    """
    if not header_value or not header_value.startswith('Basic '):
        return False
    try:
        decoded = base64.b64decode(header_value[len('Basic '):]).decode('utf-8')
    except Exception:
        return False
    user, sep, password = decoded.partition(':')
    if not sep:
        return False
    return (secrets.compare_digest(user, expected_user)
            and secrets.compare_digest(password, expected_password))


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """
    Gates every request (HTML, static assets, API, SSE stream alike) behind
    HTTP Basic Auth when SMARTRAID_USER/SMARTRAID_PASSWORD are set.

    Read from the environment on every request rather than cached at import
    time — cheap (two os.getenv calls), and it means tests can toggle the
    env vars without reimporting this module.
    """
    async def dispatch(self, request, call_next):
        expected_user = os.getenv('SMARTRAID_USER')
        expected_password = os.getenv('SMARTRAID_PASSWORD')
        if not (expected_user and expected_password):
            return await call_next(request)

        if verify_basic_auth(request.headers.get('authorization'),
                             expected_user, expected_password):
            return await call_next(request)

        return Response(status_code=401, headers={'WWW-Authenticate': 'Basic realm="SmarTraid"'})
