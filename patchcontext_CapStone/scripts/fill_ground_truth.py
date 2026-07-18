"""
Fill real ground_truth answers into benchmark_questions.json.
Run from the patchcontext/ directory:
    python scripts/fill_ground_truth.py
"""

import json
from pathlib import Path

BENCHMARK_PATH = Path("data/benchmark_questions.json")

# Real, factual reference answers for each question.
# Keyed by the exact question string.
GROUND_TRUTHS = {
    "Why does FastAPI use Depends() for dependency injection instead of another approach?": (
        "FastAPI uses Depends() for dependency injection because it integrates cleanly with "
        "Python type hints and allows dependencies to be declared as regular callables. "
        "This design means dependencies can themselves have dependencies (sub-dependencies), "
        "forming a tree that FastAPI resolves automatically. It also allows the same dependency "
        "to be reused across multiple path operations and supports caching within a single request. "
        "The Depends() approach avoids class-based views while still enabling shared logic like "
        "authentication, database sessions, and configuration injection."
    ),
    "Why does FastAPI use Pydantic for request and response validation?": (
        "FastAPI uses Pydantic for request and response validation because Pydantic leverages "
        "Python type hints to define data schemas declaratively. This allows FastAPI to "
        "automatically parse incoming request bodies, query parameters, and path parameters, "
        "validate them against the declared types, and return clear error messages when validation "
        "fails. Pydantic also enables automatic JSON serialization of response models, ensuring "
        "that only the declared fields are returned. The tight integration means the same models "
        "are used for validation, serialization, and OpenAPI schema generation."
    ),
    "Why is FastAPI built on top of Starlette instead of being a standalone framework?": (
        "FastAPI is built on top of Starlette to avoid reinventing low-level ASGI handling, "
        "routing, middleware, and WebSocket support. Starlette provides a solid, tested ASGI "
        "foundation, and FastAPI adds the API-specific layer on top: dependency injection, "
        "Pydantic validation, automatic OpenAPI docs generation, and type-hint-driven parameter "
        "handling. This means FastAPI inherits Starlette's performance characteristics and "
        "ecosystem (middleware, background tasks, static files) while focusing only on the "
        "API development experience."
    ),
    "Why does FastAPI generate OpenAPI documentation automatically?": (
        "FastAPI generates OpenAPI documentation automatically because it introspects the "
        "declared type hints, Pydantic models, and path operation function signatures at "
        "startup. Since all parameter types and response models are already declared in Python "
        "code, FastAPI can derive the complete OpenAPI schema without any additional annotation. "
        "This means developers get interactive Swagger UI and ReDoc documentation for free, "
        "always in sync with the actual API, and with no separate documentation maintenance burden."
    ),
    "Why does FastAPI support both async and sync path operation functions?": (
        "FastAPI supports both async (async def) and sync (def) path operation functions to "
        "accommodate different use cases. Async functions run directly on the event loop and "
        "are ideal for I/O-bound operations like database queries or HTTP calls. Sync functions "
        "are automatically run in a thread pool executor by FastAPI to avoid blocking the event "
        "loop. This means developers can use synchronous libraries (like standard ORMs or "
        "blocking SDKs) without needing to rewrite them, while still benefiting from FastAPI's "
        "async core."
    ),
    "Why does FastAPI use Python type hints as the primary API for defining request parameters?": (
        "FastAPI uses Python type hints as its primary API because they allow the same "
        "declaration to serve multiple purposes simultaneously: editor autocomplete, static "
        "type checking, runtime validation via Pydantic, and OpenAPI schema generation. "
        "By reading the type annotations on path operation function parameters, FastAPI "
        "automatically determines whether a parameter comes from the path, query string, "
        "request body, or headers, and what type it should be. This eliminates duplicated "
        "declarations and keeps the code concise and self-documenting."
    ),
    "What was the motivation for adding the 'response_model' parameter to path operations?": (
        "The response_model parameter was added to allow developers to declare what the "
        "response schema should be independently of what the function actually returns. "
        "This means an endpoint can return an ORM model or a dict internally, but FastAPI "
        "will filter and serialize the output to match only the declared response_model fields. "
        "It prevents accidentally leaking sensitive fields (like passwords) from internal "
        "models and enables FastAPI to include the correct response schema in OpenAPI docs. "
        "It also enables response validation and automatic JSON serialization."
    ),
    "Why does FastAPI's security utilities (OAuth2, HTTP Basic) integrate with the dependency injection system?": (
        "FastAPI's security utilities integrate with the dependency injection system because "
        "authentication and authorization are naturally cross-cutting concerns that need to be "
        "reused across many endpoints. By implementing OAuth2PasswordBearer, HTTPBasic, and "
        "other security schemes as callables compatible with Depends(), they can be composed "
        "with other dependencies and declared once but applied to many path operations. "
        "This also means the security scheme appears correctly in the OpenAPI docs, enabling "
        "Swagger UI's Authorize button to work out of the box."
    ),
    "Why does FastAPI use path operation decorators (@app.get, @app.post) instead of a class-based routing approach?": (
        "FastAPI uses decorator-based routing (@app.get, @app.post) because it is more explicit "
        "and aligns naturally with Python functions as the unit of work. Each path operation "
        "decorator directly annotates the function that handles the route, making it easy to "
        "see the method, path, and handler in one place. This approach avoids the boilerplate "
        "of class-based views while still allowing route grouping via APIRouter. It also makes "
        "the code more testable since each handler is a plain Python function."
    ),
    "What was the rationale for introducing APIRouter to split applications into multiple files?": (
        "APIRouter was introduced to allow large FastAPI applications to be split across "
        "multiple files and modules without losing any functionality. Each router can define "
        "its own path operations, dependencies, prefix, and tags, and then be included into "
        "the main app with app.include_router(). This mirrors how Flask Blueprints work and "
        "enables a clean modular project structure. It also allows teams to work on different "
        "routers independently and reuse routers across applications."
    ),
    "Why does FastAPI's CORS middleware use the Starlette implementation rather than a custom one?": (
        "FastAPI uses Starlette's CORSMiddleware because CORS handling is a well-specified "
        "HTTP concern that Starlette already implements correctly and maintains. Since FastAPI "
        "is built on Starlette, it can simply re-export the same middleware without duplicating "
        "code or introducing a custom implementation that would need separate maintenance. "
        "This means FastAPI developers get a battle-tested CORS implementation and Starlette "
        "upstream improvements automatically."
    ),
    "Why does FastAPI return 422 Unprocessable Entity for validation errors instead of 400 Bad Request?": (
        "FastAPI returns 422 Unprocessable Entity for validation errors because 422 is the "
        "semantically correct HTTP status code for this case: the request was well-formed at "
        "the HTTP level but the contained data failed semantic validation. HTTP 400 Bad Request "
        "is intended for syntactically malformed requests. Pydantic validation errors (wrong "
        "type, missing required field, value out of range) are semantic errors on otherwise "
        "parseable data, making 422 the most accurate status code per the HTTP specification."
    ),
    "What was the design rationale for BackgroundTasks in FastAPI?": (
        "BackgroundTasks were added to allow path operations to trigger work after the response "
        "has been sent to the client, without blocking the response. This is useful for tasks "
        "like sending emails, writing to a slow log, or triggering notifications that the client "
        "does not need to wait for. FastAPI's BackgroundTasks integrates with Starlette's "
        "background task system and is injected via dependency injection, making it easy to "
        "use without external task queues. For more complex needs, dedicated task queues like "
        "Celery are recommended."
    ),
    "Why does FastAPI use the 'lifespan' context manager for startup and shutdown events instead of @app.on_event?": (
        "FastAPI adopted the lifespan context manager (using @asynccontextmanager) as the "
        "recommended pattern for startup and shutdown events because it keeps both startup and "
        "teardown logic together in a single place, making it easier to reason about resource "
        "lifecycle. The older @app.on_event('startup') and @app.on_event('shutdown') decorators "
        "were split and could become inconsistent. The lifespan approach also aligns with "
        "Starlette's recommended pattern and makes it easier to share state initialized at "
        "startup (like a database connection pool) across path operations via the app.state object."
    ),
    "Why does FastAPI's TestClient use requests under the hood instead of httpx directly?": (
        "FastAPI's TestClient is inherited from Starlette and uses the requests library under "
        "the hood via the ASGI transport adapter. This allows tests to be written using the "
        "familiar requests API while actually dispatching requests through the ASGI app in-process "
        "without a running server. For async testing, FastAPI also supports using httpx with "
        "AsyncClient and the ASGITransport. The requests-based TestClient remains the default "
        "for synchronous tests due to its wide familiarity."
    ),
    "What was the motivation for FastAPI's approach to handling form data and file uploads?": (
        "FastAPI handles form data and file uploads by declaring them with Form() and File() "
        "parameter annotations, consistent with how all other parameter types are declared. "
        "This means validation, type coercion, and OpenAPI documentation work the same way "
        "for form fields as for JSON body fields. UploadFile provides an async-compatible file "
        "interface with metadata. The approach avoids a separate multipart parsing API and "
        "keeps the developer experience uniform across all input types."
    ),
    "Why does FastAPI use Annotated for declaring dependencies and parameters in newer versions?": (
        "FastAPI adopted Annotated (from typing) as the preferred way to declare parameters "
        "and dependencies because it separates the type annotation from the metadata. With "
        "Annotated[str, Query(min_length=3)], the type (str) and the FastAPI-specific metadata "
        "(Query constraint) are cleanly separated. This allows better static type checker "
        "support, since tools like mypy see the plain type while FastAPI reads the metadata. "
        "It also enables reusing the same Annotated type alias across multiple endpoints."
    ),
    "What motivated the design of HTTPException with headers support in FastAPI?": (
        "FastAPI's HTTPException includes a headers parameter to allow raising HTTP errors that "
        "carry additional response headers. This is required for correct implementation of "
        "authentication challenges: for example, a 401 Unauthorized response must include a "
        "WWW-Authenticate header per the HTTP specification. Without headers support in the "
        "exception itself, developers would have to catch the exception and manually add headers "
        "in exception handlers. The design keeps the error-raising code clean and co-located "
        "with the authentication logic."
    ),
    "Why does FastAPI expose the underlying Request object rather than abstracting it away?": (
        "FastAPI exposes the underlying Starlette Request object as an injectable parameter "
        "because there are always cases where the higher-level abstractions (path parameters, "
        "query params, body) are insufficient. Accessing raw headers, cookies, the client IP, "
        "streaming request body, or custom attributes requires the full Request object. By "
        "making Request injectable as a function parameter, FastAPI gives developers a clean "
        "escape hatch without breaking the rest of the type-hint-driven parameter system."
    ),
    "What was the rationale for FastAPI's approach to WebSocket support and its design choices?": (
        "FastAPI provides WebSocket support by inheriting Starlette's WebSocket handling and "
        "exposing it with the same decorator pattern used for HTTP routes (@app.websocket). "
        "WebSocket connections are handled as async path operations that receive a WebSocket "
        "object, on which developers can call accept(), send_text(), receive_text(), and close(). "
        "FastAPI's dependency injection system works with WebSocket routes too, so authentication "
        "and shared dependencies can be reused across HTTP and WebSocket endpoints consistently."
    ),
}


def main():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    for entry in data:
        question = entry.get("question", "")
        if question in GROUND_TRUTHS:
            entry["ground_truth"] = GROUND_TRUTHS[question]
            updated += 1
        else:
            print(f"  ⚠️  No ground truth for: {question[:60]}...")

    BENCHMARK_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"✅ Updated {updated}/{len(data)} entries in {BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
